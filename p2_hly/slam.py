import os, sys, pickle, math
from copy import deepcopy
import cv2
import skimage.draw

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20.0, 20.0
        s.ymin, s.ymax = -20.0, 20.0
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        dx = x - s.xmin
        dy = y - s.ymin

        x_idx = (dx / s.resolution).astype(int)
        y_idx = (dy / s.resolution).astype(int)

        x_idx = np.clip(x_idx, 0, s.szx - 1)
        y_idx = np.clip(y_idx, 0, s.szy - 1)

        return np.vstack((x_idx, y_idx))
    
class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = 1e-8*np.eye(3)

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX
        n = w.shape[0]
        positions = (np.arange(n) + np.random.uniform(0, 1, n)) / n  # stratified points

        # Compute cumulative sum of weights
        cumulative_sum = np.cumsum(w)
        cumulative_sum[-1] = 1.0  # Ensure numerical precision

        # Resample indices
        idx = np.zeros(n, dtype=int)
        i, j = 0, 0
        while i < n:
            if positions[i] < cumulative_sum[j]:
                idx[i] = j
                i += 1
            else:
                j += 1

        # Resample particles
        p_new = p[:, idx]
        w_new = np.ones(n) / n  # uniform weights after resampling

        return p_new, w_new

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        #### TODO: XXXXXXXXXXX

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data

        # 1. from lidar distances to points in the LiDAR frame

        # 2. from LiDAR frame to the body frame

        # 3. from body frame to world frame

        if angles is None:
            angles = s.lidar_angles

        # Clip to LiDAR range
        d = np.clip(d, s.lidar_dmin, s.lidar_dmax)

        # Step 1: LiDAR frame to points in LiDAR frame
        x_lidar = d * np.cos(angles)
        y_lidar = d * np.sin(angles)
        lidar_pts = np.vstack((x_lidar, y_lidar, np.zeros_like(d), np.ones_like(d)))  # shape: (4, N)

        # Step 2: LiDAR to body (neck + head transform)
        T_lidar_to_body = euler_to_se3(0, head_angle, neck_angle, np.array([0, 0, s.lidar_height]))

        # Step 3: Body to world (via particle pose p)
        T_body_to_world = euler_to_se3(0, 0, p[2], np.array([p[0], p[1], s.head_height]))

        # Final transformation
        points_world = T_body_to_world @ T_lidar_to_body @ lidar_pts  # (4, N)

        return points_world[:2]  # return (2, N)

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        #### TODO: XXXXXXXXXXX
        control = smart_minus_2d(s.lidar[t]['xyth'], s.lidar[t - 1]['xyth'])
        return control

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        #### TODO: XXXXXXXXXXX
        control = s.get_control(t) # Get control input (delta_x, delta_y, delta_theta)
        for i in range(s.p.shape[1]):
            noise = np.random.multivariate_normal([0, 0, 0], s.Q)
            noisy_control = control + noise
            s.p[:, i] = smart_plus_2d(s.p[:, i], noisy_control)

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        # Compute unnormalized log-weights: log(w_i * p(z | x_i)) = log(w_i) + logp
        # log_weights = np.log(w) + obs_logp

        # # Normalize in log space for numerical stability
        # log_weights -= slam_t.log_sum_exp(log_weights)

        # # Convert to linear space
        # weights = np.exp(log_weights)
        eps = 1e-12  # prevent log(0)
        log_weights = np.log(w + eps) + obs_logp
        log_weights -= slam_t.log_sum_exp(log_weights)
        new_weights = np.exp(log_weights)
        return new_weights

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX
        # 1. Get joint angles for head & neck
        joint_t = s.find_joint_t_idx_from_lidar(s.lidar[t]['t'])
        angles = s.joint['head_angles']
        neck_angle = angles[joint_name_to_index['Neck'], joint_t]
        head_angle = angles[joint_name_to_index['Head'], joint_t]

        # 2. Get LiDAR scan at time t
        scan = s.lidar[t]['scan']

        # 3. Compute log-likelihood for each particle
        log_likelihoods = np.zeros(s.n)
        for i in range(s.n):
            pose = s.p[:, i]
            world_points = s.rays2world(pose, scan, head_angle, neck_angle, s.lidar_angles)
            x_idx, y_idx = s.map.grid_cell_from_xy(world_points[0], world_points[1])
            valid = (x_idx >= 0) & (x_idx < s.map.szx) & (y_idx >= 0) & (y_idx < s.map.szy)
            log_likelihoods[i] = np.sum(s.map.log_odds[x_idx[valid], y_idx[valid]])

        # 4. Update weights
        s.w = s.update_weights(s.w, log_likelihoods)

        # 5. Mapping: update map using highest-weight particle
        best_idx = np.argmax(s.w)
        best_pose = s.p[:, best_idx]
        hit_pts = s.rays2world(best_pose, scan, head_angle, neck_angle, s.lidar_angles)
        hit_cells = s.map.grid_cell_from_xy(hit_pts[0], hit_pts[1])

        # 6. Get particle position as ray start point
        px, py = best_pose[0], best_pose[1]
        start_cell = s.map.grid_cell_from_xy(np.array([px]), np.array([py]))
        x0, y0 = start_cell[0][0], start_cell[1][0]

        # 7. Update free space by ray tracing
        free_mask = np.zeros_like(s.map.cells)
        for i in range(hit_cells.shape[1]):
            x1, y1 = hit_cells[0][i], hit_cells[1][i]
            rr, cc = skimage.draw.line(x0, y0, x1, y1)
            rr = np.clip(rr, 0, s.map.szx - 1)
            cc = np.clip(cc, 0, s.map.szy - 1)
            free_mask[rr, cc] = 1  # mark as free space

        # 8. Apply log-odds updates
        s.map.log_odds[hit_cells[0], hit_cells[1]] += s.lidar_log_odds_occ - s.lidar_log_odds_free
        s.map.log_odds[free_mask == 1] += s.lidar_log_odds_free

        # 9. Clip and update cells
        s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)
        s.map.cells = (s.map.log_odds >= s.map.log_odds_thresh).astype(np.uint8)

        # 10. Update number of observations for free cells
        s.map.num_obs_per_cell[free_mask == 1] += 1

        s.resample_particles()

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')