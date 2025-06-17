import numpy as np
import matplotlib.pyplot as plt

W, H          = 10, 10        
gamma         = 0.9            
action_cost   = -1            
goal_reward   = 10       
obstacle_pen  = -100        

start = (3, 6)                 
goal  = (8, 1)                  

internal_obs = {(3,2), (4,2), (5,2), (6,2), (4,4), (4,5), (4,6), (4,7), (5,7), (7,4), (7,5)}

border_obs   = {(x, 0) for x in range(W)} | {(x, H-1) for x in range(W)} | \
               {(0, y) for y in range(H)} | {(W-1, y) for y in range(H)}

obstacles = internal_obs | border_obs

DIRS = {'E': ( 1,  0),
        'N': ( 0,  1),
        'S': ( 0, -1),
        'W': (-1,  0)}

east_transitions = [('E', 0.7), ('N', 0.1), ('S', 0.1), ('STAY', 0.1)]
west_transitions = [('W', 0.7), ('N', 0.1), ('S', 0.1), ('STAY', 0.1)]
north_transitions = [('N', 0.7), ('W', 0.1), ('E', 0.1), ('STAY', 0.1)]
south_transitions = [('S', 0.7), ('W', 0.1), ('E', 0.1), ('STAY', 0.1)]

state_to_idx, idx_to_state = {}, []
for y in range(H):
    for x in range(W):
        state_to_idx[(x, y)] = len(idx_to_state)
        idx_to_state.append((x, y))
S = len(idx_to_state)

def in_bounds(x, y):
    return 0 <= x < W and 0 <= y < H

P = [[] for _ in range(S)]

for s_idx, (x, y) in enumerate(idx_to_state):

    if (x, y) in obstacles:
        P[s_idx].append((s_idx, 1.0, obstacle_pen))
        continue

    if (x, y) == goal:
        P[s_idx].append((s_idx, 1.0, 0.0))
        continue

    for dir_tag, prob in east_transitions:

        if dir_tag == 'STAY':
            nx, ny = x, y
        else:
            dx, dy = DIRS[dir_tag]
            nx, ny = x + dx, y + dy

        if not in_bounds(nx, ny) or (nx, ny) in obstacles:
            # obstacle_pen
            next_idx = state_to_idx.get((nx, ny), s_idx) 
            P[s_idx].append((next_idx, prob, obstacle_pen))
        else:
            reward = action_cost
            if (nx, ny) == goal:
                reward += goal_reward
            P[s_idx].append((state_to_idx[(nx, ny)], prob, reward))

J = np.zeros(S)                
theta = 1e-8                    

while True:
    delta = 0.0
    for s in range(S):
        v = J[s]
        J[s] = sum(p * (r + gamma * J[sp]) for sp, p, r in P[s])
        delta = max(delta, abs(v - J[s]))
    if delta < theta:
        break

J_grid = np.zeros((H, W))
for (x, y), idx in state_to_idx.items():
    J_grid[H - 1 - y, x] = J[idx]   # H-1-y

plt.figure(figsize=(6, 6))
im = plt.imshow(J_grid, cmap='viridis', interpolation='nearest', vmin=np.min(J), vmax=np.max(J))
plt.colorbar(im, label='Cost‑to‑go  $J^{\\pi^{(0)}}(x)$')
plt.title('Value Function under Initial Policy (Always East)')
plt.xticks(range(W))
plt.yticks(range(H))
plt.gca().set_yticks(np.arange(H))
plt.gca().set_yticklabels(np.flip(range(H)))
plt.xlabel('x')
plt.ylabel('y')
plt.show()