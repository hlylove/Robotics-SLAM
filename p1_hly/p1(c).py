import numpy as np
import matplotlib.pyplot as plt

W, H        = 10, 10
gamma       = 0.9
action_cost = -1
goal_reward = 10
obs_pen     = -100

start = (3, 6)
goal  = (8, 1)

internal_obs = {(3,2), (4,2), (5,2), (6,2), (4,4), (4,5), (4,6), (4,7), (5,7), (7,4), (7,5)}
border_obs   = {(x, 0) for x in range(W)} | {(x, H-1) for x in range(W)} | \
               {(0, y) for y in range(H)} | {(W-1, y) for y in range(H)}
obstacles    = internal_obs | border_obs

DIRS = {'N': ( 0,  1),
        'S': ( 0, -1),
        'E': ( 1,  0),
        'W': (-1,  0)}
A = list(DIRS.keys())           # ['N','S','E','W']

TRANS = {
    'N': [('N',0.7), ('E',0.1), ('W',0.1), ('STAY',0.1)],
    'S': [('S',0.7), ('E',0.1), ('W',0.1), ('STAY',0.1)],
    'E': [('E',0.7), ('N',0.1), ('S',0.1), ('STAY',0.1)],
    'W': [('W',0.7), ('N',0.1), ('S',0.1), ('STAY',0.1)]
}

state2idx, idx2state = {}, []
for y in range(H):
    for x in range(W):
        state2idx[(x,y)] = len(idx2state)
        idx2state.append((x,y))
S = len(idx2state)

def in_bounds(x,y): return 0<=x<W and 0<=y<H

# ---------- Transition_List ----------
def transition_list(x, y, a):
    """Return list of (next_idx, prob, reward) for taking action a from (x, y)."""
    s_idx = state2idx[(x, y)]

    if (x, y) == goal:
        return [(s_idx, 1.0, 0.0)]

    if (x, y) in obstacles:
        return [(s_idx, 1.0, obs_pen)]

    lst = []
    for dir_tag, p in TRANS[a]:
        if dir_tag == 'STAY':
            nx, ny = x, y
        else:
            dx, dy = DIRS[dir_tag]
            nx, ny = x + dx, y + dy

        if not in_bounds(nx, ny) or (nx, ny) in obstacles:
            lst.append((state2idx[(nx, ny)], p, action_cost + obs_pen))  # -101
            continue

        # 5) usual / goal
        reward = action_cost                       # -1
        if (nx, ny) == goal:
            reward += goal_reward                  # +10
        lst.append((state2idx[(nx, ny)], p, reward))

    return lst

# ---------- Policy Evaluation ----------
def policy_evaluation(policy, theta=1e-8):
    """Input policy: dict state->action；Return J(np.array)"""
    J = np.zeros(S)
    while True:
        delta = 0.0
        for s_idx,(x,y) in enumerate(idx2state):
            a = policy[(x,y)]
            v_old = J[s_idx]
            J[s_idx] = sum(p*(r + gamma*J[sp]) for sp,p,r in transition_list(x,y,a))
            delta = max(delta, abs(v_old - J[s_idx]))
        if delta < theta:
            return J

# ---------- Policy Improvement ----------
def policy_improvement(J, policy):
    policy_stable = True
    for (x,y) in policy.keys():
        if (x,y) in obstacles or (x,y)==goal:
            continue
        old_a = policy[(x,y)]

        Q = {}
        for a in A:
            Q[a] = sum(p*(r + gamma*J[sp]) for sp,p,r in transition_list(x,y,a))

        best_a = max(Q, key=Q.get)
        policy[(x,y)] = best_a
        if best_a != old_a:
            policy_stable = False
    return policy_stable

# ---------- Initial Policy (East) ----------
policy = {(x,y): 'E' for (x,y) in idx2state}

# ---------- Visualization ----------
def draw_policy_and_value(J, policy, title):
    Jgrid = np.zeros((H,W))
    Ux, Uy = np.zeros((H,W)), np.zeros((H,W))

    for (x,y), idx in state2idx.items():
        row = H-1-y 
        col = x
        Jgrid[row,col] = J[idx]

        if (x,y) in obstacles or (x,y)==goal:
            continue
        a = policy[(x,y)]
        dx,dy = DIRS[a]
        Ux[row,col], Uy[row,col] = dx, dy

    plt.figure(figsize=(6,6))
    im = plt.imshow(Jgrid, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='Value  J(x)')
    plt.quiver(np.arange(W), np.arange(H),
               Ux, Uy, pivot='middle', color='white', scale=20, width=0.005)
    plt.title(title)
    plt.xticks(range(W)); plt.yticks(range(H))
    plt.gca().set_yticks(np.arange(H))
    plt.gca().set_yticklabels(np.flip(range(H)))
    plt.xlabel('x'); plt.ylabel('y')
    plt.grid(color='w', ls='--', lw=0.4, alpha=0.3)
    plt.show()

# ---------- Iterations ----------
k = 0
plots_to_show = 4
while True:
    J = policy_evaluation(policy)

    if 0 < k <= plots_to_show:
        draw_policy_and_value(J, policy, f'Iteration {k}:  π^{({k})}')

    stable = policy_improvement(J, policy)

    k += 1
    if stable:               
        break

print(f'Converged after {k-1} improvements (k={k-1})')

# ---------- 9. optimal ----------
draw_policy_and_value(J, policy, 'Optimal Policy  π*  and  J*')
