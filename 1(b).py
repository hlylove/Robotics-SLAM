import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 动作编码
ACTIONS = {'N': 0, 'S': 1, 'E': 2, 'W': 3}
action_list = ['N', 'S', 'E', 'W']
NUM_ACTIONS = 4
GRID_SIZE = 10
NUM_STATES = GRID_SIZE * GRID_SIZE

# 每个动作的方向向量 (dy, dx)
DIRECTION = {
    0: (-1, 0),  # N
    1: (1, 0),   # S
    2: (0, -1),   # E
    3: (0, 1)   # W
}

# 用于将 (i, j) 转为状态编号
def pos_to_index(i, j):
    return i * GRID_SIZE + j

# 将状态编号转为 (i, j)
def index_to_pos(index):
    return divmod(index, GRID_SIZE)

# 定义障碍位置 (y, x)
obstacles_imgs = set([
    (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9),
    (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0), (9,0),
    (1,9), (2,9), (3,9), (4,9), (5,9), (6,9), (7,9), (8,9), (9,9),
    (9,1), (9,2), (9,3), (9,4), (9,5), (9,6), (9,7), (9,8),
    (3,2), (4,2), (5,2), (6,2),
    (4,4), (4,5), (4,6), (4,7),
    (5,7),
    (7,4), (7,5)
])

obstacles = [(GRID_SIZE - 1 - y, x) for (x, y) in obstacles_imgs]

# 图像坐标下目标位置（左下是 (0,0)）
goal_img = (8, 1)  # 图像上右下位置

# 转换成 row-major 坐标
def img_to_row_major(y_img, x_img):
    return GRID_SIZE - 1 - y_img, x_img

goal = img_to_row_major(*goal_img)  # → (1, 1)
goal_index = pos_to_index(*goal) 

gamma = 0.9

T = np.zeros((NUM_STATES, NUM_STATES, NUM_ACTIONS))
R = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))

# 每个动作的转移结构
ACTION_PROBS = {
    0: [ (0.7, 0), (0.1, 3), (0.1, 2), (0.1, -1) ],  # N
    1: [ (0.7, 1), (0.1, 3), (0.1, 2), (0.1, -1) ],  # S
    2: [ (0.7, 2), (0.1, 0), (0.1, 1), (0.1, -1) ],  # E
    3: [ (0.7, 3), (0.1, 0), (0.1, 1), (0.1, -1) ]   # W
}

for s in range(NUM_STATES):
    y, x = index_to_pos(s)
    if (y, x) in obstacles:
        T[s, s, :] = 1.0
        R[s, :, s] = -100
        continue

    for a in range(NUM_ACTIONS):
        for prob, direction in ACTION_PROBS[a]:
            if direction == -1:
                ny, nx = y, x
            else:
                dy, dx = DIRECTION[direction]
                ny, nx = y + dy, x + dx

            if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
                sp = pos_to_index(ny, nx)
                if (ny, nx) in obstacles:
                    sp = pos_to_index(ny, nx)
                    reward = -100
                elif sp == goal_index and s != goal_index:
                    reward = 10
                else:
                    reward = -1
            else:
                sp = s
                reward = -100

            T[s, sp, a] += prob
            R[s, a, sp] = reward

for a in range(NUM_ACTIONS):
    T[goal_index, :, a] = 0
    T[goal_index, goal_index, a] = 1
    R[goal_index, a, :] = 0

# 测试
for name, pos in [('7,4', (7,4))]:
    s = pos_to_index(*pos)
    print(f"Transitions from {name}")
    for sp in range(NUM_STATES):
            if T[s, sp, ACTIONS['E']] > 0:
                print(f"  -> {index_to_pos(sp)}: prob={T[s, sp, ACTIONS['E']]:.2f}, reward={R[s, ACTIONS['E'], sp]}")

# π(0): 所有状态下采取动作 East (2)
policy = np.full(NUM_STATES, ACTIONS['E'])

# 策略评估
J = np.zeros(NUM_STATES)
max_iter = 1000
tol = 1e-6

for it in range(max_iter):
    J_new = np.zeros_like(J)
    for s in range(NUM_STATES):
        a = policy[s]
        immediate_cost = 0
        for sp in range(NUM_STATES):
            immediate_cost += T[s, sp, a] * (-R[s, a, sp])
        J_new[s] = immediate_cost + gamma * np.dot(T[s, :, a], J)
    if np.max(np.abs(J_new - J)) < tol:
        break
    J = J_new

# 生成 J_grid：J[y, x]
J_grid = np.zeros((GRID_SIZE, GRID_SIZE))
for s in range(NUM_STATES):
    y, x = index_to_pos(s)
    J_grid[y, x] = J[s]

plt.figure(figsize=(10, 8))
ax = plt.gca()

# 直接用 J_grid，无翻转
im = ax.imshow(J_grid, cmap='viridis', interpolation='none')

# 添加 colorbar
plt.colorbar(im, ax=ax, label='Cost-to-Go')

# 坐标轴刻度
plt.xticks(ticks=np.arange(GRID_SIZE), labels=np.arange(GRID_SIZE))
plt.yticks(ticks=np.arange(GRID_SIZE), labels=np.arange(GRID_SIZE))

# 可选：添加格子线
ax.set_xticks(np.arange(GRID_SIZE) - 0.5, minor=True)
ax.set_yticks(np.arange(GRID_SIZE) - 0.5, minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

# 坐标轴标签
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Cost-to-Go Heatmap (Row-Major Coordinates)")

plt.tight_layout()
plt.show()

# # 逆时针旋转 90 度，使左下角为 (0,0)
# J_plot = np.rot90(J_grid)

# # 绘图
# plt.figure(figsize=(10, 8))

# # # 用 mask 遮掉障碍区（不参与 heatmap 显示）
# # mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
# # for y in range(GRID_SIZE):
# #     for x in range(GRID_SIZE):
# #         yy, xx = GRID_SIZE - 1 - x, y  # 旋转后坐标
# #         if (y, x) in obstacles:
# #             mask[yy, xx] = True

# # 显示 heatmap
# ax = sns.heatmap(
#     J_plot,
#     cmap="plasma_r",
#     # mask=mask,
#     square=True,
#     cbar_kws={"label": "Cost-to-Go"},
#     annot=False
# )

# # # 标注所有非零格子数值（带负号）
# # for y in range(GRID_SIZE):
# #     for x in range(GRID_SIZE):
# #         val = J_plot[y, x]
# #         if not mask[y, x]:
# #             ax.text(x + 0.5, y + 0.5, f"{-val:.2f}", ha='center', va='center', color='white', fontsize=8)

# # # 绘制障碍区为灰色方块
# # for (y, x) in obstacles:
# #     yy, xx = GRID_SIZE - 1 - x, y  # 旋转后
# #     ax.add_patch(plt.Rectangle((xx, yy), 1, 1, color='lightgray'))

# # 标注所有格子数值（带负号）
# for y in range(GRID_SIZE):
#     for x in range(GRID_SIZE):
#         val = J_plot[y, x]
#         ax.text(x + 0.5, y + 0.5, f"{-val:.2f}", ha='center', va='center', color='white', fontsize=8)

# # 标出起点和终点
# start = (3, 6)
# goal = (8, 1)
# start_rot = (GRID_SIZE - 1 - start[1], start[0])
# goal_rot = (GRID_SIZE - 1 - goal[1], goal[0])
# plt.plot(start_rot[1] + 0.5, start_rot[0] + 0.5, 'o', color='blue', label='Start')
# plt.plot(goal_rot[1] + 0.5, goal_rot[0] + 0.5, 'o', color='limegreen', label='Goal')

# # 显示终点 cost 数值
# val = J_plot[goal_rot[0], goal_rot[1]]
# ax.text(goal_rot[1] + 0.5, goal_rot[0] + 0.5, f"{-val:.2f}", ha='center', va='center', color='white', fontsize=8)

# # 坐标轴设置
# plt.xticks(ticks=np.arange(GRID_SIZE) + 0.5, labels=np.arange(GRID_SIZE))
# plt.yticks(
#     ticks=np.arange(GRID_SIZE) + 0.5,
#     labels=np.arange(GRID_SIZE)  # 从0到9，正序
# )
# plt.title("Cost-to-Go $J^{\pi^{(0)}}(x)$ with Obstacles Shown (Values Annotated)")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()