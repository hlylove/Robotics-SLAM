import matplotlib.pyplot as plt
import numpy as np

grid_size = 10

obstacles = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), 
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
    (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9),
    (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8),
    (3, 2), (4, 2), (5, 2), (6, 2),
    (4, 4), (4, 5), (4, 6), (4, 7),
    (5, 7),
    (7, 4), (7, 5),
]

start = (3, 6)
goal = (8, 1)

annotations = [(0, 0), (3, 6), (3, 2), (5, 7), (4, 4), (7, 5), (8, 1)]

fig, ax = plt.subplots(figsize=(8, 8))

for x in range(grid_size):
    for y in range(grid_size):
        rect = plt.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='white')
        ax.add_patch(rect)

for (x, y) in obstacles:
    rect = plt.Rectangle((x, y), 1, 1, facecolor='lightgray')
    ax.add_patch(rect)

sx, sy = start
rect = plt.Rectangle((sx, sy), 1, 1, facecolor='dodgerblue')
ax.add_patch(rect)

gx, gy = goal
rect = plt.Rectangle((gx, gy), 1, 1, facecolor='limegreen')
ax.add_patch(rect)

for (x, y) in annotations:
    ax.text(x + 0.1, y + 0.4, f"({x},{y})", fontsize=10, weight='bold')

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_xticks(np.arange(0, grid_size+1, 1))
ax.set_yticks(np.arange(0, grid_size+1, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_aspect('equal')
ax.grid(True)
plt.title("Grid World Environment")
plt.show()