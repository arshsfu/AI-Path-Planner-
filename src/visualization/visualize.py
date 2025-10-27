import matplotlib.pyplot as plt
import numpy as np
from utils.path_utils import reconstruct_path

def show_grid(grid, path=None, explored=None, start=None, goal=None):
    plt.figure(figsize=(7,7))
    
    plt.imshow(grid, cmap="Greys", origin="upper")

    if explored and isinstance(explored, (list, set)):
        x, y = zip(*explored)
        plt.scatter(y, x, c='deepskyblue', s=10, label='Explored', alpha=0.6)

    if path and isinstance(path, (list, set)):
        x, y = zip(*path)
        plt.plot(y, x, c='gold', linewidth=3, label='Path')

    if start:
        plt.scatter(start[1], start[0], c='lime', s=120, marker='o', edgecolors='black', label='Start')

    if goal:
        plt.scatter(goal[1], goal[0], c='red', s=120, marker='X', edgecolors='black', label='Goal')

    plt.xticks(np.arange(0, grid.shape[1], 1))
    plt.yticks(np.arange(0, grid.shape[0], 1))
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title("AI Path Planning Grid")
    plt.xlabel("Y-axis (columns)")
    plt.ylabel("X-axis (rows)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def animate_bfs(grid, start, goal, bfs_function):
    plt.ion()
    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(grid, cmap="Greys", origin="upper")

    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    ax.set_xticks(np.arange(0, grid.shape[1], 1))
    ax.set_yticks(np.arange(0, grid.shape[0], 1))
    ax.set_xticklabels(range(grid.shape[1]))
    ax.set_yticklabels(range(grid.shape[0]))

    plt.title("BFS Pathfinding Visualization")
    plt.xlabel("Y (columns)")
    plt.ylabel("X (rows)")

    ax.scatter(start[1], start[0], c='lime', s=120, marker='o', edgecolors='black', label='Start')
    ax.scatter(goal[1], goal[0], c='red', s=120, marker='X', edgecolors='black', label='Goal')
    plt.legend(loc='upper right')

    explored_scatter = None

    def update_visual(explored):
        nonlocal explored_scatter
        if explored_scatter:
            explored_scatter.remove()
        x, y = zip(*explored)
        explored_scatter = ax.scatter(y, x, c='deepskyblue', s=10, alpha=0.6)
        plt.pause(0.001)

    parent, visited, found = bfs_function(grid, start, goal, visualize_step=update_visual)

    if found:
        path = reconstruct_path(parent, start, goal)
        px, py = zip(*path)
        ax.plot(py, px, c='gold', linewidth=3, label='Path')
        plt.title("Path Found")
        plt.pause(1)
    else:
        plt.title("No Path Found!")
        ax.text(len(grid)//2 - 3, len(grid)//2, "NO PATH!", fontsize=18, color='red', weight='bold')
        plt.pause(2)

    plt.ioff()
    plt.show()
    print("\n===== BFS RESULTS =====")
    print(f"Path found: {found}")
    print(f"Nodes explored: {len(visited)}")
    if found:
        print(f"Path cost: {len(reconstruct_path(parent, start, goal)) - 1}")
    else:
        print("No valid path could be found.")
    print("========================\n")
