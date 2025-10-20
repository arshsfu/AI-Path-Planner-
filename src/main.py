from utils.grid_generator import generate_grid
from algorithms.bfs import bfs
from visualization.visualize import animate_bfs

def main():
    grid = generate_grid(size=20, obstacle_prob=0.25)
    start, goal = (0, 0), (19, 19)
    animate_bfs(grid, start, goal, bfs)

if __name__ == "__main__":
    main()
