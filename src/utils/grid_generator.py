import numpy as np

def generate_grid(size, obstacle_prob):
    grid = np.random.choice([0,1], (size,size), p=[1-obstacle_prob, obstacle_prob])
    grid[0,0], grid[-1,-1] = 0, 0
    return grid
