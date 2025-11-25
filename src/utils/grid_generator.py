import numpy as np
import os
import json

def generate_grid(size, obstacle_prob, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    grid = np.random.choice([0, 1], (size, size), p=[1-obstacle_prob, obstacle_prob])
    grid[0, 0], grid[-1, -1] = 0, 0
    return grid


def save_grid(grid, filepath, metadata=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data = {
        'grid': grid.tolist(),
        'shape': grid.shape,
        'metadata': metadata or {}
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Grid saved to {filepath}")


def load_grid(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    grid = np.array(data['grid'])
    metadata = data.get('metadata', {})
    
    print(f"Grid loaded from {filepath}")
    return grid, metadata


def generate_maze_grid(size):
    grid = np.zeros((size, size), dtype=int)
    
    def divide(x, y, width, height, horizontal):
        if width < 2 or height < 2:
            return
        
        if horizontal:
            wall_y = y + np.random.randint(0, height)
            gap_x = x + np.random.randint(0, width)
            for i in range(x, x + width):
                if i != gap_x:
                    grid[wall_y, i] = 1
            
            divide(x, y, width, wall_y - y, not horizontal)
            divide(x, wall_y + 1, width, y + height - wall_y - 1, not horizontal)
        else:
            wall_x = x + np.random.randint(0, width)
            gap_y = y + np.random.randint(0, height)
            for i in range(y, y + height):
                if i != gap_y:
                    grid[i, wall_x] = 1
            
            divide(x, y, wall_x - x, height, not horizontal)
            divide(wall_x + 1, y, x + width - wall_x - 1, height, not horizontal)
    
    horizontal = size > size
    divide(0, 0, size, size, horizontal)
    
    grid[0, 0], grid[-1, -1] = 0, 0
    
    return grid
