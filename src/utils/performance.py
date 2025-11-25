import time
import json
import os
from datetime import datetime


class PerformanceMetrics:
    
    def __init__(self, algorithm_name, heuristic_name=None):
        self.algorithm_name = algorithm_name
        self.heuristic_name = heuristic_name
        self.execution_time = 0
        self.nodes_explored = 0
        self.path_length = 0
        self.path_cost = 0
        self.path_found = False
        self.grid_size = 0
        self.obstacle_density = 0
        
    def to_dict(self):
        return {
            'algorithm': self.algorithm_name,
            'heuristic': self.heuristic_name,
            'execution_time': self.execution_time,
            'nodes_explored': self.nodes_explored,
            'path_length': self.path_length,
            'path_cost': self.path_cost,
            'path_found': self.path_found,
            'grid_size': self.grid_size,
            'obstacle_density': self.obstacle_density
        }
    
    def __str__(self):
        result = f"\n===== {self.algorithm_name}"
        if self.heuristic_name:
            result += f" ({self.heuristic_name})"
        result += " =====\n"
        result += f"Path found: {self.path_found}\n"
        result += f"Execution time: {self.execution_time:.4f} seconds\n"
        result += f"Nodes explored: {self.nodes_explored}\n"
        if self.path_found:
            result += f"Path length: {self.path_length}\n"
            result += f"Path cost: {self.path_cost}\n"
        result += "=" * 50 + "\n"
        return result


def evaluate_algorithm(algorithm_func, grid, start, goal, algorithm_name, 
                       heuristic=None, heuristic_name=None):
    from utils.path_utils import reconstruct_path
    
    metrics = PerformanceMetrics(algorithm_name, heuristic_name)
    metrics.grid_size = grid.shape[0]
    metrics.obstacle_density = (grid == 1).sum() / grid.size
    
    start_time = time.time()
    
    if heuristic:
        parent, visited, found = algorithm_func(grid, start, goal, heuristic)
    else:
        parent, visited, found = algorithm_func(grid, start, goal)
    
    end_time = time.time()
    
    metrics.execution_time = end_time - start_time
    metrics.nodes_explored = len(visited)
    metrics.path_found = found
    
    if found:
        path = reconstruct_path(parent, start, goal)
        metrics.path_length = len(path)
        metrics.path_cost = len(path) - 1
    
    return metrics, parent, visited, found


def compare_algorithms(algorithms, grid, start, goal):
    results = []
    
    for algo_data in algorithms:
        if len(algo_data) == 2:
            func, name = algo_data
            heuristic, heuristic_name = None, None
        else:
            func, name, heuristic, heuristic_name = algo_data
        
        metrics, _, _, _ = evaluate_algorithm(
            func, grid, start, goal, name, heuristic, heuristic_name
        )
        results.append(metrics)
        print(metrics)
    
    return results


def save_results(results, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'results': [r.to_dict() for r in results]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to {filepath}")


def load_results(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data['results']
