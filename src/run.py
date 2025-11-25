from utils.grid_generator import generate_grid, save_grid
from algorithms.bfs import bfs
from algorithms.ucs import ucs
from algorithms.astar import astar
from utils.heuristics import manhattan_distance, euclidean_distance
from utils.performance import PerformanceMetrics, save_results
from visualization.visualize import (
    animate_all_algorithms,
    plot_performance_comparison,
    plot_heuristic_comparison
)
from utils.path_utils import reconstruct_path
import os
from datetime import datetime


def main():
    print("=" * 70)
    print("SIMULTANEOUS ALGORITHM ANIMATION")
    print("Watch BFS, UCS, A*(Manhattan), and A*(Euclidean) explore together!")
    print("=" * 70)
    
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/sample_maps", exist_ok=True)
    
    grid_size = 20
    obstacle_prob = 0.25
    start = (0, 0)
    goal = (grid_size - 1, grid_size - 1)
    
    print(f"\nGenerating {grid_size}x{grid_size} grid...")
    grid = generate_grid(grid_size, obstacle_prob, seed=42)
    
    print(f"Start: {start}, Goal: {goal}")
    print(f"Obstacles: {(grid == 1).sum()} / {grid.size} cells ({obstacle_prob*100}%)")
    
    print("\n" + "=" * 70)
    print("ANIMATION STARTING...")
    print("=" * 70)
    print("\nAll 4 algorithms will explore simultaneously!")
    print("Watch the blue dots expand in each panel:")
    print("  - Top-Left: BFS (uninformed, explores uniformly)")
    print("  - Top-Right: UCS (cost-based, explores uniformly)")
    print("  - Bottom-Left: A* Manhattan (smart, targets goal)")
    print("  - Bottom-Right: A* Euclidean (smart, less informed)")
    print("\nNotice how A* explores fewer nodes!")
    print("=" * 70 + "\n")
    
    input("Press Enter to start animation...")
    
    algorithms_list = [
        (bfs, "BFS", None, None),
        (ucs, "UCS", None, None),
        (astar, "A*", manhattan_distance, "Manhattan"),
        (astar, "A*", euclidean_distance, "Euclidean"),
    ]
    
    delay = 0.02
    results = animate_all_algorithms(grid, start, goal, algorithms_list, delay=delay)
    
    print("\n" + "=" * 70)
    print("CREATING PERFORMANCE METRICS")
    print("=" * 70 + "\n")
    
    metrics_list = []
    for full_name, parent, visited, found, exec_time in results:
        # Parse algorithm name and heuristic
        if "(" in full_name:
            algo_name = full_name.split("(")[0].strip()
            heuristic_name = full_name.split("(")[1].rstrip(")")
        else:
            algo_name = full_name
            heuristic_name = None
        
        # Create metrics
        metrics = PerformanceMetrics(algo_name, heuristic_name)
        metrics.execution_time = exec_time
        metrics.nodes_explored = len(visited)
        metrics.path_found = found
        metrics.grid_size = grid_size
        metrics.obstacle_density = (grid == 1).sum() / grid.size
        
        if found:
            path = reconstruct_path(parent, start, goal)
            metrics.path_length = len(path)
            metrics.path_cost = len(path) - 1
        
        metrics_list.append(metrics)
        print(metrics)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_path = f"results/logs/simultaneous_{timestamp}.json"
    save_results(metrics_list, results_path)
    print(f"Results saved to {results_path}")
    
    # Generate comparison graphs
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON GRAPHS")
    print("=" * 70 + "\n")
    
    # Performance comparison
    print("Creating 3-panel performance comparison...")
    perf_chart_path = f"results/figures/performance_{timestamp}.png"
    plot_performance_comparison(metrics_list, perf_chart_path)
    
    # Heuristic comparison
    print("Creating A* heuristic comparison...")
    astar_metrics = [m for m in metrics_list if m.algorithm_name == "A*"]
    if len(astar_metrics) >= 2:
        heuristic_chart_path = f"results/figures/heuristics_{timestamp}.png"
        plot_heuristic_comparison(astar_metrics, heuristic_chart_path)
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    most_efficient = min(metrics_list, key=lambda x: x.nodes_explored)
    fastest = min(metrics_list, key=lambda x: x.execution_time)
    
    print(f"\nPerformance Winners:")
    print(f"  - Most Efficient (fewest nodes):")
    print(f"    > {most_efficient.algorithm_name}"
          + (f" ({most_efficient.heuristic_name})" if most_efficient.heuristic_name else "")
          + f" - {most_efficient.nodes_explored} nodes")
    
    print(f"\n  - Fastest Execution:")
    print(f"    > {fastest.algorithm_name}"
          + (f" ({fastest.heuristic_name})" if fastest.heuristic_name else "")
          + f" - {fastest.execution_time:.2f} seconds")
    
    if all(m.path_found for m in metrics_list):
        print(f"\n  - All algorithms found optimal path!")
        print(f"    > Path length: {metrics_list[0].path_length} nodes")
        print(f"    > Path cost: {metrics_list[0].path_cost} moves")
    
    bfs_nodes = next(m.nodes_explored for m in metrics_list if m.algorithm_name == "BFS")
    astar_man_nodes = next(m.nodes_explored for m in metrics_list 
                          if m.algorithm_name == "A*" and m.heuristic_name == "Manhattan")
    improvement = ((bfs_nodes - astar_man_nodes) / bfs_nodes) * 100
    
    print(f"\nKey Insight:")
    print(f"  A* (Manhattan) explored {improvement:.1f}% fewer nodes than BFS!")
    print(f"  ({astar_man_nodes} vs {bfs_nodes} nodes)")
    
    print(f"\nFiles saved:")
    print(f"  - Metrics: {results_path}")
    print(f"  - Performance chart: {perf_chart_path}")
    if len(astar_metrics) >= 2:
        print(f"  - Heuristic chart: {heuristic_chart_path}")
    
    print("\n" + "=" * 70)
    print("Complete! Check the charts for visual comparison.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
