import matplotlib.pyplot as plt
import numpy as np
from utils.path_utils import reconstruct_path
import os

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

def animate_algorithm(grid, start, goal, algorithm_func, algorithm_name, 
                      heuristic=None, delay=0.05):
    """
    Generic animation function for any pathfinding algorithm.
    Shows real-time exploration with blue dots expanding outward.
    
    Args:
        grid: 2D numpy array
        start: start position tuple
        goal: goal position tuple
        algorithm_func: the algorithm function to animate
        algorithm_name: name of the algorithm for display
        heuristic: optional heuristic function for A*
        delay: delay between frames (default 0.05s)
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap="Greys", origin="upper", aspect='equal')
    
    # Set equal aspect ratio to make boxes square
    ax.set_aspect('equal', adjustable='box')

    # Grid styling - make gridlines more visible
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1.0, alpha=0.8)
    ax.tick_params(which="minor", size=0)
    
    ax.set_xticks(np.arange(0, grid.shape[1], 1))
    ax.set_yticks(np.arange(0, grid.shape[0], 1))
    ax.set_xticklabels(range(grid.shape[1]))
    ax.set_yticklabels(range(grid.shape[0]))

    plt.title(f"{algorithm_name} - Exploring...")
    plt.xlabel("Y (columns)")
    plt.ylabel("X (rows)")

    ax.scatter(start[1], start[0], c='lime', s=150, marker='o', 
              edgecolors='black', linewidths=2, label='Start', zorder=5)
    ax.scatter(goal[1], goal[0], c='red', s=150, marker='X', 
              edgecolors='black', linewidths=2, label='Goal', zorder=5)
    plt.legend(loc='upper right')

    explored_scatter = None
    node_count_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                             verticalalignment='top', fontsize=11,
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def update_visual(explored):
        nonlocal explored_scatter
        if explored_scatter:
            explored_scatter.remove()
        if explored:
            x, y = zip(*explored)
            explored_scatter = ax.scatter(y, x, c='deepskyblue', s=15, alpha=0.6)
            node_count_text.set_text(f'Nodes explored: {len(explored)}')
        plt.pause(0.001)

    # Run algorithm with visualization
    import time
    start_time = time.time()
    
    if heuristic:
        parent, visited, found = algorithm_func(grid, start, goal, heuristic, 
                                               visualize_step=update_visual, delay=delay)
    else:
        parent, visited, found = algorithm_func(grid, start, goal, 
                                               visualize_step=update_visual, delay=delay)
    
    execution_time = time.time() - start_time

    # Show final result
    if found:
        path = reconstruct_path(parent, start, goal)
        px, py = zip(*path)
        ax.plot(py, px, c='gold', linewidth=3, label='Path', zorder=4)
        plt.title(f"{algorithm_name} - Path Found!", color='green', fontweight='bold')
        node_count_text.set_text(f'Nodes: {len(visited)} | Path: {len(path)} | Time: {execution_time:.2f}s')
        plt.legend(loc='upper right')
        plt.pause(2)
    else:
        plt.title(f"{algorithm_name} - No Path Found!", color='red', fontweight='bold')
        ax.text(len(grid)//2 - 3, len(grid)//2, "NO PATH!", 
               fontsize=18, color='red', weight='bold')
        plt.pause(2)

    plt.ioff()
    plt.show()
    
    # Print results
    print(f"\n===== {algorithm_name} RESULTS =====")
    print(f"Path found: {found}")
    print(f"Nodes explored: {len(visited)}")
    print(f"Execution time: {execution_time:.2f}s")
    if found:
        print(f"Path length: {len(path)}")
        print(f"Path cost: {len(path) - 1}")
    print("=" * (len(algorithm_name) + 16) + "\n")
    
    return parent, visited, found


def animate_bfs(grid, start, goal, bfs_function):
    """Legacy BFS animation function - calls generic animator."""
    return animate_algorithm(grid, start, goal, bfs_function, "BFS", delay=0.05)


def animate_all_algorithms(grid, start, goal, algorithms_list, delay=0.03):
    """
    Animate all algorithms simultaneously in a 2x2 grid layout.
    Shows real-time exploration with expanding blue dots for all algorithms at once!
    
    Args:
        grid: 2D numpy array
        start: start position tuple
        goal: goal position tuple
        algorithms_list: list of tuples (func, name, heuristic, heuristic_name)
                        For non-A* algorithms, use (func, name, None, None)
        delay: animation delay between frames
    
    Returns:
        list of tuples (name, parent, visited, found, execution_time)
    """
    import time
    
    n_algorithms = len(algorithms_list)
    
    # Create 2x2 subplot layout with equal aspect ratio (square boxes)
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    plt.ion()
    
    # Initialize each subplot
    plot_objects = []
    for idx, (func, name, heuristic, heuristic_name) in enumerate(algorithms_list):
        ax = axes[idx]
        ax.imshow(grid, cmap="Greys", origin="upper", aspect='equal')
        
        # Set equal aspect ratio to make boxes square
        ax.set_aspect('equal', adjustable='box')
        
        # Grid styling - make gridlines more visible
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=1.0, alpha=0.8)
        ax.tick_params(which="minor", size=0)
        
        # Major ticks for labels
        ax.set_xticks(np.arange(0, grid.shape[1], 1))
        ax.set_yticks(np.arange(0, grid.shape[0], 1))
        
        # Start and goal markers
        ax.scatter(start[1], start[0], c='lime', s=150, marker='o', 
                  edgecolors='black', linewidths=2, label='Start', zorder=5)
        ax.scatter(goal[1], goal[0], c='red', s=150, marker='X', 
                  edgecolors='black', linewidths=2, label='Goal', zorder=5)
        
        # Title with heuristic name if applicable
        full_name = name if not heuristic_name else f"{name} ({heuristic_name})"
        ax.set_title(f"{full_name}\nExploring...", fontsize=12, fontweight='bold')
        ax.set_xlabel("Y (columns)")
        ax.set_ylabel("X (rows)")
        ax.legend(loc='upper right', fontsize=8)
        
        # Create text for node counter
        text = ax.text(0.02, 0.98, 'Nodes: 0', transform=ax.transAxes,
                      verticalalignment='top', fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plot_objects.append({
            'ax': ax,
            'scatter': None,
            'text': text,
            'name': full_name
        })
    
    plt.tight_layout()
    
    # Run all algorithms concurrently with synchronized visualization
    results = []
    algorithm_states = []
    
    print("\nStarting simultaneous animation of all algorithms...")
    print("Watch how different algorithms explore the space!\n")
    
    # Initialize algorithm states
    for idx, (func, name, heuristic, heuristic_name) in enumerate(algorithms_list):
        visited_so_far = set()
        algorithm_states.append({
            'func': func,
            'name': name,
            'heuristic': heuristic,
            'visited': visited_so_far,
            'parent': None,
            'found': False,
            'completed': False,
            'start_time': time.time()
        })
    
    # Function to update visualization for a specific algorithm
    def update_visual_for_algo(idx, visited):
        if plot_objects[idx]['scatter']:
            plot_objects[idx]['scatter'].remove()
        if visited:
            x, y = zip(*visited)
            plot_objects[idx]['scatter'] = plot_objects[idx]['ax'].scatter(
                y, x, c='deepskyblue', s=15, alpha=0.6, zorder=3)
            plot_objects[idx]['text'].set_text(f'Nodes: {len(visited)}')
    
    # Run each algorithm and collect results
    for idx, (func, name, heuristic, heuristic_name) in enumerate(algorithms_list):
        full_name = name if not heuristic_name else f"{name} ({heuristic_name})"
        start_time = time.time()
        
        # Create callback for this specific algorithm
        def make_callback(algorithm_idx):
            def callback(visited):
                update_visual_for_algo(algorithm_idx, visited)
                plt.pause(0.001)
            return callback
        
        # Run the algorithm
        if heuristic:
            parent, visited, found = func(grid, start, goal, heuristic,
                                         visualize_step=make_callback(idx), 
                                         delay=delay)
        else:
            parent, visited, found = func(grid, start, goal,
                                         visualize_step=make_callback(idx), 
                                         delay=delay)
        
        execution_time = time.time() - start_time
        
        # Draw final path if found
        ax = plot_objects[idx]['ax']
        if found:
            path = reconstruct_path(parent, start, goal)
            px, py = zip(*path)
            ax.plot(py, px, c='gold', linewidth=3, label='Path', zorder=4)
            ax.set_title(f"{full_name}\nPath Found ({len(visited)} nodes)", 
                        color='green', fontsize=12, fontweight='bold')
            plot_objects[idx]['text'].set_text(
                f'Nodes: {len(visited)} | Path: {len(path)} | Time: {execution_time:.2f}s')
        else:
            ax.set_title(f"{full_name}\nNo Path", 
                        color='red', fontsize=12, fontweight='bold')
        
        results.append((full_name, parent, visited, found, execution_time))
        print(f"  [OK] {full_name}: {len(visited)} nodes, {execution_time:.2f}s")
    
    plt.ioff()
    
    # Add overall title
    fig.suptitle('Algorithm Comparison - Side by Side Exploration', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.pause(2)
    plt.show()
    
    return results


def compare_algorithms_visual(grid, start, goal, results_data):
    """
    Create a visual comparison of multiple algorithms side by side.
    
    Args:
        grid: 2D numpy array
        start: start position tuple
        goal: goal position tuple
        results_data: list of tuples (algorithm_name, parent, visited, found)
    """
    n_algorithms = len(results_data)
    fig, axes = plt.subplots(1, n_algorithms, figsize=(6*n_algorithms, 6))
    
    if n_algorithms == 1:
        axes = [axes]
    
    for idx, (algo_name, parent, visited, found) in enumerate(results_data):
        ax = axes[idx]
        ax.imshow(grid, cmap="Greys", origin="upper")
        
        # Plot explored nodes
        if visited:
            x, y = zip(*visited)
            ax.scatter(y, x, c='deepskyblue', s=10, alpha=0.6, label='Explored')
        
        # Plot path if found
        if found:
            path = reconstruct_path(parent, start, goal)
            if path:
                px, py = zip(*path)
                ax.plot(py, px, c='gold', linewidth=3, label='Path')
        
        # Plot start and goal
        ax.scatter(start[1], start[0], c='lime', s=120, marker='o', 
                  edgecolors='black', label='Start', zorder=5)
        ax.scatter(goal[1], goal[0], c='red', s=120, marker='X', 
                  edgecolors='black', label='Goal', zorder=5)
        
        # Formatting
        ax.set_title(f"{algo_name}\nExplored: {len(visited)}")
        ax.set_xlabel("Y (columns)")
        ax.set_ylabel("X (rows)")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(metrics_list, save_path=None):
    """
    Create bar charts comparing algorithm performance metrics.
    
    Args:
        metrics_list: list of PerformanceMetrics objects
        save_path: optional path to save the figure
    """
    if not metrics_list:
        print("No metrics to plot")
        return
    
    # Prepare data
    names = []
    execution_times = []
    nodes_explored = []
    path_lengths = []
    
    for m in metrics_list:
        label = m.algorithm_name
        if m.heuristic_name:
            label += f"\n({m.heuristic_name})"
        names.append(label)
        execution_times.append(m.execution_time * 1000)  # Convert to ms
        nodes_explored.append(m.nodes_explored)
        path_lengths.append(m.path_length if m.path_found else 0)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Execution time
    axes[0].bar(names, execution_times, color='steelblue', alpha=0.7)
    axes[0].set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Time (milliseconds)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Nodes explored
    axes[1].bar(names, nodes_explored, color='coral', alpha=0.7)
    axes[1].set_title('Nodes Explored Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Number of Nodes', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Path length
    axes[2].bar(names, path_lengths, color='mediumseagreen', alpha=0.7)
    axes[2].set_title('Path Length Comparison', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Path Length', fontsize=12)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved to {save_path}")
    
    plt.show()


def plot_heuristic_comparison(astar_metrics_list, save_path=None):
    """
    Create a focused comparison of A* with different heuristics.
    
    Args:
        astar_metrics_list: list of PerformanceMetrics for A* with different heuristics
        save_path: optional path to save the figure
    """
    if not astar_metrics_list:
        print("No A* metrics to plot")
        return
    
    heuristics = [m.heuristic_name for m in astar_metrics_list]
    times = [m.execution_time * 1000 for m in astar_metrics_list]
    nodes = [m.nodes_explored for m in astar_metrics_list]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Execution time comparison
    ax1.bar(heuristics, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax1.set_title('A* Heuristic: Execution Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (milliseconds)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Nodes explored comparison
    ax2.bar(heuristics, nodes, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax2.set_title('A* Heuristic: Nodes Explored', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Nodes', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heuristic comparison saved to {save_path}")
    
    plt.show()

