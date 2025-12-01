"""
Interactive RL Demo - Click to Select Goals, Watch Agent Learn!

Combines interactive goal selection with reinforcement learning:
1. Click on grid to select goals
2. Watch RL agent train on your selected goals
3. Compare RL with A* on the same goals
"""

from use_cases.warehouse_robot import WarehouseRobot
from algorithms.goal_conditioned_qlearning import GoalConditionedQLearningAgent
from algorithms.astar import astar
from utils.heuristics import manhattan_distance
from utils.path_utils import reconstruct_path
import matplotlib.pyplot as plt
import numpy as np
import time


def main():
    print("\n" + "="*70)
    print(" INTERACTIVE RL vs A* DEMO ".center(70, "="))
    print("="*70)
    print("\nThis demo combines:")
    print("  1. Interactive goal selection (click on grid)")
    print("  2. RL agent training (learns from the goals)")
    print("  3. Side-by-side comparison (RL vs A*)")
    print("="*70)
    
    print("\n" + "="*70)
    print("STEP 1: SELECT YOUR GOALS")
    print("="*70)
    print("\nClick on the grid to select pickup locations...")
    print("(The RL agent will learn to navigate to YOUR chosen goals!)")
    input("\nPress Enter to start goal selection...")
    
    warehouse = WarehouseRobot(grid_size=40) 
    start = (0, 0) 
 
    goals = warehouse.select_goals_interactive(start, num_goals=4) 
    
    if len(goals) < 1:
        print("\nNo goals selected. Exiting.")
        return
    
    print("\n" + "="*70)
    print("STEP 2: TRAIN RL AGENT ON YOUR GOALS")
    print("="*70)
    print("\n✓ Using GOAL-CONDITIONED Q-Learning")
    print("  State = (position_x, position_y, goal_x, goal_y)")
    print(f"\nTraining on {len(goals)} segments (exact route):")
    print(f"\nTotal: {len(goals)} segments × 600 episodes = {len(goals) * 600} episodes")
    
    input("\nPress Enter to start training...")
    
    train_start = time.time()
    
    agent = GoalConditionedQLearningAgent(
        grid_size=warehouse.grid_size,
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print("\n" + "="*70)
    
    all_segments = [(start, goals[0])]
    for i in range(len(goals) - 1):
        all_segments.append((goals[i], goals[i+1]))
    
    print(f"Training segments (exact route only):")
    for i, (from_pos, to_pos) in enumerate(all_segments, 1):
        print(f"  {i}. {from_pos} → {to_pos}")
    print(f"\nTotal: {len(all_segments)} segments, 600 episodes each")
    print(f"Total episodes: {len(all_segments) * 600}")
    print("Note: More episodes = Better path quality (closer to A* optimality)")
    print("="*70)
    
    from collections import defaultdict
    self_training_rewards = []
    self_training_steps = []
    segment_success_rates = defaultdict(list)
    
    EPISODES_PER_SEGMENT = 600
    total_episodes = len(all_segments) * EPISODES_PER_SEGMENT
    episode_count = 0
    
    for episode in range(EPISODES_PER_SEGMENT):
        for seg_idx, (from_pos, to_pos) in enumerate(all_segments):
            episode_count += 1
            
            total_reward, steps, success = agent.train_episode(warehouse.grid, from_pos, to_pos)
            
            self_training_rewards.append(total_reward)
            self_training_steps.append(steps)
            segment_success_rates[seg_idx].append(1 if success else 0)
            
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            
            if episode_count % 100 == 0:
                recent_success = sum(1 for r in self_training_rewards[-100:] if r > 0) / min(100, len(self_training_rewards))
                avg_reward = np.mean(self_training_rewards[-100:])
                print(f"Episode {episode_count}/{total_episodes} | "
                      f"Success: {recent_success:.1%} | "
                      f"Avg Reward: {avg_reward:.1f} | "
                      f"ε: {agent.epsilon:.3f}")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    for seg_idx, (from_pos, to_pos) in enumerate(all_segments):
        success_rate = np.mean(segment_success_rates[seg_idx][-50:])
        print(f"Segment {from_pos} → {to_pos}: {success_rate:.1%} success")
    
    overall_success = sum(1 for r in self_training_rewards if r > 0) / len(self_training_rewards)
    print(f"\nOverall Success Rate: {overall_success:.1%}")
    print(f"Q-table size: {len(agent.q_table)} states")
    print(f"{'='*70}\n")
    
    # Create history for compatibility
    history = {
        'rewards': self_training_rewards,
        'steps': self_training_steps,
        'overall_success_rate': overall_success
    }
    
    train_time = time.time() - train_start
    
    print(f"\n✓ Training complete in {train_time:.2f}s")
    print(f"  Overall success rate: {history['overall_success_rate']:.1%}")
    print(f"  Q-table size: {len(agent.q_table)} states")
    
    print("\nGenerating learning curve...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    window = 25
    if len(history['rewards']) >= window:
        smoothed_rewards = np.convolve(history['rewards'], 
                                       np.ones(window)/window, mode='valid')
        plt.plot(smoothed_rewards, linewidth=2, color='blue')
    else:
        plt.plot(history['rewards'], linewidth=2, color='blue')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Learning Progress: Rewards Over Time', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if len(history['steps']) >= window:
        smoothed_steps = np.convolve(history['steps'], 
                                     np.ones(window)/window, mode='valid')
        plt.plot(smoothed_steps, linewidth=2, color='orange')
    else:
        plt.plot(history['steps'], linewidth=2, color='orange')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Steps to Goal', fontsize=12)
    plt.title('Efficiency: Steps Decreasing Over Time', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*70)
    print("STEP 3: COMPLETE ROUTE COMPARISON - RL vs A*")
    print("="*70)
    print(f"\nComplete route through all {len(goals)} goals:")
    print(f"  Start: {start}")
    for i, goal in enumerate(goals, 1):
        print(f"  Goal {i}: {goal}")
    print("\n- Left: Q-Learning (learned policy)")
    print("- Right: A* (search algorithm)")
    print("\nWatch both algorithms find the complete route in real-time!")
    input("\nPress Enter to start comparison...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.ion()
    
    for idx, (ax, title) in enumerate(zip([ax1, ax2], ['Q-Learning (Your Trained Agent)', 'A* (Classical Search)'])):
        ax.imshow(warehouse.grid, cmap='binary', origin='lower')
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=11)
        ax.set_ylabel('Row', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        ax.plot(start[1], start[0], 'go', markersize=15, 
               markeredgecolor='darkgreen', markeredgewidth=2, label='Start')
        
        goal_colors = ['blue', 'orange', 'purple', 'deeppink', 'gold', 'lime']
        for i, goal in enumerate(goals):
            color_idx = i % len(goal_colors)
            ax.plot(goal[1], goal[0], '*', color=goal_colors[color_idx], 
                   markersize=20, markeredgecolor='black', markeredgewidth=2, 
                   label=f'Goal {i+1}')
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=5, fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.draw()
    plt.pause(1)
    
    rl_total_steps = 0
    rl_total_explored = 0
    rl_total_time = 0
    rl_all_paths = []
    
    astar_total_steps = 0
    astar_total_explored = 0
    astar_total_time = 0
    astar_all_paths = []
    
    route_segments = [(start, goals[0])] + [(goals[i], goals[i+1]) for i in range(len(goals)-1)]
    
    print("\n" + "="*70)
    print("FINDING COMPLETE ROUTE...")
    print("="*70)
    
    print("\n1. Q-Learning Agent finding route through all goals...")
    current_pos = start
    
    for seg_idx, (from_pos, to_pos) in enumerate(route_segments, 1):
        print(f"   Segment {seg_idx}/{len(route_segments)}: {from_pos} → {to_pos}")
        
        rl_explored_scatter = None
        
        def rl_visualize(visited):
            nonlocal rl_explored_scatter
            if rl_explored_scatter:
                try:
                    rl_explored_scatter.remove()
                except:
                    pass
            if visited:
                vx, vy = zip(*visited)
                rl_explored_scatter = ax1.scatter(vy, vx, c='cyan', s=25, alpha=0.6, zorder=2)
                plt.draw()
                plt.pause(0.02)
        
        seg_start = time.time()
        rl_parent, rl_visited, rl_found = agent.find_path(
            warehouse.grid, from_pos, to_pos, visualize_step=rl_visualize, delay=0.02
        )
        seg_time = time.time() - seg_start
        
        if rl_found:
            rl_path = reconstruct_path(rl_parent, from_pos, to_pos)
            rl_all_paths.extend(rl_path if not rl_all_paths else rl_path[1:])
            rl_total_steps += len(rl_path) - 1
            rl_total_explored += len(rl_visited)
            rl_total_time += seg_time
            
            if rl_explored_scatter:
                try:
                    rl_explored_scatter.remove()
                except:
                    pass
            rx, ry = zip(*rl_path)
            ax1.plot(ry, rx, color='blue', linewidth=3, alpha=0.8, zorder=3)
            plt.draw()
            plt.pause(0.3)
        else:
            print(f"   ✗ RL failed on segment {seg_idx}")
            break
    
    rl_complete = (seg_idx == len(route_segments) and rl_found)
    if rl_complete:
        print(f"   ✓ RL complete! Total: {rl_total_steps} steps, {rl_total_explored} nodes explored")
    else:
        print(f"   ⚠ RL incomplete (failed at segment {seg_idx}/{len(route_segments)})")
        print(f"     Partial route: {rl_total_steps} steps, {rl_total_explored} nodes explored")
    
    plt.pause(1)
    
    # A* Algorithm - Find complete route
    print("\n2. A* Algorithm finding route through all goals...")
    current_pos = start
    
    for seg_idx, (from_pos, to_pos) in enumerate(route_segments, 1):
        print(f"   Segment {seg_idx}/{len(route_segments)}: {from_pos} → {to_pos}")
        
        astar_explored_scatter = None
        
        def astar_visualize(visited):
            nonlocal astar_explored_scatter
            if astar_explored_scatter:
                try:
                    astar_explored_scatter.remove()
                except:
                    pass
            if visited:
                vx, vy = zip(*visited)
                astar_explored_scatter = ax2.scatter(vy, vx, c='cyan', s=25, alpha=0.6, zorder=2)
                plt.draw()
                plt.pause(0.02)
        
        seg_start = time.time()
        astar_parent, astar_visited, astar_found = astar(
            warehouse.grid, from_pos, to_pos, manhattan_distance,
            visualize_step=astar_visualize, delay=0.02
        )
        seg_time = time.time() - seg_start
        
        if astar_found:
            astar_path = reconstruct_path(astar_parent, from_pos, to_pos)
            astar_all_paths.extend(astar_path if not astar_all_paths else astar_path[1:])
            astar_total_steps += len(astar_path) - 1
            astar_total_explored += len(astar_visited)
            astar_total_time += seg_time
            
            if astar_explored_scatter:
                try:
                    astar_explored_scatter.remove()
                except:
                    pass
            ax, ay = zip(*astar_path)
            ax2.plot(ay, ax, color='orange', linewidth=3, alpha=0.8, zorder=3)
            plt.draw()
            plt.pause(0.3)
        else:
            print(f"   ✗ A* failed on segment {seg_idx}")
            break
    
    astar_complete = (seg_idx == len(route_segments) and astar_found)
    if astar_complete:
        print(f"   ✓ A* complete! Total: {astar_total_steps} steps, {astar_total_explored} nodes explored")
    else:
        print(f"   ⚠ A* incomplete (failed at segment {seg_idx}/{len(route_segments)})")
        print(f"     Partial route: {astar_total_steps} steps, {astar_total_explored} nodes explored")
    
    plt.ioff()
    plt.show()
    
    print("\n" + "="*70)
    print("COMPLETE ROUTE RESULTS")
    print("="*70)
    
    print(f"\nRoute: {start} → {' → '.join(str(g) for g in goals)}")
    print(f"Total segments: {len(route_segments)}")
    
    if rl_complete and astar_complete:
        print("\n✓ Both algorithms completed the full route!")
    elif rl_complete:
        print("\n✓ RL completed full route | ✗ A* incomplete")
    elif astar_complete:
        print("\n✗ RL incomplete | ✓ A* completed full route")
    else:
        print("\n⚠ Both algorithms had partial failures")
    
    print(f"\nQ-Learning (Your Trained Agent):")
    if rl_complete:
        print(f"  Status: ✓ Complete route")
    else:
        print(f"  Status: ⚠ Partial route (completed {seg_idx-1}/{len(route_segments)} segments)")
    print(f"  Total path length: {rl_total_steps} steps")
    print(f"  Total nodes explored: {rl_total_explored}")
    print(f"  Total execution time: {rl_total_time:.4f}s")
    print(f"  Training time: {train_time:.2f}s")
    
    print(f"\nA* Algorithm:")
    if astar_complete:
        print(f"  Status: ✓ Complete route")
    else:
        print(f"  Status: ⚠ Partial route")
    print(f"  Total path length: {astar_total_steps} steps")
    print(f"  Total nodes explored: {astar_total_explored}")
    print(f"  Total execution time: {astar_total_time:.4f}s")
    
    if rl_complete and astar_complete:
        print(f"\nComparison (Full Route):")
        if rl_total_steps == astar_total_steps:
            print(f"  ✓ Both found same total path length ({rl_total_steps} steps)")
        elif rl_total_steps < astar_total_steps:
            print(f"  ✓ RL found shorter path! ({rl_total_steps} vs {astar_total_steps} steps)")
        else:
            print(f"  ✓ A* found shorter path ({astar_total_steps} vs {rl_total_steps} steps)")
        
        exploration_diff = astar_total_explored - rl_total_explored
        if exploration_diff > 0:
            pct = (exploration_diff / astar_total_explored) * 100
            print(f"  ✓ RL explored {exploration_diff} ({pct:.1f}%) FEWER nodes than A*!")
        else:
            pct = (-exploration_diff / rl_total_explored) * 100
            print(f"  ✓ A* explored {-exploration_diff} ({pct:.1f}%) fewer nodes")
        
        time_diff = astar_total_time - rl_total_time
        if time_diff > 0:
            print(f"  ✓ RL was {time_diff:.4f}s faster")
        else:
            print(f"  ✓ A* was {-time_diff:.4f}s faster")
    elif rl_total_steps > 0 and astar_total_steps > 0:
        print(f"\n⚠ Partial Comparison (Unfair - different route lengths):")
        print(f"  RL: {rl_total_steps} steps (partial)")
        print(f"  A*: {astar_total_steps} steps ({'complete' if astar_complete else 'partial'})")
        if not rl_complete:
            print(f"\n💡 Note: RL failed due to limited training or difficult segments")
            print(f"   Try: More training episodes or different goals")
    
    if rl_complete and astar_complete:
        print("\nGenerating comparison charts...")
        fig_compare, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Chart 1: Path Length Comparison
        ax = axes[0]
        algorithms = ['Q-Learning', 'A*']
        path_lengths = [rl_total_steps, astar_total_steps]
        colors = ['#3498db', '#e67e22']
        bars = ax.bar(algorithms, path_lengths, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Total Steps', fontsize=12, fontweight='bold')
        ax.set_title('Path Length Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(path_lengths) * 1.2)
        for bar, val in zip(bars, path_lengths):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(val)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Chart 2: Nodes Explored Comparison
        ax = axes[1]
        nodes_explored = [rl_total_explored, astar_total_explored]
        bars = ax.bar(algorithms, nodes_explored, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Nodes Explored', fontsize=12, fontweight='bold')
        ax.set_title('Exploration Efficiency', fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(nodes_explored) * 1.2)
        for bar, val in zip(bars, nodes_explored):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(val)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Chart 3: Execution Time Comparison
        ax = axes[2]
        exec_times = [rl_total_time * 1000, astar_total_time * 1000]  # Convert to ms
        bars = ax.bar(algorithms, exec_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
        ax.set_title('Execution Speed', fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(exec_times) * 1.2)
        for bar, val in zip(bars, exec_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}ms',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Complete Route Performance Comparison', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. You selected custom goals by clicking")
    print("  2. RL agent learned to reach ALL goals through training")
    if rl_complete and astar_complete:
        print("  3. Both algorithms completed the full route")
    elif rl_complete:
        print("  3. RL completed full route (A* had issues)")
    elif astar_complete:
        print("  3. A* completed full route (RL needs more training)")
    else:
        print("  3. Both had partial routes (challenging goals!)")
    print("  4. Learned policy executes efficiently with minimal exploration")


if __name__ == "__main__":
    main()
