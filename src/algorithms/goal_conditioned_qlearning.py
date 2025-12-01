"""
Goal-Conditioned Q-Learning Agent

This agent learns a UNIVERSAL policy that works for ANY goal by including
the goal coordinates in the state representation.

State = (position_x, position_y, goal_x, goal_y)

This allows the agent to learn different optimal actions for the same position
depending on where it's trying to go.
"""

import numpy as np
import time
from collections import defaultdict
import pickle


class GoalConditionedQLearningAgent:
    """
    Goal-conditioned Q-Learning agent that learns policies for multiple goals.
    
    Key difference from standard Q-Learning:
    - State includes BOTH current position AND goal position
    - Can learn to reach any goal with a single Q-table
    - No catastrophic forgetting across different goals
    """
    
    def __init__(self, grid_size, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Goal-Conditioned Q-Learning agent.
        
        Args:
            grid_size: Size of the grid
            learning_rate (alpha): How much to update Q-values (0-1)
            discount_factor (gamma): Future reward importance (0-1)
            epsilon: Initial exploration rate (1.0 = random, 0.0 = greedy)
            epsilon_decay: Rate at which exploration decreases
            epsilon_min: Minimum exploration rate
        """
        self.grid_size = grid_size
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: maps (state, action) -> expected reward
        # state = (pos_x, pos_y, goal_x, goal_y)
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 actions
        
        # Action mapping
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        # Training statistics
        self.training_rewards = []
        self.training_steps = []
        
    def get_state(self, position, goal):
        """
        Convert position and goal to state representation.
        
        KEY DIFFERENCE: State includes BOTH position and goal!
        """
        return (position[0], position[1], goal[0], goal[1])
    
    def choose_action(self, state, grid, training=True):
        """
        Choose action using epsilon-greedy policy.
        
        During training: Explore with probability epsilon
        During testing: Always choose best action
        """
        # Extract current position from state
        pos_x, pos_y = state[0], state[1]
        
        if training and np.random.random() < self.epsilon:
            # Explore: random valid action
            valid_actions = []
            for action_idx in range(4):
                dx, dy = self.actions[action_idx]
                nx, ny = pos_x + dx, pos_y + dy
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                    grid[nx, ny] == 0):
                    valid_actions.append(action_idx)
            
            if valid_actions:
                return np.random.choice(valid_actions)
            return np.random.randint(4)
        else:
            # Exploit: choose best action
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning update rule:
        Q(s,a) = Q(s,a) + α * [reward + γ * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def train_episode(self, grid, start, goal, max_steps=500):
        """
        Train agent for one episode.
        
        Returns:
            total_reward: Total reward earned this episode
            steps: Number of steps taken
            success: Whether goal was reached
        """
        position = start
        state = self.get_state(position, goal)  # State includes goal!
        total_reward = 0
        steps = 0
        visited = set()
        
        for step in range(max_steps):
            visited.add(position)
            steps += 1
            
            # Choose action
            action = self.choose_action(state, grid, training=True)
            
            # Take action
            dx, dy = self.actions[action]
            next_pos = (position[0] + dx, position[1] + dy)
            
            # Calculate reward
            reward = self._calculate_reward(next_pos, goal, grid, visited)
            
            # Update Q-value
            next_state = self.get_state(next_pos, goal) if self._is_valid_position(next_pos, grid) else state
            self.update_q_value(state, action, reward, next_state)
            
            # Move to next state
            if self._is_valid_position(next_pos, grid):
                position = next_pos
                state = next_state
                total_reward += reward
                
                # Check if goal reached
                if position == goal:
                    return total_reward, steps, True
            else:
                # Hit obstacle or boundary
                total_reward += reward
        
        return total_reward, steps, False
    
    def _calculate_reward(self, position, goal, grid, visited):
        """
        Reward function:
        +100: Reach goal
        -1: Each step (encourages shorter paths)
        -50: Hit obstacle
        -10: Revisit same cell (discourage loops)
        """
        # Check boundary
        if not (0 <= position[0] < self.grid_size and 0 <= position[1] < self.grid_size):
            return -50
        
        # Check obstacle
        if grid[position[0], position[1]] == 1:
            return -50
        
        # Check goal
        if position == goal:
            return 100
        
        # Check if revisiting
        if position in visited:
            return -10
        
        # Standard step cost
        return -1
    
    def _is_valid_position(self, position, grid):
        """Check if position is valid (in bounds and not obstacle)"""
        return (0 <= position[0] < self.grid_size and 
                0 <= position[1] < self.grid_size and 
                grid[position[0], position[1]] == 0)
    
    def train_multi_goal(self, grid, start, goals, episodes_per_goal=200, verbose=True):
        """
        Train agent on multiple goals simultaneously.
        
        This is the KEY ADVANTAGE of goal-conditioned RL:
        - Train on all goals at once
        - No forgetting
        - Single unified Q-table
        
        Args:
            grid: Environment grid
            start: Start position
            goals: List of goal positions
            episodes_per_goal: Episodes per goal
            verbose: Print progress
        
        Returns:
            training_history: Dict with stats
        """
        print(f"\n{'='*70}")
        print(f"TRAINING GOAL-CONDITIONED Q-LEARNING AGENT")
        print(f"{'='*70}")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Number of goals: {len(goals)}")
        print(f"Episodes per goal: {episodes_per_goal}")
        print(f"Total episodes: {len(goals) * episodes_per_goal}")
        print(f"Learning rate (α): {self.alpha}")
        print(f"Discount factor (γ): {self.gamma}")
        print(f"{'='*70}\n")
        
        self.training_rewards = []
        self.training_steps = []
        goal_success_rates = {goal: [] for goal in goals}
        
        total_episodes = len(goals) * episodes_per_goal
        episode_count = 0
        
        # Train by cycling through goals
        for episode in range(episodes_per_goal):
            for goal in goals:
                episode_count += 1
                
                total_reward, steps, success = self.train_episode(grid, start, goal)
                
                self.training_rewards.append(total_reward)
                self.training_steps.append(steps)
                goal_success_rates[goal].append(1 if success else 0)
                
                # Decay exploration rate
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                # Print progress
                if verbose and episode_count % 100 == 0:
                    recent_success = sum(1 for r in self.training_rewards[-100:] if r > 0) / min(100, len(self.training_rewards))
                    avg_reward = np.mean(self.training_rewards[-100:])
                    print(f"Episode {episode_count}/{total_episodes} | "
                          f"Success: {recent_success:.1%} | "
                          f"Avg Reward: {avg_reward:.1f} | "
                          f"ε: {self.epsilon:.3f}")
        
        # Calculate per-goal success rates
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*70}")
        for goal in goals:
            success_rate = np.mean(goal_success_rates[goal][-50:])  # Last 50 episodes
            print(f"Goal {goal}: Success rate = {success_rate:.1%}")
        
        overall_success = sum(1 for r in self.training_rewards if r > 0) / len(self.training_rewards)
        print(f"\nOverall Success Rate: {overall_success:.1%}")
        print(f"Q-table size: {len(self.q_table)} states")
        print(f"{'='*70}\n")
        
        return {
            'rewards': self.training_rewards,
            'steps': self.training_steps,
            'goal_success_rates': goal_success_rates,
            'final_epsilon': self.epsilon,
            'overall_success_rate': overall_success
        }
    
    def find_path(self, grid, start, goal, max_steps=200, visualize_step=None, delay=0.02):
        """
        Use learned policy to find path (inference mode).
        
        Returns:
            parent: Dict for path reconstruction
            visited: Set of visited cells
            found: Whether goal was reached
        """
        position = start
        state = self.get_state(position, goal)  # State includes goal!
        visited = set()
        parent = {start: None}
        path = [start]
        
        for step in range(max_steps):
            visited.add(position)
            
            if visualize_step:
                visualize_step(visited)
            
            if position == goal:
                # Convert path to parent dict
                for i in range(len(path) - 1):
                    parent[path[i + 1]] = path[i]
                return parent, visited, True
            
            # Choose best action (greedy)
            action = self.choose_action(state, grid, training=False)
            
            # Take action
            dx, dy = self.actions[action]
            next_pos = (position[0] + dx, position[1] + dy)
            
            # Check validity
            if not self._is_valid_position(next_pos, grid):
                break
            
            # Check for loops
            if next_pos in visited:
                break
            
            # Move to next state
            position = next_pos
            state = self.get_state(position, goal)  # Update state with goal
            path.append(position)
            parent[position] = path[-2]
            
            if delay > 0:
                time.sleep(delay)
        
        return parent, visited, False
    
    def save(self, filepath):
        """Save Q-table and hyperparameters"""
        data = {
            'q_table': dict(self.q_table),
            'grid_size': self.grid_size,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'training_rewards': self.training_rewards,
            'training_steps': self.training_steps
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load Q-table and hyperparameters"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(4), data['q_table'])
        self.grid_size = data['grid_size']
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.training_rewards = data.get('training_rewards', [])
        self.training_steps = data.get('training_steps', [])
        
        print(f"Agent loaded from {filepath}")
