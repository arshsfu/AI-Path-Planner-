"""
Q-Learning Reinforcement Learning Agent for Pathfinding

This agent LEARNS optimal paths through trial and error, unlike A*/BFS which use search.
After training, it can find paths as efficiently as A* but through learned experience.
"""

import numpy as np
import time
from collections import defaultdict
import pickle


class QLearningAgent:
    """
    Q-Learning agent that learns optimal pathfinding policies.
    
    The agent learns a Q-table: Q(state, action) -> expected reward
    After training, it can navigate optimally without search algorithms.
    """
    
    def __init__(self, grid_size, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Q-Learning agent.
        
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
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 actions: up, down, left, right
        
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
        
    def get_state(self, position):
        """Convert position to state representation"""
        return position
    
    def choose_action(self, state, grid, training=True):
        """
        Choose action using epsilon-greedy policy.
        
        During training: Explore with probability epsilon
        During testing: Always choose best action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random valid action
            valid_actions = []
            for action_idx in range(4):
                dx, dy = self.actions[action_idx]
                nx, ny = state[0] + dx, state[1] + dy
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                    grid[nx, ny] == 0):
                    valid_actions.append(action_idx)
            
            if valid_actions:
                return np.random.choice(valid_actions)
            return np.random.randint(4)  # If stuck, random
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
        state = self.get_state(start)
        total_reward = 0
        steps = 0
        visited = set()
        
        for step in range(max_steps):
            visited.add(state)
            steps += 1
            
            # Choose action
            action = self.choose_action(state, grid, training=True)
            
            # Take action
            dx, dy = self.actions[action]
            next_pos = (state[0] + dx, state[1] + dy)
            
            # Calculate reward
            reward = self._calculate_reward(next_pos, goal, grid, visited)
            
            # Update Q-value
            next_state = self.get_state(next_pos) if self._is_valid_position(next_pos, grid) else state
            self.update_q_value(state, action, reward, next_state)
            
            # Move to next state
            if self._is_valid_position(next_pos, grid):
                state = next_state
                total_reward += reward
                
                # Check if goal reached
                if state == goal:
                    return total_reward, steps, True
            else:
                # Hit obstacle or boundary - stay in place but penalize
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
    
    def train(self, grid, start, goal, episodes=1000, verbose=True, print_every=100):
        """
        Train agent over multiple episodes.
        
        Args:
            grid: Environment grid
            start: Start position
            goal: Goal position
            episodes: Number of training episodes
            verbose: Print progress
            print_every: Print frequency
        
        Returns:
            training_history: Dict with rewards and steps per episode
        """
        print(f"\n{'='*70}")
        print(f"TRAINING Q-LEARNING AGENT")
        print(f"{'='*70}")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Episodes: {episodes}")
        print(f"Learning rate (α): {self.alpha}")
        print(f"Discount factor (γ): {self.gamma}")
        print(f"Exploration rate (ε): {self.epsilon} -> {self.epsilon_min}")
        print(f"{'='*70}\n")
        
        self.training_rewards = []
        self.training_steps = []
        success_count = 0
        
        for episode in range(episodes):
            total_reward, steps, success = self.train_episode(grid, start, goal)
            
            self.training_rewards.append(total_reward)
            self.training_steps.append(steps)
            
            if success:
                success_count += 1
            
            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Print progress
            if verbose and (episode + 1) % print_every == 0:
                recent_success_rate = sum(1 for r in self.training_rewards[-100:] if r > 0) / min(100, len(self.training_rewards))
                avg_reward = np.mean(self.training_rewards[-100:])
                avg_steps = np.mean(self.training_steps[-100:])
                print(f"Episode {episode+1}/{episodes} | "
                      f"Success Rate: {recent_success_rate:.2%} | "
                      f"Avg Reward: {avg_reward:.1f} | "
                      f"Avg Steps: {avg_steps:.1f} | "
                      f"ε: {self.epsilon:.3f}")
        
        final_success_rate = success_count / episodes
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"Overall Success Rate: {final_success_rate:.2%}")
        print(f"Q-table size: {len(self.q_table)} states")
        print(f"{'='*70}\n")
        
        return {
            'rewards': self.training_rewards,
            'steps': self.training_steps,
            'final_epsilon': self.epsilon,
            'success_rate': final_success_rate
        }
    
    def find_path(self, grid, start, goal, max_steps=200, visualize_step=None, delay=0.02):
        """
        Use learned policy to find path (inference mode).
        
        Returns:
            parent: Dict for path reconstruction
            visited: Set of visited cells
            found: Whether goal was reached
        """
        state = self.get_state(start)
        visited = set()
        parent = {start: None}
        path = [start]
        
        for step in range(max_steps):
            visited.add(state)
            
            if visualize_step:
                visualize_step(visited)
            
            if state == goal:
                # Convert path to parent dict for compatibility
                for i in range(len(path) - 1):
                    parent[path[i + 1]] = path[i]
                return parent, visited, True
            
            # Choose best action (greedy, no exploration)
            action = self.choose_action(state, grid, training=False)
            
            # Take action
            dx, dy = self.actions[action]
            next_pos = (state[0] + dx, state[1] + dy)
            
            # Check validity
            if not self._is_valid_position(next_pos, grid):
                # Agent stuck - return failure
                break
            
            # Check for loops
            if next_pos in visited:
                # Agent looping - return failure
                break
            
            # Move to next state
            state = next_pos
            path.append(state)
            parent[state] = path[-2]
            
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


def q_learning(grid, start, goal, episodes=1000, visualize_step=None, delay=0.02):
    """
    Wrapper function for Q-Learning that matches the interface of other algorithms.
    
    This function trains a Q-Learning agent and then uses it to find a path.
    
    Args:
        grid: 2D numpy array
        start: Start position tuple
        goal: Goal position tuple
        episodes: Number of training episodes (default: 1000)
        visualize_step: Optional callback for visualization
        delay: Delay for visualization
    
    Returns:
        parent: Dict for path reconstruction
        visited: Set of visited cells during inference
        found: Whether path was found
    """
    agent = QLearningAgent(
        grid_size=grid.shape[0],
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Train agent
    agent.train(grid, start, goal, episodes=episodes, verbose=False)
    
    # Find path using learned policy
    parent, visited, found = agent.find_path(grid, start, goal, 
                                            visualize_step=visualize_step, 
                                            delay=delay)
    
    return parent, visited, found
