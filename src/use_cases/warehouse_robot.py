"""
Use Case: Warehouse Robot Navigation
Automated warehouse robot navigating through shelves to pick items.
"""

import numpy as np
import matplotlib.pyplot as plt


class WarehouseRobot:
    """
    Warehouse robot navigating through aisles and shelves.
    Application: Fulfillment centers, automated warehouses.
    """
    
    def __init__(self, grid_size=30):
        self.grid_size = grid_size
        self.grid = self._create_warehouse_layout()
        
    def _create_warehouse_layout(self):
        """Create a warehouse-like grid with aisles and shelves"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        for col in range(4, self.grid_size - 4, 5):
            for row in range(3, self.grid_size - 3):
                if row % 7 != 0 and row % 7 != 6:
                    grid[row, col] = 1
                    if col + 1 < self.grid_size - 4:
                        grid[row, col + 1] = 1
        
        grid[0:2, :] = 0
        grid[-3:, :] = 0
        grid[:, 0:2] = 0
        grid[:, -3:] = 0
        
        return grid
    
    def select_goals_interactive(self, start, num_goals=4):
        """
        Interactive goal selection by clicking on the grid.
        
        Args:
            start: Starting position (row, col)
            num_goals: Number of goals to select (default: 4)
        
        Returns:
            List of selected goal positions
        """
        import matplotlib.pyplot as plt
        
        selected_goals = []
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(self.grid, cmap='binary', origin='lower')
        ax.set_aspect('equal')
        ax.set_title(f'Click to Select {num_goals} Pickup Locations\n({len(selected_goals)}/{num_goals} selected)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Column (Aisle Direction)', fontsize=12)
        ax.set_ylabel('Row (Shelf Direction)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        ax.plot(start[1], start[0], 'go', markersize=15, 
               markeredgecolor='darkgreen', markeredgewidth=2, label='Start')
        ax.legend(loc='upper right', fontsize=10)
        
        goal_markers = []
        
        def onclick(event):
            nonlocal selected_goals, goal_markers
            
            if event.inaxes != ax or len(selected_goals) >= num_goals:
                return
            
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            
            if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
                print(f"Click outside grid bounds: ({row}, {col})")
                return
            
            if self.grid[row, col] == 1:
                print(f"Cannot select obstacle at ({row}, {col})")
                ax.text(col, row, 'X', color='red', fontsize=20, 
                       ha='center', va='center', weight='bold')
                plt.draw()
                return
            
            if (row, col) in selected_goals:
                print(f"Position ({row}, {col}) already selected")
                return
            
            goal_colors = ['blue', 'orange', 'purple', 'deeppink', 'gold', 'lime']
            color_idx = len(selected_goals) % len(goal_colors)
            goal_color = goal_colors[color_idx]
            
            selected_goals.append((row, col))
            marker = ax.plot(col, row, '*', color=goal_color, markersize=20,
                           markeredgecolor='black', markeredgewidth=2)[0]
            goal_markers.append(marker)
            
            ax.text(col, row + 1, str(len(selected_goals)), 
                   color=goal_color, fontsize=12, ha='center', 
                   weight='bold', bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
            
            print(f"Goal {len(selected_goals)}: ({row}, {col})")
            
            if len(selected_goals) < num_goals:
                ax.set_title(f'Click to Select {num_goals} Pickup Locations\n({len(selected_goals)}/{num_goals} selected)', 
                            fontsize=14, fontweight='bold')
            else:
                ax.set_title(f'All {num_goals} Goals Selected! Close window to continue...', 
                            fontsize=14, fontweight='bold', color='green')
            
            plt.draw()
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        
        print(f"\n{'=' * 60}")
        print("INTERACTIVE GOAL SELECTION")
        print('=' * 60)
        print(f"Starting position: {start}")
        print(f"Click on {num_goals} grid cells to select pickup locations")
        print("- Click on white/gray cells (avoid black obstacles)")
        print("- Selected goals will be marked with stars")
        print("- Close the window when done")
        print('=' * 60 + "\n")
        
        plt.show(block=True)
        
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)
        
        print(f"\n✓ Selected {len(selected_goals)} goals: {selected_goals}\n")
        
        return selected_goals
