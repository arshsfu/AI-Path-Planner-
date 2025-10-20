from collections import deque
import time

def bfs(grid, start, goal, visualize_step=None, delay=0.02):
    queue = deque([start])
    parent = {start: None}
    visited = set([start])

    while queue:
        node = queue.popleft()

        # optional live visualization
        if visualize_step:
            visualize_step(visited)

        # goal check
        if node == goal:
            return parent, visited, True  # ✅ path found

        # explore 4-connected neighbors
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    queue.append(neighbor)

        time.sleep(delay)  # control animation speed

    # if loop finishes without reaching goal
    return parent, visited, False  # ❌ no path found
