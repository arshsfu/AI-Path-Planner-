import heapq
import time

def ucs(grid, start, goal, visualize_step=None, delay=0.02):
    counter = 0
    pq = [(0, counter, start)]
    counter += 1
    
    parent = {start: None}
    cost = {start: 0}
    visited = set()
    
    while pq:
        current_cost, _, node = heapq.heappop(pq)
        
        if node in visited:
            continue
            
        visited.add(node)
        
        if visualize_step:
            visualize_step(visited)
        
        if node == goal:
            return parent, visited, True
        
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = node[0] + dx, node[1] + dy
            
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                neighbor = (nx, ny)
                new_cost = current_cost + 1
                
                if neighbor not in visited:
                    if neighbor not in cost or new_cost < cost[neighbor]:
                        cost[neighbor] = new_cost
                        parent[neighbor] = node
                        heapq.heappush(pq, (new_cost, counter, neighbor))
                        counter += 1
        
        time.sleep(delay)
    
    return parent, visited, False
