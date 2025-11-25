import heapq
import time

def astar(grid, start, goal, heuristic, visualize_step=None, delay=0.02):
    counter = 0
    f_start = heuristic(start, goal)
    pq = [(f_start, counter, start)]
    counter += 1
    
    parent = {start: None}
    g_cost = {start: 0}
    visited = set()
    
    while pq:
        current_f, _, node = heapq.heappop(pq)
        
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
                new_g_cost = g_cost[node] + 1
                
                if neighbor not in visited:
                    if neighbor not in g_cost or new_g_cost < g_cost[neighbor]:
                        g_cost[neighbor] = new_g_cost
                        h_cost = heuristic(neighbor, goal)
                        f_cost = new_g_cost + h_cost
                        parent[neighbor] = node
                        heapq.heappush(pq, (f_cost, counter, neighbor))
                        counter += 1
        
        time.sleep(delay)
    
    return parent, visited, False
