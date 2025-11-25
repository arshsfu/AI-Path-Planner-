def manhattan_distance(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def euclidean_distance(node, goal):
    return ((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2) ** 0.5


def diagonal_distance(node, goal):
    dx = abs(node[0] - goal[0])
    dy = abs(node[1] - goal[1])
    return max(dx, dy)


def zero_heuristic(node, goal):
    return 0
