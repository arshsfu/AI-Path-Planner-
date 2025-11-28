# AI Path Planner - NavMind

An intelligent pathfinding visualization system implementing and comparing classical search algorithms on grid-based environments.

## Project Overview

NavMind is a comprehensive pathfinding project that implements multiple search algorithms and provides detailed performance analysis and visualization capabilities.

## Features

### Implemented Algorithms
- **BFS (Breadth-First Search)** - Uninformed search, guarantees shortest path
- **UCS (Uniform Cost Search)** - Cost-based search with priority queue
- **A\* Search** - Informed search with heuristic guidance
  - Manhattan Distance heuristic (optimal for 4-connected grids)
  - Euclidean Distance heuristic (straight-line distance)

### Capabilities
- Real-time algorithm visualization
- Side-by-side algorithm comparison
- Performance metrics tracking
  - Execution time
  - Nodes explored
  - Path length and cost
- Performance comparison graphs
- Heuristic efficiency analysis
- Random grid generation with obstacle control
- Grid save/load functionality
- Comprehensive logging and reporting

## Project Structure

```
AI-Path-Planner/
├── src/
│   ├── algorithms/
│   │   ├── bfs.py          # Breadth-First Search
│   │   ├── ucs.py          # Uniform Cost Search
│   │   └── astar.py        # A* Search
│   ├── core/
│   │   └── grid.py         # Grid data structure (future)
│   ├── utils/
│   │   ├── grid_generator.py   # Grid generation & I/O
│   │   ├── heuristics.py       # Heuristic functions
│   │   ├── path_utils.py       # Path reconstruction
│   │   └── performance.py      # Performance evaluation
│   ├── visualization/
│   │   └── visualize.py        # Visualization tools
│   ├── results/
│   │   ├── figures/            # Generated charts
│   │   ├── logs/               # Performance data
│   │   └── sample_maps/        # Saved grids
│   ├── config.py          # Configuration settings
│   └── main.py            # Main entry point
└── README.md
```

## Getting Started

### Prerequisites
```bash
pip install numpy matplotlib
```

### Running Milestone 2

```bash
cd src
python main.py
```

This will:
1. Generate a 20×20 grid with 25% obstacles
2. Run all 4 algorithms (BFS, UCS, A* Manhattan, A* Euclidean)
3. Display side-by-side path visualizations
4. Generate performance comparison charts
5. Create heuristic efficiency comparison
6. Save all results to `results/` directory

## Milestone 2 Deliverables

### Uniform Cost Search (UCS)
- Priority queue based on path cost (g-value)
- Guarantees optimal path with non-negative edge costs
- Implemented in `algorithms/ucs.py`

### A* Search Algorithm
- Combines path cost (g) and heuristic estimate (h)
- f(n) = g(n) + h(n) optimization
- Support for pluggable heuristic functions
- Implemented in `algorithms/astar.py`

### Heuristic Functions
**Manhattan Distance:**
- `h(n) = |x₁ - x₂| + |y₁ - y₂|`
- Optimal for 4-connected grids (no diagonal movement)
- Most efficient for our grid structure

**Euclidean Distance:**
- `h(n) = √[(x₁ - x₂)² + (y₁ - y₂)²]`
- Straight-line distance
- Admissible but less informed for 4-connected grids

Both implemented in `utils/heuristics.py`

### Performance Evaluation
Metrics tracked for each algorithm:
- **Execution time** (milliseconds)
- **Nodes explored** (efficiency measure)
- **Path length** (solution quality)
- **Path cost** (total movement cost)
- **Success rate** (path found or not)

### Visualization & Analysis
1. **Side-by-side comparison** - Visual comparison of all algorithms
2. **Performance bar charts** - Time, nodes, path length comparison
3. **Heuristic analysis** - Manhattan vs Euclidean efficiency

### Random Map Generation
- Configurable grid size and obstacle density
- Seeded generation for reproducibility
- Save/load functionality for grid reuse
- Ensures start and goal positions are always accessible

## Performance Analysis

### Expected Results (20×20 grid, 25% obstacles)

**Efficiency (Nodes Explored):**
1. A* (Manhattan) - Most efficient (~40-60 nodes)
2. A* (Euclidean) - Efficient (~50-70 nodes)
3. UCS - Moderate (~100-150 nodes)
4. BFS - Least efficient (~150-200 nodes)

**Execution Speed:**
- All algorithms complete in <50ms for 20×20 grids
- A* variants typically fastest due to fewer node explorations

**Optimality:**
- All algorithms guarantee optimal path length
- Path length is identical across all successful runs

### Key Findings

1. **Manhattan heuristic is superior** for 4-connected grids
   - Explores fewer nodes than Euclidean
   - Better informed estimate of actual path cost

2. **A* significantly outperforms uninformed search**
   - 60-70% reduction in nodes explored vs BFS
   - Maintains optimality guarantee

3. **UCS and BFS explore similar node counts**
   - Both uninformed, but UCS uses priority queue
   - BFS may be faster due to simpler queue operations

## Usage Examples

### Basic Algorithm Comparison
```python
from utils.grid_generator import generate_grid
from algorithms.astar import astar
from utils.heuristics import manhattan_distance
from utils.performance import evaluate_algorithm

grid = generate_grid(20, 0.25, seed=42)
start, goal = (0, 0), (19, 19)

metrics, parent, visited, found = evaluate_algorithm(
    astar, grid, start, goal, "A*", 
    manhattan_distance, "Manhattan"
)
print(metrics)
```

### Custom Grid Generation
```python
from utils.grid_generator import generate_grid, save_grid, load_grid

# Generate and save
grid = generate_grid(size=30, obstacle_prob=0.3, seed=123)
save_grid(grid, "my_map.json", metadata={'description': 'Dense obstacles'})

# Load later
grid, metadata = load_grid("my_map.json")
```

### Visualize Single Algorithm
```python
from visualization.visualize import compare_algorithms_visual

results_data = [
    ("A* Manhattan", parent_m, visited_m, True),
    ("A* Euclidean", parent_e, visited_e, True)
]

compare_algorithms_visual(grid, start, goal, results_data)
```

## Academic Context

This project demonstrates fundamental AI concepts:
- **Search Algorithms**: BFS, UCS, A*
- **Heuristic Design**: Admissibility and consistency
- **Performance Analysis**: Time/space complexity comparison
- **Algorithm Optimization**: Informed vs uninformed search

## Future Enhancements

- Diagonal movement support (8-connected grid)
- Weighted grids (terrain costs)
- Dynamic obstacles (moving barriers)
- Bidirectional search variants
- Jump Point Search optimization
- 3D pathfinding extension

## Contributors

NavMind AI Path Planner - Milestone 2 Implementation

---

**Last Updated**: November 2025
**Status**: Milestone 2 Complete 
