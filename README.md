# Swarm-Based Path Planning with Obstacles (PSO)

**Swarm Intelligence Lab, Assignment 1**
Bahria University, H-11 Campus Islamabad, Department of Computer Science

| | |
|---|---|
| **Name** | Wisam Saddique |
| **Roll Number** | 01-136232-096 |
| **Class** | BS (AI) |
| **Seed used** | `1136232096` (digits of the roll number) |
| **Algorithm** | Particle Swarm Optimization (PSO) |

---

## 1. Problem

Find a short, obstacle-free path from a **start** cell to a **goal** cell on a 2D grid using PSO.

The problem instance is generated from my roll number, so it is unique to me. Nothing is hardcoded:

```python
SEED = int("01136232096")   # digits of roll number 01-136232-096 -> 1136232096
random.seed(SEED)           # used for obstacles, start and goal
```

| Parameter | Value |
|---|---|
| Grid size | 20 × 20 |
| Obstacles | 88 cells (22% of the grid, `random.sample`) |
| Start (row, col) | (3, 14) |
| Goal (row, col) | (19, 12) |
| Solvability | BFS checks the goal is reachable; otherwise the grid is regenerated |

## 2. Approach

### Path representation
Each particle is a candidate path made of **8 intermediate waypoints** `(x, y)` in continuous grid
coordinates, so one particle is a vector of 16 numbers:

```
START -> w1 -> w2 -> ... -> w8 -> GOAL     (joined by straight segments)
```

### Cost (fitness) function
```
cost = path_length + 100 × (number of collision samples)
```
- **path_length**: sum of the Euclidean lengths of all segments.
- **collisions**: every segment is sampled every 0.05 units. A sample collides if a small square
  (half-size 0.15) around it touches an obstacle cell or leaves the grid. This keeps a safety margin,
  so the path cannot slip between two diagonal obstacles.

The penalty is large, so any colliding path is worse than every collision-free path. The swarm first
learns to avoid obstacles and then makes the path shorter.

### PSO update
```
v = w·v + c1·r1·(pBest − x) + c2·r2·(gBest − x)
x = x + v          (velocity clamped, position kept inside the grid)
```
The inertia weight `w` decreases linearly from 0.9 to 0.4: the swarm explores at the start and fine-tunes at the end.

### Initialisation
- 1/4 of the particles start near the **BFS route** from start to goal (a rough but feasible guess).
- 1/4 start near the **straight line** start → goal, with noise.
- 1/2 start **randomly** over the grid, for diversity.

### Restarts
PSO can get stuck in a local minimum. The program runs at least 3 independent swarms (seeds `seed`,
`seed+1`, …, up to 10 runs) and keeps the best path, stopping once a collision-free path has been found.

### Parameters
| Parameter | Value |
|---|---|
| Particles | 60 |
| Waypoints per particle | 8 |
| Max iterations | 300 |
| Inertia weight `w` | 0.9 → 0.4 |
| `c1`, `c2` | 1.5, 1.5 |
| Max velocity | 20% of grid size |
| Collision penalty | 100 |
| Early stop | no gBest improvement for 60 iterations |
| Restarts | min 3, max 10 |

### Stopping condition
A run stops after 300 iterations, or earlier if gBest has not improved for 60 iterations.
The final answer is the best gBest across all runs.

## 3. Project Structure

```
swarm-pathplanning-01-136232-096/
├── main.py            # entry point: generate problem, run PSO, print + plot results
├── config.py          # roll number, seed and all parameters
├── grid.py            # grid / obstacle / start / goal generation + BFS
├── pso.py             # cost function, collision check, PSO, restarts
├── visualize.py       # plots (grid + path, convergence)
├── requirements.txt
└── images/
    ├── result_path.png      # final path
    ├── convergence.png      # gBest cost per iteration
    ├── sample_output.txt    # console output
    └── flow_diagram.jpg     # photo of the hand-drawn flow diagram
```

## 4. How to Run

```bash
git clone https://github.com/<your-username>/swarm-pathplanning-01-136232-096.git
cd swarm-pathplanning-01-136232-096
pip install -r requirements.txt
python main.py
```

To run with a different roll number (a different problem), either edit `ROLL_NUMBER` in `config.py` or pass it on the command line:

```bash
python main.py 01-136232-096
```

The program prints the seed, start, goal, best path length and waypoints. It saves the plots in `images/`.

## 5. Results

```
Seed        : 1136232096
Grid        : 20x20, obstacles = 88
Start (r,c) : (3, 14)
Goal  (r,c) : (19, 12)
PSO runs    : 3  (best run kept)
Path length : 19.099
Collisions  : 0  (obstacle-free)
```

| Metric | Value |
|---|---|
| Best path length | **19.099** grid units |
| Collisions | **0** (obstacle-free) |
| Waypoints | 8 (+ start and goal) |

### Best path
Black = obstacles, green square = start, red star = goal, blue = PSO best path.

![PSO best path](images/result_path.png)

### Convergence
The gBest cost decreases as the swarm shortens the path.

![Convergence](images/convergence.png)

## 6. Hand-Drawn Flow Diagram

![Hand-drawn flow diagram](images/flow_diagram.jpg)

**Flow:** Start → read roll number / set seed → generate grid, obstacles, start and goal → BFS check
(regenerate if unreachable) → initialise swarm → evaluate cost → set pBest / gBest → update velocity →
update position → check collision with obstacles and update the cost → stopping condition? (no → loop back to
the velocity update) → restart if needed and keep the best → output the best path and plot → End.

## 7. Conclusion

PSO found an obstacle-free path of length **19.099** from (3, 14) to (19, 12) on my seeded 20×20 grid.
The path is encoded as waypoints and a heavy collision penalty is added to its length, so standard PSO
can handle obstacles. BFS-guided initialisation and a few restarts make the result reliable. PSO is
stochastic, so the path is short but not guaranteed to be the shortest possible.
