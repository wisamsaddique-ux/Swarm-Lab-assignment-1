"""
Particle Swarm Optimisation for path planning.

A particle = N_WAYPOINTS intermediate waypoints (x, y) in continuous space.
Full path = start -> w1 -> w2 -> ... -> wK -> goal
Cell (r, c) occupies the square [c, c+1) x [r, r+1); x = column, y = row.
"""
import numpy as np

from grid import bfs_path
from config import (N_WAYPOINTS, COLLISION_PENALTY, SAMPLE_STEP, SAFETY,
                    N_PARTICLES, MAX_ITERS, W_MAX, W_MIN, C1, C2,
                    V_MAX_RATIO, PATIENCE, N_RESTARTS, MIN_RUNS)


OFFSETS = ((-SAFETY, -SAFETY), (-SAFETY, SAFETY), (SAFETY, -SAFETY), (SAFETY, SAFETY))


# ---------------- Path helpers and cost ----------------
def cell_center(cell):
    r, c = cell
    return np.array([c + 0.5, r + 0.5])


def build_path(particle, start, goal):
    pts = particle.reshape(-1, 2)
    return np.vstack([cell_center(start), pts, cell_center(goal)])


def path_length(path):
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def collision_count(path, grid):
    """Sample points along every segment and count samples inside obstacles."""
    n = grid.shape[0]
    # all sample points on all segments, shape (m, 2)
    samples = []
    for p, q in zip(path[:-1], path[1:]):
        steps = max(2, int(np.linalg.norm(q - p) / SAMPLE_STEP))
        t = np.linspace(0, 1, steps)[:, None]
        samples.append(p + t * (q - p))
    pts = np.vstack(samples)

    # check a small square (4 corners) around each point so the path keeps a
    # safety margin and cannot squeeze between diagonal obstacles
    hit = np.zeros(len(pts), dtype=bool)
    for dx, dy in OFFSETS:
        c = np.floor(pts[:, 0] + dx).astype(int)
        r = np.floor(pts[:, 1] + dy).astype(int)
        inside = (r >= 0) & (r < n) & (c >= 0) & (c < n)
        blocked = ~inside
        blocked[inside] = grid[r[inside], c[inside]] == 1
        hit |= blocked
    return int(hit.sum())


def cost(particle, grid, start, goal):
    """Fitness = path length + big penalty for every collision sample."""
    path = build_path(particle, start, goal)
    return path_length(path) + COLLISION_PENALTY * collision_count(path, grid)


# ---------------- PSO main loop ----------------
def pso(grid, start, goal, seed):
    """Returns (best_particle, best_cost, history_of_gbest_cost)."""
    rng = np.random.default_rng(seed)
    n = grid.shape[0]
    dim = 2 * N_WAYPOINTS
    lo, hi = SAFETY, n - SAFETY
    v_max = V_MAX_RATIO * n

    # Initialise the swarm:
    #   1/4 near the BFS route (informed guess, small noise)
    #   1/4 near the straight line start->goal (larger noise)
    #   1/2 uniformly random over the grid (diversity / exploration)
    s, g = cell_center(start), cell_center(goal)
    X = rng.uniform(lo, hi, (N_PARTICLES, dim))
    t = np.linspace(0, 1, N_WAYPOINTS + 2)[1:-1]
    line = np.concatenate([s + ti * (g - s) for ti in t])
    route = np.array([cell_center(cell) for cell in bfs_path(grid, start, goal)])
    idx = np.round(t * (len(route) - 1)).astype(int)
    guide = route[idx].ravel()
    q = N_PARTICLES // 4
    for i in range(q):
        X[i] = np.clip(guide + rng.normal(0, 0.3, dim), lo, hi)
    for i in range(q, 2 * q):
        X[i] = np.clip(line + rng.normal(0, 2.0, dim), lo, hi)
    V = rng.uniform(-1, 1, (N_PARTICLES, dim))

    # personal bests and global best
    pbest = X.copy()
    pbest_cost = np.array([cost(x, grid, start, goal) for x in X])
    gi = int(np.argmin(pbest_cost))
    gbest, gbest_cost = pbest[gi].copy(), pbest_cost[gi]

    history = [gbest_cost]
    no_improve = 0
    for it in range(MAX_ITERS):
        w = W_MAX - (W_MAX - W_MIN) * it / MAX_ITERS
        r1 = rng.random((N_PARTICLES, dim))
        r2 = rng.random((N_PARTICLES, dim))

        # velocity & position update
        V = w * V + C1 * r1 * (pbest - X) + C2 * r2 * (gbest - X)
        V = np.clip(V, -v_max, v_max)
        X = np.clip(X + V, lo, hi)

        # evaluate (cost includes collision check) and update bests
        improved = False
        for i in range(N_PARTICLES):
            c = cost(X[i], grid, start, goal)
            if c < pbest_cost[i]:
                pbest[i], pbest_cost[i] = X[i].copy(), c
                if c < gbest_cost - 1e-9:
                    gbest, gbest_cost = X[i].copy(), c
                    improved = True
        history.append(gbest_cost)

        # stopping condition
        no_improve = 0 if improved else no_improve + 1
        if no_improve >= PATIENCE:
            break
    return gbest, gbest_cost, history


def pso_with_restarts(grid, start, goal, seed):
    """
    PSO can get stuck in a local minimum (e.g. a path that still clips an
    obstacle corner). Run several independent swarms (seed, seed+1, ...) and
    keep the best. Stop early once a run finds a collision-free path and all
    runs so far have been done.
    Returns (best_particle, best_cost, history_of_best_run, runs_used).
    """
    best = None
    for k in range(N_RESTARTS):
        result = pso(grid, start, goal, seed + k)
        if best is None or result[1] < best[1]:
            best = result
        path = build_path(best[0], start, goal)
        if k + 1 >= MIN_RUNS and collision_count(path, grid) == 0:
            return (*best, k + 1)
    return (*best, N_RESTARTS)
