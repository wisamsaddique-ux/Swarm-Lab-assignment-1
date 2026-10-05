"""
Swarm Intelligence Lab - Assignment 1
PSO-based path planning on a 2D grid with obstacles.

Usage:
    python main.py                    # uses ROLL_NUMBER from config.py
    python main.py 01-136232-096      # or pass a roll number
"""
import os
import sys

from config import ROLL_NUMBER, STUDENT_NAME, GRID_SIZE, roll_to_seed
from grid import generate_problem
from pso import pso_with_restarts, build_path, path_length, collision_count
from visualize import plot_result, plot_convergence


def main():
    roll = sys.argv[1] if len(sys.argv) > 1 else ROLL_NUMBER
    seed = roll_to_seed(roll)
    print(f"Student     : {STUDENT_NAME}")
    print(f"Roll number : {roll}")
    print(f"Seed        : {seed}")

    grid, start, goal = generate_problem(seed)
    print(f"Grid        : {GRID_SIZE}x{GRID_SIZE}, obstacles = {int(grid.sum())}")
    print(f"Start (r,c) : {start}")
    print(f"Goal  (r,c) : {goal}")

    best, best_cost, history, runs = pso_with_restarts(grid, start, goal, seed)
    path = build_path(best, start, goal)
    hits = collision_count(path, grid)

    print(f"PSO runs    : {runs}  (best run kept)")
    print(f"Iterations  : {len(history) - 1}  (in best run)")
    print(f"Path length : {path_length(path):.3f}")
    print(f"Collisions  : {hits}  ({'obstacle-free' if hits == 0 else 'NOT free'})")
    print(f"Final cost  : {best_cost:.3f}")
    print("Waypoints (x, y):")
    for p in path:
        print(f"   ({p[0]:6.2f}, {p[1]:6.2f})")

    os.makedirs("images", exist_ok=True)
    plot_result(grid, start, goal, path, seed)
    plot_convergence(history)
    print("Saved: images/result_path.png, images/convergence.png")


if __name__ == "__main__":
    main()
