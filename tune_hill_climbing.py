"""Automated pilot-seed grid search for hill_climbing.py."""
import itertools
import numpy as np

from experiment import run_once, PILOT_SEEDS
from hill_climbing import hill_climbing
from random_search import random_search
from params_store import save_parameters

SIGMA_FRACS = [0.02, 0.05, 0.1]
RESTART_THRESHOLDS = [100, 500, 1000]
EPSILONS = [1e-4]              # kept fixed for now; widen later if needed
MAX_LOCAL_BUDGETS = [2000]     # kept fixed for now; widen later if needed

BENCHMARKS = ["sphere", "rastrigin", "rosenbrock"]


def baseline_median(benchmark_name: str) -> float:
    finals = [
        run_once(random_search, benchmark_name, seed, {}, max_evaluations=20_000).best_value
        for seed in PILOT_SEEDS
    ]
    return float(np.median(finals))


def tune_for_benchmark(benchmark_name: str, baseline: float) -> dict:
    best_score = float("inf")
    best_params = None

    grid = itertools.product(SIGMA_FRACS, RESTART_THRESHOLDS, EPSILONS, MAX_LOCAL_BUDGETS)
    for sigma_frac, restart_threshold, epsilon, max_local_budget in grid:
        params = {
            "sigma_frac": sigma_frac,
            "restart_threshold": restart_threshold,
            "epsilon": epsilon,
            "max_local_budget": max_local_budget,
        }
        finals = [
            run_once(hill_climbing, benchmark_name, seed, params, max_evaluations=20_000).best_value
            for seed in PILOT_SEEDS
        ]
        median = float(np.median(finals))
        normalized = median / baseline if baseline > 0 else median

        print(f"  {benchmark_name}: {params} -> median={median:.4g}, normalized={normalized:.4g}")

        if normalized < best_score:
            best_score = normalized
            best_params = params

    return best_params


def main():
    for benchmark_name in BENCHMARKS:
        print(f"\n=== Tuning hill_climbing on {benchmark_name} ===")
        baseline = baseline_median(benchmark_name)
        print(f"  random-search baseline median: {baseline:.4g}")
        best_params = tune_for_benchmark(benchmark_name, baseline)
        print(f"  WINNER: {best_params}")
        save_parameters("hill_climbing", benchmark_name, best_params)


if __name__ == "__main__":
    main()