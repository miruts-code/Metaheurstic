"""Automated pilot-seed grid search for simulated_annealing.py."""
import itertools
import numpy as np

from experiment import run_once, PILOT_SEEDS
from simulated_annealing import simulated_annealing
from random_search import random_search
from params_store import save_parameters

SIGMA_FRACS = [0.02, 0.05, 0.1]
ALPHAS = [0.99, 0.995, 0.999]
P0S = [0.5, 0.9]
REHEAT_ENABLEDS = [False]  # fixed off for this pilot pass; revisit for Rastrigin later

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

    grid = itertools.product(SIGMA_FRACS, ALPHAS, P0S, REHEAT_ENABLEDS)
    for sigma_frac, alpha, p0, reheat_enabled in grid:
        params = {
            "sigma_frac": sigma_frac,
            "alpha": alpha,
            "p0": p0,
            "reheat_enabled": reheat_enabled,
        }
        finals = [
            run_once(simulated_annealing, benchmark_name, seed, params, max_evaluations=20_000).best_value
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
        print(f"\n=== Tuning simulated_annealing on {benchmark_name} ===")
        baseline = baseline_median(benchmark_name)
        print(f"  random-search baseline median: {baseline:.4g}")
        best_params = tune_for_benchmark(benchmark_name, baseline)
        print(f"  WINNER: {best_params}")
        save_parameters("simulated_annealing", benchmark_name, best_params)


if __name__ == "__main__":
    main()