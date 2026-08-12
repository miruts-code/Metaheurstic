"""Automated pilot-seed grid search for genetic_algorithm.py."""
import itertools
import numpy as np

from experiment import run_once, PILOT_SEEDS
from genetic_algorithm import genetic_algorithm
from random_search import random_search
from params_store import save_parameters

POPULATION_SIZES = [30, 50, 100]
CROSSOVER_RATES = [0.7, 0.9]
SIGMA_FRACS = [0.02, 0.05, 0.1]

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

    grid = itertools.product(POPULATION_SIZES, CROSSOVER_RATES, SIGMA_FRACS)
    for population_size, crossover_rate, sigma_frac in grid:
        params = {
            "population_size": population_size,
            "crossover_rate": crossover_rate,
            "sigma_frac": sigma_frac,
        }
        finals = [
            run_once(genetic_algorithm, benchmark_name, seed, params, max_evaluations=20_000).best_value
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
        print(f"\n=== Tuning genetic_algorithm on {benchmark_name} ===")
        baseline = baseline_median(benchmark_name)
        print(f"  random-search baseline median: {baseline:.4g}")
        best_params = tune_for_benchmark(benchmark_name, baseline)
        print(f"  WINNER: {best_params}")
        save_parameters("genetic_algorithm", benchmark_name, best_params)


if __name__ == "__main__":
    main()