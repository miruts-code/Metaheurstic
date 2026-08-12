"""Automated pilot-seed grid search for evolution_strategy.py.

Grid (per our discussion): mu_over_lambda x lambda_ x sigma_frac swept;
replacement fixed at "comma" for this pilot pass (weakest-justified dial --
revisit with a targeted comma-vs-plus comparison later if time allows).
-> 2 x 3 x 3 = 18 combinations per benchmark, matching SA's grid size.
"""
import itertools
import numpy as np

from experiment import run_once, PILOT_SEEDS
from evolution_strategy import evolution_strategy
from random_search import random_search
from params_store import save_parameters

MU_OVER_LAMBDAS = [0.2, 0.5]
LAMBDAS = [20, 50, 100]
SIGMA_FRACS = [0.02, 0.05, 0.1]
REPLACEMENTS = ["comma"]  # fixed for pilot; widen to ["comma","plus"] later if time allows

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

    grid = itertools.product(MU_OVER_LAMBDAS, LAMBDAS, SIGMA_FRACS, REPLACEMENTS)
    for mu_over_lambda, lambda_, sigma_frac, replacement in grid:
        params = {
            "mu_over_lambda": mu_over_lambda,
            "lambda_": lambda_,
            "sigma_frac": sigma_frac,
            "replacement": replacement,
        }
        finals = [
            run_once(evolution_strategy, benchmark_name, seed, params, max_evaluations=20_000).best_value
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
        print(f"\n=== Tuning evolution_strategy on {benchmark_name} ===")
        baseline = baseline_median(benchmark_name)
        print(f"  random-search baseline median: {baseline:.4g}")
        best_params = tune_for_benchmark(benchmark_name, baseline)
        print(f"  WINNER: {best_params}")
        save_parameters("evolution_strategy", benchmark_name, best_params)


if __name__ == "__main__":
    main()