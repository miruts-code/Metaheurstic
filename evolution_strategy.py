"""Student file: Evolution Strategy, truncation selection + Gaussian mutation.
"""
import numpy as np

from experiment import run_experiment
from boundary import reflect
from params_store import load_parameters

BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = range(100, 120)  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000

DEFAULT_PARAMETERS = {
    "mu_over_lambda": 0.3,
    "lambda_": 50,
    "sigma_frac": 0.05,
    "replacement": "comma",  # "comma" = (mu,lambda), "plus" = (mu+lambda)
}


def evolution_strategy(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    mu_over_lambda = parameters.get("mu_over_lambda", DEFAULT_PARAMETERS["mu_over_lambda"])
    lambda_ = parameters.get("lambda_", DEFAULT_PARAMETERS["lambda_"])
    sigma_frac = parameters.get("sigma_frac", DEFAULT_PARAMETERS["sigma_frac"])
    replacement = parameters.get("replacement", DEFAULT_PARAMETERS["replacement"])

    mu = max(1, round(mu_over_lambda * lambda_))
    domain_width = upper_bound - lower_bound
    sigma = sigma_frac * domain_width

    # --- initialize mu parents (uniform inside bounds) ---
    parents = []
    parent_values = []
    for _ in range(mu):
        if objective.remaining <= 0:
            break
        individual = rng.uniform(lower_bound, upper_bound, size=dimension)
        value = objective(individual)
        parents.append(individual)
        parent_values.append(value)

    parents = np.array(parents)
    parent_values = np.array(parent_values)

    # --- main generational loop ---
    while objective.remaining > 0 and len(parents) > 0:
        offspring = []
        offspring_values = []

        for _ in range(lambda_):
            if objective.remaining <= 0:
                break
            parent_index = rng.integers(0, len(parents))
            child = reflect(
                parents[parent_index] + rng.normal(0, sigma, size=dimension),
                lower_bound,
                upper_bound,
            )
            child_value = objective(child)
            offspring.append(child)
            offspring_values.append(child_value)

        if not offspring:
            break  # budget ran out mid-generation, nothing to select from

        offspring = np.array(offspring)
        offspring_values = np.array(offspring_values)

        if replacement == "plus":
            candidates = np.vstack([parents, offspring])
            candidate_values = np.concatenate([parent_values, offspring_values])
        else:  # "comma": next generation comes ONLY from offspring
            candidates = offspring
            candidate_values = offspring_values

        # --- truncation selection: keep the best mu ---
        num_survivors = min(mu, len(candidates))
        survivor_indices = np.argsort(candidate_values)[:num_survivors]
        parents = candidates[survivor_indices]
        parent_values = candidate_values[survivor_indices]


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        parameters = load_parameters("evolution_strategy", benchmark_name) or DEFAULT_PARAMETERS
        results = run_experiment(
            evolution_strategy,
            benchmark_name,
            list(SEEDS),
            parameters,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()