"""Student file: Genetic Algorithm -- tournament selection, blend crossover,
Gaussian mutation."""


import numpy as np

from experiment import run_experiment
from boundary import reflect
from params_store import load_parameters

BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = range(100, 120)  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000

TOURNAMENT_SIZE = 3         # fixed (see design rationale above)
MUTATION_RATE = 0.1         # fixed (~1/dimension)
BLEND_ALPHA = 0.5           # BLX-alpha extrapolation factor

DEFAULT_PARAMETERS = {
    "population_size": 50,
    "crossover_rate": 0.8,
    "sigma_frac": 0.05,
}


def _tournament_select(values, rng, tournament_size):
    competitors = rng.choice(len(values), size=tournament_size, replace=False)
    return competitors[np.argmin(values[competitors])]


def _blend_crossover(parent_a, parent_b, rng, alpha, lower_bound, upper_bound):
    low = np.minimum(parent_a, parent_b) - alpha * np.abs(parent_a - parent_b)
    high = np.maximum(parent_a, parent_b) + alpha * np.abs(parent_a - parent_b)
    child = rng.uniform(low, high)
    return reflect(child, lower_bound, upper_bound)


def genetic_algorithm(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    population_size = parameters.get("population_size", DEFAULT_PARAMETERS["population_size"])
    crossover_rate = parameters.get("crossover_rate", DEFAULT_PARAMETERS["crossover_rate"])
    sigma_frac = parameters.get("sigma_frac", DEFAULT_PARAMETERS["sigma_frac"])

    domain_width = upper_bound - lower_bound
    sigma = sigma_frac * domain_width

    # --- initialize population (uniform inside bounds) ---
    population = []
    values = []
    for _ in range(population_size):
        if objective.remaining <= 0:
            break
        individual = rng.uniform(lower_bound, upper_bound, size=dimension)
        value = objective(individual)
        population.append(individual)
        values.append(value)

    population = np.array(population)
    values = np.array(values)

    # --- main generational loop ---
    while objective.remaining > 0 and len(population) > 0:
        best_index = np.argmin(values)
        elite = population[best_index].copy()
        elite_value = values[best_index]

        offspring = [elite]        # elitism: best individual always survives
        offspring_values = [elite_value]

        while len(offspring) < len(population):
            if objective.remaining <= 0:
                break

            parent_a_index = _tournament_select(values, rng, TOURNAMENT_SIZE)
            parent_b_index = _tournament_select(values, rng, TOURNAMENT_SIZE)

            if rng.random() < crossover_rate:
                child = _blend_crossover(
                    population[parent_a_index], population[parent_b_index],
                    rng, BLEND_ALPHA, lower_bound, upper_bound,
                )
            else:
                child = population[parent_a_index].copy()

            # --- Gaussian mutation, per-gene, applied with probability MUTATION_RATE ---
            mutation_mask = rng.random(dimension) < MUTATION_RATE
            child[mutation_mask] += rng.normal(0, sigma, size=dimension)[mutation_mask]
            child = reflect(child, lower_bound, upper_bound)

            child_value = objective(child)
            offspring.append(child)
            offspring_values.append(child_value)

        if len(offspring) == 0:
            break

        population = np.array(offspring)
        values = np.array(offspring_values)


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        parameters = load_parameters("genetic_algorithm", benchmark_name) or DEFAULT_PARAMETERS
        results = run_experiment(
            genetic_algorithm,
            benchmark_name,
            list(SEEDS),
            parameters,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()