"""Student file: Genetic Algorithm -- tournament selection, blend crossover,
Gaussian mutation.

Design choices (justified in the report):
- Selection: tournament (size=3, fixed). Chosen over roulette because
  roulette's selection probability is proportional to RAW fitness value,
  which makes it highly sensitive to a benchmark's numeric scale (e.g. one
  outlier individual on Rosenbrock can dominate >90% of selection
  probability). Tournament only ever compares individuals within a small
  random group, so it stays well-behaved across benchmarks of very
  different raw scale without needing extra fitness-rescaling machinery.
  tournament_size=3 is a moderate, literature-standard choice: meaningful
  pressure toward quality without collapsing population diversity in 1-2
  generations (fixed for the pilot grid; not swept, see tune file).
- Crossover: blend/arithmetic (child = w*parentA + (1-w)*parentB, w drawn
  per-gene from a widened [-alpha, 1+alpha] range -- BLX-alpha). Chosen over
  one-/two-point or uniform crossover because those can only ever
  reassign EXISTING parent coordinate values to a child -- for continuous
  variables, blend crossover can produce genuinely new intermediate (or
  mildly extrapolated) values, a real source of exploration the discrete-
  style operators lack.
- Mutation: Gaussian, sigma = sigma_frac * domain_width (same normalized
  design as hill_climbing.py / simulated_annealing.py / evolution_strategy.py),
  applied independently per-gene with probability mutation_rate=0.1 (fixed;
  approximately 1/dimension, a standard heuristic -- sigma_frac already
  covers the primary mutation-STRENGTH dial, so sweeping mutation_rate too
  would mostly duplicate that effect rather than test something new).
- Replacement: generational -- the whole population is replaced by
  population_size offspring each generation (elitism: the single best
  individual is always carried over unchanged, so the best-ever solution
  can never be lost even though selection/crossover/mutation are lossy).
"""
import numpy as np

from experiment import run_experiment
from boundary import reflect
from params_store import load_parameters

BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = [0]  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
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