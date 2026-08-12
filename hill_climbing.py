"""Student file: Random-Restart Hill Climbing.

Design choices (justified in the report):
- Neighborhood: Gaussian step, sigma = sigma_frac * domain_width. Small,
  consistent local moves so the "hill climbing" part stays meaningful.
- Boundary policy: reflect (see boundary.py) -- avoids the boundary-pileup
  problem clip would cause, and correctly handles any number of crossings.
- Restart trigger: no restart on ANY improvement -- only on a MEANINGFUL
  improvement (> epsilon). This avoids "improvement stagnation," where tiny
  genuine improvements inside one basin (common on Rastrigin) reset the
  patience counter forever and the run never restarts.
- Hard local-budget cap: in addition to the no-improvement trigger, no
  single basin may consume more than max_local_budget evaluations, win or
  lose. This is what actually forces diversity on Rastrigin, WITHOUT needing
  a different (larger) sigma there -- sigma stays uniform across benchmarks.
"""
from experiment import run_experiment
from boundary import reflect
from params_store import load_parameters

BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = [0]  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000

# Fallback defaults, used only if best_parameters.json has no entry yet
# for a given benchmark (e.g. before pilot tuning has been run).
DEFAULT_PARAMETERS = {
    "sigma_frac": 0.05,
    "restart_threshold": 200,
    "epsilon": 1e-4,
    "max_local_budget": 2000,
}


def hill_climbing(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    sigma_frac = parameters.get("sigma_frac", DEFAULT_PARAMETERS["sigma_frac"])
    restart_threshold = parameters.get("restart_threshold", DEFAULT_PARAMETERS["restart_threshold"])
    epsilon = parameters.get("epsilon", DEFAULT_PARAMETERS["epsilon"])
    max_local_budget = parameters.get("max_local_budget", DEFAULT_PARAMETERS["max_local_budget"])

    domain_width = upper_bound - lower_bound
    sigma = sigma_frac * domain_width

    while objective.remaining > 0:
        # --- fresh random restart ---
        current = rng.uniform(lower_bound, upper_bound, size=dimension)
        current_value = objective(current)  # counts as this basin's init eval
        evals_since_improvement = 0
        local_evals = 1

        # --- local search within this basin ---
        while (
            evals_since_improvement < restart_threshold
            and local_evals < max_local_budget
            and objective.remaining > 0
        ):
            step = rng.normal(0, sigma, size=dimension)
            neighbor = reflect(current + step, lower_bound, upper_bound)
            neighbor_value = objective(neighbor)
            local_evals += 1

            if neighbor_value < current_value:
                improved_by = current_value - neighbor_value
                current, current_value = neighbor, neighbor_value
                if improved_by > epsilon:
                    evals_since_improvement = 0
                else:
                    evals_since_improvement += 1
            else:
                evals_since_improvement += 1
        # loop exits -> outer while starts a brand-new restart (if budget remains)


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        parameters = load_parameters("hill_climbing", benchmark_name) or DEFAULT_PARAMETERS
        results = run_experiment(
            hill_climbing,
            benchmark_name,
            list(SEEDS),
            parameters,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()