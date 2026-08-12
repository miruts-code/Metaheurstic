"""Student file: Simulated Annealing."""
import numpy as np

from experiment import run_experiment
from boundary import reflect
from params_store import load_parameters

BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = range(100, 120)  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000

DEFAULT_PARAMETERS = {
    "sigma_frac": 0.05,
    "alpha": 0.995,
    "p0": 0.8,
    "reheat_enabled": False,
}

N_CALIBRATION_PROBES = 30
REHEAT_T_MIN_FRAC = 1e-3
REHEAT_TO_FRAC = 0.5


def _calibrate_T0(objective, rng, sigma, dimension, lower_bound, upper_bound, p0):
    """Probe this objective's typical 'bad move' size, then solve for T0
    such that P(accept a typical bad move) ~= p0 at the start of the run."""
    deltas = []
    for _ in range(N_CALIBRATION_PROBES):
        if objective.remaining <= 0:
            break
        point = rng.uniform(lower_bound, upper_bound, size=dimension)
        point_value = objective(point)
        if objective.remaining <= 0:
            break
        neighbor = reflect(point + rng.normal(0, sigma, size=dimension), lower_bound, upper_bound)
        neighbor_value = objective(neighbor)
        delta = neighbor_value - point_value
        if delta > 0:
            deltas.append(delta)

    if not deltas:
        return 1.0

    typical_delta = float(np.mean(deltas))
    p0_safe = min(p0, 0.999999)
    return max(-typical_delta / np.log(p0_safe), 1e-8)


def simulated_annealing(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    sigma_frac = parameters.get("sigma_frac", DEFAULT_PARAMETERS["sigma_frac"])
    alpha = parameters.get("alpha", DEFAULT_PARAMETERS["alpha"])
    p0 = parameters.get("p0", DEFAULT_PARAMETERS["p0"])
    reheat_enabled = parameters.get("reheat_enabled", DEFAULT_PARAMETERS["reheat_enabled"])

    domain_width = upper_bound - lower_bound
    sigma = sigma_frac * domain_width

    # --- calibrate starting temperature to THIS objective's own scale ---
    temperature = _calibrate_T0(objective, rng, sigma, dimension, lower_bound, upper_bound, p0)
    temperature_min = REHEAT_T_MIN_FRAC * temperature
    temperature_reheat_to = REHEAT_TO_FRAC * temperature

    # --- initialize current point (uniform inside bounds) ---
    current = rng.uniform(lower_bound, upper_bound, size=dimension)
    current_value = objective(current)

    # --- main annealing loop ---
    while objective.remaining > 0:
        step = rng.normal(0, sigma, size=dimension)
        neighbor = reflect(current + step, lower_bound, upper_bound)
        neighbor_value = objective(neighbor)

        delta = neighbor_value - current_value
        if delta < 0 or rng.random() < np.exp(-delta / temperature):
            current, current_value = neighbor, neighbor_value

        temperature *= alpha

        if reheat_enabled and temperature < temperature_min and objective.remaining > 0:
            temperature = temperature_reheat_to


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        parameters = load_parameters("simulated_annealing", benchmark_name) or DEFAULT_PARAMETERS
        results = run_experiment(
            simulated_annealing,
            benchmark_name,
            list(SEEDS),
            parameters,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()