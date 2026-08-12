"""Load/save frozen hyperparameters chosen during pilot tuning.

Stores one JSON file at the repo root, structured as:
{
  "hill_climbing": {
    "sphere":     {"sigma_frac": 0.05, "restart_threshold": 200, ...},
    "rastrigin":  {...},
    "rosenbrock": {...}
  },
  "simulated_annealing": {...},
  ...
}

This file is committed to the repo (NOT in .gitignore) since it records a
real design decision your mentor/report should be able to see and reproduce.
"""
import json
from pathlib import Path
from typing import Any

STORE_PATH = Path(__file__).parent / "best_parameters.json"


def load_all() -> dict[str, dict[str, dict[str, Any]]]:
    if not STORE_PATH.exists():
        return {}
    with STORE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_parameters(algorithm_name: str, benchmark_name: str) -> dict[str, Any]:
    """Return the frozen PARAMETERS for one (algorithm, benchmark) pair.

    Returns an empty dict if nothing has been tuned/saved yet for that pair
    (useful for quick manual testing before pilot tuning is done).
    """
    all_params = load_all()
    return all_params.get(algorithm_name, {}).get(benchmark_name, {})


def save_parameters(algorithm_name: str, benchmark_name: str, parameters: dict[str, Any]) -> None:
    all_params = load_all()
    all_params.setdefault(algorithm_name, {})[benchmark_name] = parameters
    with STORE_PATH.open("w", encoding="utf-8") as file:
        json.dump(all_params, file, indent=2)
    print(f"Saved {algorithm_name}/{benchmark_name} -> {parameters}")