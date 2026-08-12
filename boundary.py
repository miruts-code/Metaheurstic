"""Shared boundary-handling helper used by all four student algorithms."""
import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


def reflect(x: Vector, lower_bound: float, upper_bound: float) -> Vector:
    """Reflect out-of-bounds coordinates back into [lower_bound, upper_bound].

    Uses a closed-form triangle-wave formula so that any number of
    boundary crossings (not just a single overshoot) is handled correctly
    in one pass, with no explicit loop needed.
    """
    x = np.asarray(x, dtype=float)
    width = upper_bound - lower_bound
    period = 2 * width
    x_shifted = (x - lower_bound) % period
    x_reflected = np.where(x_shifted > width, period - x_shifted, x_shifted)
    return x_reflected + lower_bound