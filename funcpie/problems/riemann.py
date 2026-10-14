"""Riemann Hypothesis Critical Line and Non-Trivial Zero Approximation."""

from __future__ import annotations

from typing import List, Tuple

from funcpie.backend import get_backend


def riemann_zeta_critical_line(
    t_samples: List[float],
    terms: int = 1000,
) -> List[Tuple[float, float, float, float]]:
    """Evaluate the Riemann Zeta function along the critical line s = 1/2 + i*t.

    Args:
        t_samples: List of real heights t along the critical line.
        terms: Number of expansion terms for the accelerated Dirichlet Eta series.

    Returns:
        List of tuples (t, real_part, imag_part, magnitude).
    """
    return get_backend().riemann_zeta_critical(t_samples, terms)


def find_zeta_zeros_approx(
    t_start: float,
    t_end: float,
    steps: int = 200,
    terms: int = 1000,
) -> List[Tuple[float, float]]:
    """Scan for approximate locations of non-trivial zeros along the critical line s = 1/2 + i*t.

    Identifies local minima of the modulus |zeta(1/2 + i*t)| below a threshold.

    Args:
        t_start: Starting height t.
        t_end: Ending height t.
        steps: Number of sampling points.
        terms: Number of expansion terms.

    Returns:
        List of tuples (approx_t, min_magnitude).
    """
    if steps < 3 or t_end <= t_start:
        raise ValueError("Invalid range or insufficient steps.")

    dt = (t_end - t_start) / (steps - 1)
    t_vals = [t_start + i * dt for i in range(steps)]
    samples = riemann_zeta_critical_line(t_vals, terms)

    zeros: List[Tuple[float, float]] = []
    # Identify local minima
    for i in range(1, len(samples) - 1):
        prev_mag = samples[i - 1][3]
        curr_mag = samples[i][3]
        next_mag = samples[i + 1][3]
        if curr_mag < prev_mag and curr_mag < next_mag and curr_mag < 0.5:
            zeros.append((samples[i][0], curr_mag))

    return zeros
