"""Lorenz Attractor Dynamical System and Lyapunov Exponent Estimation."""

from __future__ import annotations

import math
from typing import List, Tuple

from funcpie.backend import get_backend


def lorenz_step(
    state: Tuple[float, float, float],
    dt: float,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> Tuple[float, float, float]:
    """Single RK4 integration step for the Lorenz system."""
    res = get_backend().lorenz_ensemble([state], dt, 1, sigma, rho, beta)
    return res[0]


def lorenz_ensemble(
    initial_states: List[Tuple[float, float, float]],
    dt: float,
    steps: int,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> List[Tuple[float, float, float]]:
    """Evolve an ensemble of initial states under the Lorenz attractor equations in parallel.

    Args:
        initial_states: List of (x, y, z) initial states.
        dt: Integration step.
        steps: Number of integration steps.
        sigma: Prandtl number (default 10.0).
        rho: Rayleigh number (default 28.0).
        beta: Geometric aspect ratio (default 8/3).

    Returns:
        List of evolved (x, y, z) states.
    """
    return get_backend().lorenz_ensemble(initial_states, dt, steps, sigma, rho, beta)


def lyapunov_exponent_estimate(
    state: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    perturbation: float = 1e-6,
    dt: float = 0.01,
    steps: int = 1000,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> float:
    """Estimate the maximal Lyapunov exponent to quantify chaotic divergence (the butterfly effect).

    Positive Lyapunov exponent indicates chaos and sensitive dependence on initial conditions.

    Args:
        state: Base initial state.
        perturbation: Initial distance between perturbed trajectory.
        dt: Timestep.
        steps: Total integration steps.
        sigma: Lorenz sigma.
        rho: Lorenz rho.
        beta: Lorenz beta.

    Returns:
        Estimated maximal Lyapunov exponent lambda.
    """
    perturbed = (state[0] + perturbation, state[1], state[2])
    states = [state, perturbed]

    total_time = dt * steps
    evolved = lorenz_ensemble(states, dt, steps, sigma, rho, beta)

    p1, p2 = evolved[0], evolved[1]
    final_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    if final_dist <= 0.0 or perturbation <= 0.0:
        return 0.0

    return (1.0 / total_time) * math.log(final_dist / perturbation)
