"""Traveling Salesperson Problem (TSP) Parallel Simulated Annealing Solver."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from funcpie.backend import get_backend


def tsp_tour_distance(coords: List[Tuple[float, float]], tour: List[int]) -> float:
    """Compute the total Euclidean closed-loop distance of a given TSP city tour.

    Args:
        coords: List of (x, y) coordinates for cities.
        tour: Order of city indices visited.

    Returns:
        Total path length.
    """
    n = len(tour)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        c1 = coords[tour[i]]
        c2 = coords[tour[(i + 1) % n]]
        total += math.hypot(c1[0] - c2[0], c1[1] - c2[1])
    return total


def tsp_simulated_annealing(
    coords: List[Tuple[float, float]],
    num_walkers: int = 16,
    steps: int = 500,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.95,
    seed: Optional[int] = 42,
) -> Tuple[List[int], float]:
    """Solve the Traveling Salesperson Problem using parallel multi-walker simulated annealing.

    Employs 2-opt neighborhood permutations with probabilistic Metropolis acceptance,
    searching for global minimum distance across multiple concurrent walkers.

    Args:
        coords: List of (x, y) 2D city coordinates.
        num_walkers: Number of concurrent stochastic search walkers.
        steps: Annealing temperature cooling iterations per walker.
        initial_temp: Initial temperature T0.
        cooling_rate: Multiplicative temperature decay factor alpha in [0.8, 0.99].
        seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (best_tour_indices, best_distance).
    """
    return get_backend().tsp_simulated_annealing(
        coords,
        num_walkers=num_walkers,
        steps=steps,
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
        seed=seed,
    )
