"""Goldbach's Conjecture Solver and Batch Partitioner."""

from __future__ import annotations

import math
from typing import List, Tuple

from funcpie.backend import get_backend


def goldbach_partition(even_number: int) -> Tuple[int, int]:
    """Find the minimal prime pair (p1, p2) such that p1 + p2 == even_number.

    Args:
        even_number: Even integer >= 4.

    Returns:
        Tuple (p1, p2) of primes.

    Raises:
        ValueError: If even_number < 4 or is odd.
    """
    if even_number < 4 or even_number % 2 != 0:
        raise ValueError("even_number must be an even integer >= 4.")

    res = get_backend().goldbach_batch(even_number, 1)
    if not res or res[0][1] == -1:
        raise RuntimeError(f"No Goldbach partition found for {even_number}!")
    return res[0][1], res[0][2]


def goldbach_batch_verify(start_even: int, count: int) -> List[Tuple[int, int, int]]:
    """Verify Goldbach's conjecture for consecutive even integers starting at start_even.

    Args:
        start_even: Starting even integer >= 4.
        count: Number of consecutive even integers to verify.

    Returns:
        List of tuples (even_number, p1, p2).
    """
    return get_backend().goldbach_batch(start_even, count)


def goldbach_count_partitions(even_number: int) -> int:
    """Count the total number of distinct prime pairs (p1, p2) with p1 <= p2 and p1 + p2 == even_number."""
    if even_number < 4 or even_number % 2 != 0:
        raise ValueError("even_number must be an even integer >= 4.")

    limit = int(math.isqrt(even_number))
    is_prime = [True] * (even_number + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, limit + 1):
        if is_prime[i]:
            for j in range(i * i, even_number + 1, i):
                is_prime[j] = False

    count = 0
    for p in range(2, even_number // 2 + 1):
        if is_prime[p] and is_prime[even_number - p]:
            count += 1
    return count
