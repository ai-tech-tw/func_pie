"""Collatz Conjecture (3n + 1 Problem) Solver and Batch Verifier."""

from __future__ import annotations

from typing import List, Tuple

from funcpie.backend import get_backend


def collatz_sequence(n: int, max_steps: int = 100000) -> List[int]:
    """Generate the full Collatz trajectory sequence for a given positive integer n.

    Args:
        n: Positive integer starting value.
        max_steps: Maximum steps allowed before termination.

    Returns:
        List of integers in the trajectory ending at 1.

    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")

    seq: List[int] = [n]
    curr = n
    steps = 0
    while curr > 1 and steps < max_steps:
        if curr % 2 == 0:
            curr //= 2
        else:
            curr = 3 * curr + 1
        seq.append(curr)
        steps += 1
    return seq


def collatz_batch_verify(
    start: int,
    count: int,
    max_steps: int = 100000,
) -> List[Tuple[int, int, int]]:
    """Batch-verify the Collatz conjecture for a range of integers with GPU/CPU acceleration.

    Args:
        start: Starting positive integer.
        count: Number of consecutive integers to verify.
        max_steps: Maximum steps allowed before giving up.

    Returns:
        List of tuples (n, stopping_time, peak_value) for each integer.
    """
    backend = get_backend()
    return backend.collatz_batch(start, count, max_steps)


def collatz_longest_chain(
    start: int,
    count: int,
    max_steps: int = 100000,
) -> Tuple[int, int, int]:
    """Find the integer in [start, start + count - 1] that takes the longest steps to reach 1.

    Returns:
        Tuple of (best_n, max_steps, peak_value).
    """
    results = collatz_batch_verify(start, count, max_steps)
    if not results:
        raise ValueError("Empty batch range.")
    return max(results, key=lambda item: item[1])
