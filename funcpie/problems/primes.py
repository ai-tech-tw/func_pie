"""Mersenne and Twin Primes Algorithms."""

from __future__ import annotations

from typing import List, Tuple

from funcpie.backend import get_backend


def mersenne_number(p: int) -> int:
    """Calculate the Mersenne number M_p = 2^p - 1.

    Args:
        p: Prime exponent.

    Returns:
        Integer value 2^p - 1.
    """
    if p < 1:
        raise ValueError("p must be a positive integer.")
    return (1 << p) - 1


def lucas_lehmer_mersenne(p: int) -> bool:
    """Lucas-Lehmer primality test for Mersenne number M_p = 2^p - 1.

    Determines if M_p is a Mersenne prime with deterministic mathematical certainty.

    Args:
        p: Exponent (typically prime).

    Returns:
        True if 2^p - 1 is prime, False otherwise.
    """
    return get_backend().lucas_lehmer(p)


def segmented_prime_sieve(low: int, high: int) -> List[int]:
    """Segmented Sieve of Eratosthenes to efficiently find all primes in [low, high].

    Args:
        low: Lower bound of the interval.
        high: Upper bound of the interval.

    Returns:
        List of prime numbers in the range.
    """
    return get_backend().segmented_sieve(low, high)


def twin_primes_scan(low: int, high: int) -> List[Tuple[int, int]]:
    """Search for all twin prime pairs (p, p + 2) within the range [low, high].

    Args:
        low: Lower bound.
        high: Upper bound.

    Returns:
        List of tuples (p, p + 2) of twin primes.
    """
    primes = segmented_prime_sieve(max(2, low), high)
    twins: List[Tuple[int, int]] = []
    prime_set = set(primes)
    for p in primes:
        if (p + 2) in prime_set and (p + 2) <= high:
            twins.append((p, p + 2))
    return twins
