"""Modular Forms and L-Functions verification suite.

Focuses on the discriminant modular form Delta(z) = q prod_{n=1}^inf (1 - q^n)^24,
Ramanujan's tau function, Hecke multiplicativity, and Pierre Deligne's bound
|tau(p)| <= 2 * p^(11/2) (Fields Medal, 1978).
"""

from __future__ import annotations

import math
from typing import List, Tuple

from funcpie.backend import get_backend


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor."""
    while b != 0:
        a, b = b, a % b
    return abs(a)


def ramanujan_tau(n: int) -> int:
    """Compute Ramanujan's tau function tau(n) for positive integer n.

    tau(n) is the n-th Fourier coefficient of the weight 12 cusp form Delta(z).
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    taus = get_backend().ramanujan_tau_batch(n)
    return taus[n - 1]


def ramanujan_tau_batch(limit: int) -> List[int]:
    """Compute Ramanujan tau values tau(1), ..., tau(limit)."""
    if limit < 1:
        return []
    return get_backend().ramanujan_tau_batch(limit)


def verify_multiplicative(m: int, n: int) -> bool:
    """Verify Hecke multiplicativity tau(m * n) == tau(m) * tau(n) for coprime m, n."""
    if m < 1 or n < 1:
        raise ValueError("m and n must be positive integers.")
    if gcd(m, n) != 1:
        raise ValueError(f"m={m} and n={n} are not coprime (gcd={gcd(m, n)}).")

    tau_mn = ramanujan_tau(m * n)
    tau_m = ramanujan_tau(m)
    tau_n = ramanujan_tau(n)
    return tau_mn == (tau_m * tau_n)


def verify_deligne_bound(p: int) -> Tuple[bool, int, float]:
    """Verify Deligne's theorem (Ramanujan conjecture) |tau(p)| <= 2 * p^(11/2) for prime p.

    Returns:
        Tuple of (is_valid, tau_p, bound).
    """
    if p < 2:
        raise ValueError(f"p must be prime, got {p}")
    tau_p = ramanujan_tau(p)
    bound = 2.0 * (float(p) ** 5.5)
    is_valid = abs(tau_p) <= bound
    return (is_valid, tau_p, bound)


__all__ = [
    "gcd",
    "ramanujan_tau",
    "ramanujan_tau_batch",
    "verify_multiplicative",
    "verify_deligne_bound",
]
