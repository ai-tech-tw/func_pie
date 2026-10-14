"""Lattice Theory and Lenstra-Lenstra-Lovasz (LLL) Basis Reduction.

Implements polynomial-time lattice basis reduction and shortest vector
approximation used in algebraic number theory, Diophantine approximation,
and cryptography.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from funcpie.backend import get_backend


def dot_product(v1: List[float], v2: List[float]) -> float:
    """Compute dot product of two vectors."""
    return sum(x * y for x, y in zip(v1, v2))


def vector_norm(v: List[float]) -> float:
    """Compute Euclidean L2 norm of a vector."""
    return math.sqrt(dot_product(v, v))


def gram_schmidt(basis: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """Gram-Schmidt orthogonalization of a vector basis.

    Returns:
        Tuple of (orthogonal_basis b_star, mu coefficients matrix).
    """
    n = len(basis)
    if n == 0:
        return [], []
    m = len(basis[0])
    b_star: List[List[float]] = []
    mu: List[List[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        v = list(basis[i])
        for j in range(i):
            denom = dot_product(b_star[j], b_star[j])
            if abs(denom) > 1e-15:
                mu[i][j] = dot_product(basis[i], b_star[j]) / denom
            else:
                mu[i][j] = 0.0
            for d in range(m):
                v[d] -= mu[i][j] * b_star[j][d]
        b_star.append(v)
        mu[i][i] = 1.0

    return b_star, mu


def lll_reduction(basis: List[List[float]], delta: float = 0.75) -> List[List[float]]:
    """Perform Lenstra-Lenstra-Lovasz (LLL) lattice basis reduction.

    Args:
        basis: List of basis vectors representing the lattice.
        delta: Reduction parameter in (0.25, 1.0), typically 0.75.

    Returns:
        LLL-reduced basis vectors.
    """
    return get_backend().lll_reduction(basis, delta)


def lattice_determinant(basis: List[List[float]]) -> float:
    """Compute lattice determinant det(Lambda) = prod ||b_i*||."""
    if not basis:
        return 0.0
    b_star, _ = gram_schmidt(basis)
    det = 1.0
    for v in b_star:
        det *= vector_norm(v)
    return det


def shortest_vector_bound(basis: List[List[float]]) -> float:
    """Calculate Minkowski's first theorem upper bound on the shortest non-zero vector length:

    lambda_1(Lambda) <= sqrt(n) * det(Lambda)^(1/n).
    """
    n = len(basis)
    if n == 0:
        return 0.0
    det = lattice_determinant(basis)
    if det <= 0.0:
        return 0.0
    return math.sqrt(float(n)) * (det ** (1.0 / float(n)))


__all__ = [
    "dot_product",
    "vector_norm",
    "gram_schmidt",
    "lll_reduction",
    "lattice_determinant",
    "shortest_vector_bound",
]
