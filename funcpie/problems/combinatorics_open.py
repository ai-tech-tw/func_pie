"""Extremal Combinatorics and Ramsey Theory verification module.

Provides algorithms for testing Ramsey graph properties, searching for counterexample
graphs (avoiding k-cliques and l-independent sets), and checking theoretical bounds
for Ramsey numbers R(k, l).
"""

from __future__ import annotations

import random
from typing import List, Optional

from funcpie.backend import get_backend


def is_ramsey_graph(adj_matrix: List[List[int]], k: int, l: int) -> bool:
    """Check if an adjacency matrix represents a valid (k, l)-Ramsey graph.

    A valid (k, l)-Ramsey graph contains neither a clique of size k nor
    an independent set of size l, proving R(k, l) > n.
    """
    return get_backend().ramsey_check(adj_matrix, k, l)


def ramsey_edge_density(adj_matrix: List[List[int]]) -> float:
    """Compute the edge density of an undirected graph."""
    n = len(adj_matrix)
    if n <= 1:
        return 0.0
    total_edges = sum(sum(row) for row in adj_matrix) // 2
    max_edges = n * (n - 1) // 2
    return float(total_edges) / float(max_edges)


def ramsey_random_search(
    n: int,
    k: int,
    l: int,
    trials: int = 1000,
    seed: Optional[int] = None,
) -> Optional[List[List[int]]]:
    """Search for an n-vertex graph containing no k-clique and no l-independent set.

    Args:
        n: Number of vertices.
        k: Maximum prohibited clique size.
        l: Maximum prohibited independent set size.
        trials: Maximum stochastic trials.
        seed: Optional RNG seed.

    Returns:
        Adjacency matrix if found, None otherwise.
    """
    if seed is not None:
        random.seed(seed)

    for _ in range(trials):
        # Generate random Erdős–Rényi G(n, 0.5) graph
        adj = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                val = 1 if random.random() < 0.5 else 0
                adj[i][j] = adj[j][i] = val

        if is_ramsey_graph(adj, k, l):
            return adj

    return None


__all__ = [
    "is_ramsey_graph",
    "ramsey_edge_density",
    "ramsey_random_search",
]
