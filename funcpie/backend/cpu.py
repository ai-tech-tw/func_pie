"""CPU fallback backend implementing numeric operations in pure Python."""

from __future__ import annotations

import math
from typing import List, Tuple


class CPUBackend:
    """Pure Python CPU execution engine for numeric and scientific operations."""

    name: str = "cpu"

    @staticmethod
    def is_available() -> bool:
        """CPU backend is always available."""
        return True

    @staticmethod
    def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Compute matrix multiplication C = A x B on CPU."""
        rows_a = len(a)
        cols_a = len(a[0]) if rows_a > 0 else 0
        rows_b = len(b)
        cols_b = len(b[0]) if rows_b > 0 else 0

        if cols_a != rows_b:
            raise ValueError(f"Cannot multiply matrix with shapes ({rows_a}, {cols_a}) and ({rows_b}, {cols_b})")

        # Transpose b for cache-friendly row-by-row dot product
        b_t: List[List[float]] = [[b[r][c] for r in range(rows_b)] for c in range(cols_b)]
        result: List[List[float]] = []
        for row_a in a:
            res_row: List[float] = [
                sum(x * y for x, y in zip(row_a, col_b))
                for col_b in b_t
            ]
            result.append(res_row)
        return result

    @staticmethod
    def vector_dot(v1: List[float], v2: List[float]) -> float:
        """Compute dot product of two vectors on CPU."""
        if len(v1) != len(v2):
            raise ValueError(f"Vector length mismatch: {len(v1)} != {len(v2)}")
        return sum(x * y for x, y in zip(v1, v2))

    @staticmethod
    def vector_add(v1: List[float], v2: List[float]) -> List[float]:
        """Compute vector addition on CPU."""
        if len(v1) != len(v2):
            raise ValueError(f"Vector length mismatch: {len(v1)} != {len(v2)}")
        return [x + y for x, y in zip(v1, v2)]

    @staticmethod
    def simpson_integrate_samples(samples: List[float], h: float) -> float:
        """Composite Simpson's rule over evenly spaced sample points on CPU."""
        n = len(samples) - 1
        if n <= 0 or n % 2 != 0:
            raise ValueError("Simpson's rule requires an even number of intervals (odd number of samples)")

        total = samples[0] + samples[-1]
        for i in range(1, n, 2):
            total += 4.0 * samples[i]
        for i in range(2, n, 2):
            total += 2.0 * samples[i]
        return total * (h / 3.0)

    @staticmethod
    def batch_projectile(
        v0_list: List[float],
        angle_deg_list: List[float],
        g: float = 9.80665,
    ) -> List[Tuple[float, float, float]]:
        """Compute projectile motion results (max_height, flight_time, range) for batches."""
        if len(v0_list) != len(angle_deg_list):
            raise ValueError("Mismatched batch lengths for v0 and angles")

        results: List[Tuple[float, float, float]] = []
        for v0, angle_deg in zip(v0_list, angle_deg_list):
            rad = math.radians(angle_deg)
            vy0 = v0 * math.sin(rad)
            vx0 = v0 * math.cos(rad)
            t_flight = 2.0 * vy0 / g if vy0 > 0 else 0.0
            h_max = (vy0 ** 2) / (2.0 * g) if vy0 > 0 else 0.0
            h_range = vx0 * t_flight
            results.append((h_max, t_flight, h_range))
        return results
