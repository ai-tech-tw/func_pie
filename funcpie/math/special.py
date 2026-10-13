"""Special mathematical and activation functions."""

from __future__ import annotations

import math
from typing import List, Sequence, Union

Number = Union[int, float]


def gamma(x: float) -> float:
    """Gamma function Gamma(x)."""
    return math.gamma(x)


def beta(a: float, b: float) -> float:
    """Beta function B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b).

    Computed via log-gamma to avoid intermediate numerical overflow.
    """
    if a <= 0 or b <= 0:
        raise ValueError("Beta function requires positive arguments")
    log_b = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return math.exp(log_b)


def erf(x: float) -> float:
    """Gauss error function."""
    return math.erf(x)


def erfc(x: float) -> float:
    """Complementary error function 1 - erf(x)."""
    return math.erfc(x)


def sigmoid(x: float) -> float:
    """Standard logistic sigmoid function: 1 / (1 + e^(-x))."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def relu(x: float) -> float:
    """Rectified Linear Unit: max(0, x)."""
    return max(0.0, float(x))


def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """Leaky Rectified Linear Unit."""
    return float(x) if x > 0 else alpha * float(x)


def log_sum_exp(vector: Sequence[float]) -> float:
    """Numerically stable log(sum(exp(x_i)))."""
    if not vector:
        raise ValueError("Vector cannot be empty")
    max_val = max(vector)
    return max_val + math.log(sum(math.exp(x - max_val) for x in vector))


def softmax(vector: Sequence[float]) -> List[float]:
    """Numerically stable softmax probability distribution."""
    if not vector:
        raise ValueError("Vector cannot be empty")
    max_val = max(vector)
    exp_vals = [math.exp(x - max_val) for x in vector]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]
