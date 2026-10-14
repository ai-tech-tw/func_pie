"""Numerical calculus: differentiation, integration, and root finding."""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Union

ScalarFunc = Callable[[float], float]


def derivative(
    f: ScalarFunc,
    x: float,
    h: float = 1e-5,
    method: str = "central",
) -> float:
    """Compute numerical first derivative of f at x.

    Methods:
        'central': (f(x+h) - f(x-h)) / (2h)   [O(h^2)]
        'forward': (f(x+h) - f(x)) / h        [O(h)]
        'backward': (f(x) - f(x-h)) / h       [O(h)]
    """
    if method == "central":
        return (f(x + h) - f(x - h)) / (2.0 * h)
    elif method == "forward":
        return (f(x + h) - f(x)) / h
    elif method == "backward":
        return (f(x) - f(x - h)) / h
    else:
        raise ValueError(f"Unknown method '{method}', choose from 'central', 'forward', 'backward'")


def second_derivative(f: ScalarFunc, x: float, h: float = 1e-4) -> float:
    """Compute numerical second derivative of f at x using central differences:

    f''(x) ~ (f(x+h) - 2f(x) + f(x-h)) / h^2
    """
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)


def gradient(
    f: Callable[[Sequence[float]], float],
    point: Sequence[float],
    h: float = 1e-5,
) -> List[float]:
    """Compute numerical gradient of multivariable function f at given point."""
    pt = list(point)
    grad = []
    for i in range(len(pt)):
        pt_forward = list(pt)
        pt_backward = list(pt)
        pt_forward[i] += h
        pt_backward[i] -= h
        df_dxi = (f(pt_forward) - f(pt_backward)) / (2.0 * h)
        grad.append(df_dxi)
    return grad


def trapezoid_rule(f: ScalarFunc, a: float, b: float, n: int = 1000) -> float:
    """Approximate definite integral of f from a to b using Trapezoidal rule."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if a == b:
        return 0.0
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h


def simpsons_rule(f: ScalarFunc, a: float, b: float, n: int = 1000) -> float:
    """Approximate definite integral of f from a to b using Simpson's 1/3 rule.

    n must be an even integer.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n % 2 != 0:
        n += 1  # Simpson's rule requires even number of intervals
    if a == b:
        return 0.0

    h = (b - a) / n
    from funcpie.backend import get_backend
    samples = [f(a + i * h) for i in range(n + 1)]
    return get_backend().simpson_integrate_samples(samples, h)


def integrate(
    f: ScalarFunc,
    a: float,
    b: float,
    method: str = "simpson",
    n: int = 1000,
) -> float:
    """Compute definite integral of f over [a, b].

    method can be 'simpson' or 'trapezoid'.
    """
    if method == "simpson":
        return simpsons_rule(f, a, b, n=n)
    elif method == "trapezoid":
        return trapezoid_rule(f, a, b, n=n)
    else:
        raise ValueError(f"Unknown integration method: '{method}'")


def monte_carlo_integrate(
    f: ScalarFunc,
    a: float,
    b: float,
    samples: int = 10000,
    seed: Optional[int] = None,
) -> float:
    """Approximate definite integral of f from a to b using Monte Carlo sampling."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if seed is not None:
        random.seed(seed)
    total = sum(f(random.uniform(a, b)) for _ in range(samples))
    return (b - a) * (total / samples)


def find_root_bisection(
    f: ScalarFunc,
    a: float,
    b: float,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> float:
    """Find a root of f(x) = 0 in interval [a, b] using the Bisection method.

    f(a) and f(b) must have opposite signs.
    """
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for bisection")

    low, high = a, b
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        fmid = f(mid)
        if abs(fmid) < tol or (high - low) / 2.0 < tol:
            return mid
        if fa * fmid < 0:
            high = mid
            fb = fmid
        else:
            low = mid
            fa = fmid
    return (low + high) / 2.0


def find_root_newton(
    f: ScalarFunc,
    x0: float,
    f_prime: Optional[ScalarFunc] = None,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> float:
    """Find a root of f(x) = 0 using the Newton-Raphson method.

    If f_prime is not provided, numerical differentiation is used.
    """
    x = float(x0)
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = f_prime(x) if f_prime is not None else derivative(f, x)
        if dfx == 0.0:
            raise ZeroDivisionError("Derivative became zero during Newton-Raphson iteration")
        x_next = x - fx / dfx
        if abs(x_next - x) < tol:
            return x_next
        x = x_next
    return x


def find_root_secant(
    f: ScalarFunc,
    x0: float,
    x1: float,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> float:
    """Find a root of f(x) = 0 using the Secant method."""
    p0, p1 = float(x0), float(x1)
    q0, q1 = f(p0), f(p1)
    for _ in range(max_iter):
        if abs(q1 - q0) < 1e-15:
            raise ZeroDivisionError("Secant slope became too close to zero")
        p = p1 - q1 * (p1 - p0) / (q1 - q0)
        if abs(p - p1) < tol:
            return p
        p0, q0 = p1, q1
        p1 = p
        q1 = f(p1)
    return p1
