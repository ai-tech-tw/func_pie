"""Algebra, number theory, polynomials, and combinatorics."""

from __future__ import annotations

import cmath
import functools
import math
from typing import Dict, List, Sequence, Tuple, Union

Number = Union[int, float]
ComplexNumber = Union[int, float, complex]


def gcd(*numbers: int) -> int:
    """Calculate the greatest common divisor of one or more integers.

    Example:
        >>> gcd(48, 180, 24)
        12
    """
    if not numbers:
        raise ValueError("gcd requires at least one integer")
    return functools.reduce(math.gcd, numbers)


def lcm(*numbers: int) -> int:
    """Calculate the least common multiple of one or more integers.

    Example:
        >>> lcm(12, 18, 30)
        180
    """
    if not numbers:
        raise ValueError("lcm requires at least one integer")

    def _lcm(a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // math.gcd(a, b)

    return functools.reduce(_lcm, numbers)


def is_prime(n: int) -> bool:
    """Check if an integer n is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def primes_sieve(limit: int) -> List[int]:
    """Generate all prime numbers up to `limit` using the Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(math.isqrt(limit)) + 1):
        if sieve[p]:
            for multiple in range(p * p, limit + 1, p):
                sieve[multiple] = False
    return [i for i, prime in enumerate(sieve) if prime]


def prime_factors(n: int) -> Dict[int, int]:
    """Return dictionary of prime factors and their multiplicities.

    Example:
        >>> prime_factors(360)
        {2: 3, 3: 2, 5: 1}
    """
    if n <= 1:
        raise ValueError("n must be greater than 1 to factorize into primes")
    factors: Dict[int, int] = {}
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                count += 1
                temp //= d
            factors[d] = count
        d = 3 if d == 2 else d + 2
    if temp > 1:
        factors[temp] = 1
    return factors


def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (0-indexed: F(0)=0, F(1)=1, F(2)=1...)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def factorial(n: int) -> int:
    """Return n factorial (n!)."""
    if n < 0:
        raise ValueError("factorial only defined for non-negative integers")
    return math.factorial(n)


def permutations_count(n: int, r: int) -> int:
    """Compute nPr = n! / (n - r)!"""
    if n < 0 or r < 0 or r > n:
        raise ValueError("Requires 0 <= r <= n")
    return math.perm(n, r) if hasattr(math, "perm") else math.factorial(n) // math.factorial(n - r)


def combinations_count(n: int, r: int) -> int:
    """Compute nCr = n! / (r! * (n - r)!)"""
    if n < 0 or r < 0 or r > n:
        raise ValueError("Requires 0 <= r <= n")
    return math.comb(n, r) if hasattr(math, "comb") else math.factorial(n) // (
        math.factorial(r) * math.factorial(n - r)
    )


binomial_coefficient = combinations_count


def solve_linear(a: Number, b: Number) -> float:
    """Solve linear equation a*x + b = 0.

    Returns:
        Root x as float.

    Raises:
        ValueError: If a == 0 (no unique solution).
    """
    if a == 0:
        raise ValueError("Linear coefficient 'a' cannot be 0")
    return -float(b) / float(a)


def solve_quadratic(a: Number, b: Number, c: Number) -> Tuple[ComplexNumber, ComplexNumber]:
    """Solve quadratic equation a*x^2 + b*x + c = 0.

    Returns:
        Tuple of two roots (real or complex).
    """
    if a == 0:
        raise ValueError("Quadratic coefficient 'a' cannot be 0")
    disc = (b ** 2) - (4 * a * c)
    if disc >= 0:
        sqrt_disc = math.sqrt(disc)
        x1 = (-b + sqrt_disc) / (2 * a)
        x2 = (-b - sqrt_disc) / (2 * a)
        return (x1, x2)
    else:
        sqrt_disc = cmath.sqrt(disc)
        x1 = (-b + sqrt_disc) / (2 * a)
        x2 = (-b - sqrt_disc) / (2 * a)
        return (x1, x2)


class Polynomial:
    """Polynomial representation: P(x) = c[n]*x^n + ... + c[1]*x + c[0].

    Coefficients can be provided highest power first (default):
        Polynomial([3, 2, 1]) -> 3x^2 + 2x + 1
    """

    def __init__(self, coeffs: Sequence[Number]):
        clean = list(coeffs)
        # Strip leading zeros
        while len(clean) > 1 and clean[0] == 0:
            clean.pop(0)
        self.coeffs: List[float] = [float(c) for c in (clean if clean else [0.0])]

    @property
    def degree(self) -> int:
        """Degree of the polynomial."""
        return len(self.coeffs) - 1

    def __call__(self, x: Number) -> float:
        """Evaluate polynomial at x using Horner's method."""
        res = 0.0
        for c in self.coeffs:
            res = res * x + c
        return res

    def derivative(self) -> Polynomial:
        """Compute the first derivative of the polynomial."""
        deg = self.degree
        if deg == 0:
            return Polynomial([0.0])
        new_coeffs = [
            self.coeffs[i] * (deg - i)
            for i in range(deg)
        ]
        return Polynomial(new_coeffs)

    def integrate(self, c: Number = 0.0) -> Polynomial:
        """Compute the indefinite integral of the polynomial with integration constant c."""
        deg = self.degree
        new_coeffs = [
            self.coeffs[i] / (deg - i + 1)
            for i in range(deg + 1)
        ] + [float(c)]
        return Polynomial(new_coeffs)

    def __add__(self, other: Union[Polynomial, Number]) -> Polynomial:
        if isinstance(other, (int, float)):
            other = Polynomial([other])
        d1, d2 = len(self.coeffs), len(other.coeffs)
        pad = abs(d1 - d2)
        if d1 < d2:
            c1 = [0.0] * pad + list(self.coeffs)
            c2 = list(other.coeffs)
        else:
            c1 = list(self.coeffs)
            c2 = [0.0] * pad + list(other.coeffs)
        return Polynomial([a + b for a, b in zip(c1, c2)])

    def __radd__(self, other: Number) -> Polynomial:
        return self.__add__(other)

    def __sub__(self, other: Union[Polynomial, Number]) -> Polynomial:
        if isinstance(other, (int, float)):
            other = Polynomial([other])
        return self + Polynomial([-c for c in other.coeffs])

    def __rsub__(self, other: Number) -> Polynomial:
        return Polynomial([other]) - self

    def __mul__(self, other: Union[Polynomial, Number]) -> Polynomial:
        if isinstance(other, (int, float)):
            return Polynomial([c * other for c in self.coeffs])
        result = [0.0] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, c1 in enumerate(self.coeffs):
            for j, c2 in enumerate(other.coeffs):
                result[i + j] += c1 * c2
        return Polynomial(result)

    def __rmul__(self, other: Number) -> Polynomial:
        return self.__mul__(other)

    def roots(self) -> List[ComplexNumber]:
        """Compute roots for degree 1 or 2 polynomials."""
        if self.degree == 0:
            return []
        if self.degree == 1:
            return [solve_linear(self.coeffs[0], self.coeffs[1])]
        if self.degree == 2:
            return list(solve_quadratic(self.coeffs[0], self.coeffs[1], self.coeffs[2]))
        raise NotImplementedError("Analytical roots for degree > 2 not implemented; use numerical calculus")

    def __repr__(self) -> str:
        parts = []
        deg = self.degree
        for i, c in enumerate(self.coeffs):
            power = deg - i
            if c == 0 and deg > 0:
                continue
            coeff_str = f"{c:g}"
            if power == 0:
                parts.append(coeff_str)
            elif power == 1:
                parts.append(f"{coeff_str}x")
            else:
                parts.append(f"{coeff_str}x^{power}")
        return " + ".join(parts).replace("+ -", "- ") or "0"
