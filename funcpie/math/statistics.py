"""Statistical analysis, distributions, and regression utilities."""

from __future__ import annotations

import collections
import math
from typing import Dict, List, NamedTuple, Sequence, Tuple, Union

Number = Union[int, float]


class RegressionResult(NamedTuple):
    slope: float
    intercept: float
    r_value: float
    r_squared: float


def mean(data: Sequence[Number]) -> float:
    """Arithmetic mean of numbers."""
    if not data:
        raise ValueError("Cannot calculate mean of empty sequence")
    return sum(data) / float(len(data))


def geometric_mean(data: Sequence[Number]) -> float:
    """Geometric mean of positive numbers."""
    if not data:
        raise ValueError("Cannot calculate geometric mean of empty sequence")
    if any(x <= 0 for x in data):
        raise ValueError("Geometric mean requires all positive values")
    log_sum = sum(math.log(x) for x in data)
    return math.exp(log_sum / len(data))


def harmonic_mean(data: Sequence[Number]) -> float:
    """Harmonic mean of positive numbers."""
    if not data:
        raise ValueError("Cannot calculate harmonic mean of empty sequence")
    if any(x <= 0 for x in data):
        raise ValueError("Harmonic mean requires all positive values")
    return len(data) / sum(1.0 / x for x in data)


def median(data: Sequence[Number]) -> float:
    """Median value of numbers."""
    if not data:
        raise ValueError("Cannot calculate median of empty sequence")
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def mode(data: Sequence[Number]) -> Number:
    """Most frequent value in data (returns first found if tied)."""
    if not data:
        raise ValueError("Cannot calculate mode of empty sequence")
    counts = collections.Counter(data)
    return counts.most_common(1)[0][0]


def variance(data: Sequence[Number], sample: bool = True) -> float:
    """Variance of data (sample variance n-1 or population variance n)."""
    n = len(data)
    min_count = 2 if sample else 1
    if n < min_count:
        raise ValueError(f"Variance requires at least {min_count} data points")
    m = mean(data)
    sq_diff = sum((x - m) ** 2 for x in data)
    return sq_diff / (n - 1 if sample else n)


def stdev(data: Sequence[Number], sample: bool = True) -> float:
    """Standard deviation of data."""
    return math.sqrt(variance(data, sample=sample))


def z_scores(data: Sequence[Number]) -> List[float]:
    """Standardize data into z-scores (x - mu) / sigma."""
    m = mean(data)
    s = stdev(data, sample=True)
    if s == 0:
        return [0.0 for _ in data]
    return [(x - m) / s for x in data]


def percentile(data: Sequence[Number], p: float) -> float:
    """Compute p-th percentile of data (p in [0, 100]) using linear interpolation."""
    if not data:
        raise ValueError("data cannot be empty")
    if not 0.0 <= p <= 100.0:
        raise ValueError("p must be between 0 and 100")
    s = sorted(data)
    if len(s) == 1:
        return float(s[0])
    rank = (p / 100.0) * (len(s) - 1)
    k = int(rank)
    d = rank - k
    if k + 1 < len(s):
        return s[k] + d * (s[k + 1] - s[k])
    return float(s[-1])


def quantiles(data: Sequence[Number], n: int = 4) -> List[float]:
    """Divide data into n continuous intervals with equal probability."""
    if n < 1:
        raise ValueError("n must be at least 1")
    return [percentile(data, (i * 100.0) / n) for i in range(1, n)]


def skewness(data: Sequence[Number]) -> float:
    """Sample skewness (Fisher-Pearson coefficient)."""
    n = len(data)
    if n < 3:
        raise ValueError("Skewness requires at least 3 data points")
    m = mean(data)
    s = stdev(data, sample=True)
    if s == 0:
        return 0.0
    m3 = sum((x - m) ** 3 for x in data) / n
    return m3 / (s ** 3)


def kurtosis(data: Sequence[Number], excess: bool = True) -> float:
    """Kurtosis of data (excess kurtosis relative to normal distribution if excess=True)."""
    n = len(data)
    if n < 4:
        raise ValueError("Kurtosis requires at least 4 data points")
    m = mean(data)
    s = stdev(data, sample=False)
    if s == 0:
        return 0.0
    m4 = sum((x - m) ** 4 for x in data) / n
    k = m4 / (s ** 4)
    return k - 3.0 if excess else k


def covariance(x: Sequence[Number], y: Sequence[Number], sample: bool = True) -> float:
    """Sample or population covariance between two variables."""
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    n = len(x)
    min_count = 2 if sample else 1
    if n < min_count:
        raise ValueError(f"Covariance requires at least {min_count} data points")
    mx, my = mean(x), mean(y)
    total = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return total / (n - 1 if sample else n)


def correlation(x: Sequence[Number], y: Sequence[Number]) -> float:
    """Pearson correlation coefficient r."""
    cov = covariance(x, y, sample=True)
    sx = stdev(x, sample=True)
    sy = stdev(y, sample=True)
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def linear_regression(x: Sequence[Number], y: Sequence[Number]) -> RegressionResult:
    """Ordinary least squares simple linear regression y = slope * x + intercept."""
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if len(x) < 2:
        raise ValueError("Linear regression requires at least 2 points")
    cov = covariance(x, y, sample=True)
    var_x = variance(x, sample=True)
    if var_x == 0:
        raise ValueError("Variance of x is zero; vertical line cannot be fitted")
    slope = cov / var_x
    intercept = mean(y) - slope * mean(x)
    r_val = correlation(x, y)
    return RegressionResult(slope=slope, intercept=intercept, r_value=r_val, r_squared=r_val ** 2)


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Probability density function of normal distribution N(mu, sigma^2)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * (((x - mu) / sigma) ** 2)
    return coeff * math.exp(exponent)


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Cumulative distribution function of normal distribution."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Probability mass function of binomial distribution B(n, p)."""
    if n < 0 or k < 0 or k > n:
        return 0.0
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    comb = math.comb(n, k) if hasattr(math, "comb") else math.factorial(n) // (
        math.factorial(k) * math.factorial(n - k)
    )
    return comb * (p ** k) * ((1.0 - p) ** (n - k))


def poisson_pmf(k: int, lam: float) -> float:
    """Probability mass function of Poisson distribution with rate lambda."""
    if k < 0:
        return 0.0
    if lam <= 0:
        raise ValueError("lambda must be positive")
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)
