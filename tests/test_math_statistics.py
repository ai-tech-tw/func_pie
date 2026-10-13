import unittest
import math
from funcpie.math.statistics import (
    mean, geometric_mean, harmonic_mean, median, mode,
    variance, stdev, z_scores, percentile, quantiles,
    skewness, kurtosis, covariance, correlation, linear_regression,
    normal_pdf, normal_cdf, binomial_pmf, poisson_pmf
)


class TestMathStatistics(unittest.TestCase):
    def test_descriptive(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(mean(data), 5.0)
        self.assertEqual(median(data), 5.0)
        self.assertAlmostEqual(geometric_mean([2, 8]), 4.0)
        self.assertAlmostEqual(harmonic_mean([2, 6]), 3.0)
        self.assertEqual(mode([1, 2, 2, 3]), 2)
        self.assertAlmostEqual(variance([2, 4, 4, 4, 5, 5, 7, 9], sample=False), 4.0)
        self.assertAlmostEqual(stdev([2, 4, 4, 4, 5, 5, 7, 9], sample=False), 2.0)
        self.assertEqual(percentile([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 50), 50.0)
        self.assertEqual(len(quantiles(data, 4)), 3)

    def test_bivariate(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        self.assertAlmostEqual(correlation(x, y), 1.0)
        reg = linear_regression(x, y)
        self.assertAlmostEqual(reg.slope, 2.0)
        self.assertAlmostEqual(reg.intercept, 0.0)
        self.assertAlmostEqual(reg.r_squared, 1.0)

    def test_distributions(self):
        # standard normal pdf at 0 is 1 / sqrt(2*pi) ~ 0.398942
        self.assertAlmostEqual(normal_pdf(0.0), 1.0 / math.sqrt(2.0 * math.pi), places=5)
        self.assertAlmostEqual(normal_cdf(0.0), 0.5, places=5)
        # Binomial pmf 2 heads in 2 flips with p=0.5 -> 0.25
        self.assertAlmostEqual(binomial_pmf(2, 2, 0.5), 0.25)
        # Poisson pmf k=0, lam=1 -> e^-1 ~ 0.367879
        self.assertAlmostEqual(poisson_pmf(0, 1.0), math.exp(-1.0), places=5)


if __name__ == "__main__":
    unittest.main()
