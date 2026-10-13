import unittest
import math
from funcpie.math.special import (
    gamma, beta, erf, erfc, sigmoid, relu, leaky_relu, softmax, log_sum_exp
)


class TestMathSpecial(unittest.TestCase):
    def test_gamma_beta(self):
        self.assertAlmostEqual(gamma(5), 24.0)  # (5-1)! = 24
        self.assertAlmostEqual(beta(1, 1), 1.0)
        self.assertAlmostEqual(beta(2, 3), gamma(2) * gamma(3) / gamma(5))

    def test_erf(self):
        self.assertAlmostEqual(erf(0.0), 0.0)
        self.assertAlmostEqual(erfc(0.0), 1.0)

    def test_activations(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertEqual(relu(-5), 0.0)
        self.assertEqual(relu(5), 5.0)
        self.assertEqual(leaky_relu(-10, alpha=0.1), -1.0)
        self.assertEqual(leaky_relu(10, alpha=0.1), 10.0)

    def test_softmax_and_logsumexp(self):
        sm = softmax([1.0, 1.0])
        self.assertAlmostEqual(sm[0], 0.5)
        self.assertAlmostEqual(sm[1], 0.5)
        self.assertAlmostEqual(sum(sm), 1.0)

        lse = log_sum_exp([0.0, 0.0])
        self.assertAlmostEqual(lse, math.log(2.0))


if __name__ == "__main__":
    unittest.main()
