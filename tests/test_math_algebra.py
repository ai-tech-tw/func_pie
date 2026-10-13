import unittest
from funcpie.math.algebra import (
    gcd, lcm, is_prime, primes_sieve, prime_factors, fibonacci,
    factorial, permutations_count, combinations_count, binomial_coefficient,
    solve_linear, solve_quadratic, Polynomial
)


class TestMathAlgebra(unittest.TestCase):
    def test_gcd_lcm(self):
        self.assertEqual(gcd(48, 180, 24), 12)
        self.assertEqual(lcm(12, 18, 30), 180)

    def test_primes(self):
        self.assertFalse(is_prime(0))
        self.assertFalse(is_prime(1))
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(17))
        self.assertFalse(is_prime(18))
        self.assertEqual(primes_sieve(20), [2, 3, 5, 7, 11, 13, 17, 19])
        self.assertEqual(prime_factors(360), {2: 3, 3: 2, 5: 1})

    def test_fibonacci(self):
        self.assertEqual([fibonacci(i) for i in range(8)], [0, 1, 1, 2, 3, 5, 8, 13])

    def test_combinatorics(self):
        self.assertEqual(factorial(5), 120)
        self.assertEqual(permutations_count(5, 2), 20)
        self.assertEqual(combinations_count(5, 2), 10)
        self.assertEqual(binomial_coefficient(6, 3), 20)

    def test_equations(self):
        # 2x - 6 = 0 -> x = 3
        self.assertAlmostEqual(solve_linear(2, -6), 3.0)
        # x^2 - 5x + 6 = 0 -> x = 3, 2
        r1, r2 = solve_quadratic(1, -5, 6)
        self.assertAlmostEqual(r1, 3.0)
        self.assertAlmostEqual(r2, 2.0)
        # Complex roots: x^2 + 1 = 0 -> +1j, -1j
        c1, c2 = solve_quadratic(1, 0, 1)
        self.assertEqual(c1, 1j)
        self.assertEqual(c2, -1j)

    def test_polynomial(self):
        # p(x) = 2x^2 + 3x + 1
        p = Polynomial([2, 3, 1])
        self.assertEqual(p(2), 2 * 4 + 3 * 2 + 1)  # 15
        self.assertEqual(p.degree, 2)

        # Derivative: 4x + 3
        dp = p.derivative()
        self.assertEqual(dp.coeffs, [4.0, 3.0])
        self.assertEqual(dp(2), 11.0)

        # Integral: (2/3)x^3 + 1.5x^2 + x + 5
        ip = p.integrate(5)
        self.assertAlmostEqual(ip.coeffs[0], 2.0 / 3.0)
        self.assertAlmostEqual(ip.coeffs[3], 5.0)

        # Arithmetic
        p1 = Polynomial([1, 2])  # x + 2
        p2 = Polynomial([1, -1])  # x - 1
        p_prod = p1 * p2  # x^2 + x - 2
        self.assertEqual(p_prod.coeffs, [1.0, 1.0, -2.0])
        self.assertEqual(p_prod.roots(), [1.0, -2.0])


if __name__ == "__main__":
    unittest.main()
