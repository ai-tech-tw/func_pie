import unittest
import math
from funcpie.math.calculus import (
    derivative, second_derivative, gradient,
    trapezoid_rule, simpsons_rule, integrate,
    find_root_bisection, find_root_newton, find_root_secant
)


class TestMathCalculus(unittest.TestCase):
    def test_derivative(self):
        # f(x) = x^3, f'(2) = 3*4 = 12
        f = lambda x: x ** 3
        self.assertAlmostEqual(derivative(f, 2.0), 12.0, places=5)
        self.assertAlmostEqual(derivative(f, 2.0, method="forward"), 12.0, places=3)
        self.assertAlmostEqual(derivative(f, 2.0, method="backward"), 12.0, places=3)

    def test_second_derivative(self):
        # f(x) = x^3, f''(2) = 6*2 = 12
        f = lambda x: x ** 3
        self.assertAlmostEqual(second_derivative(f, 2.0), 12.0, places=4)

    def test_gradient(self):
        # f(x, y) = x^2 + 3*y^2 at (2, 1) -> [4, 6]
        f = lambda pt: pt[0] ** 2 + 3.0 * (pt[1] ** 2)
        grad = gradient(f, [2.0, 1.0])
        self.assertAlmostEqual(grad[0], 4.0, places=4)
        self.assertAlmostEqual(grad[1], 6.0, places=4)

    def test_integration(self):
        # integral of x^2 from 0 to 3 is 3^3 / 3 = 9
        f = lambda x: x ** 2
        trap = trapezoid_rule(f, 0, 3, n=1000)
        simp = simpsons_rule(f, 0, 3, n=1000)
        self.assertAlmostEqual(trap, 9.0, places=3)
        self.assertAlmostEqual(simp, 9.0, places=5)
        self.assertAlmostEqual(integrate(f, 0, 3), 9.0, places=5)

    def test_root_finding(self):
        # f(x) = x^2 - 4 = 0, root at 2 in [0, 5]
        f = lambda x: x ** 2 - 4.0
        r_bisect = find_root_bisection(f, 0, 5)
        r_newton = find_root_newton(f, 3.0)
        r_secant = find_root_secant(f, 1.0, 3.0)
        self.assertAlmostEqual(r_bisect, 2.0, places=6)
        self.assertAlmostEqual(r_newton, 2.0, places=6)
        self.assertAlmostEqual(r_secant, 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
