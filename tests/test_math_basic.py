import unittest
from funcpie.math.basic import Calculation, add, sub, mul, div, power, clamp, sign, is_close, lerp, remap


class TestMathBasic(unittest.TestCase):
    def setUp(self):
        self.calc = Calculation()

    def test_original_calculation_interface(self):
        self.assertEqual(self.calc.add(10, 5), 15)
        self.assertEqual(self.calc.sub(10, 5), 5)
        self.assertEqual(self.calc.mul(10, 5), 50)
        self.assertEqual(self.calc.div(10, 5), 2.0)

        # Usage dict test
        exec_dict, exec_codes = self.calc.usage()
        self.assertIn("+", exec_dict)
        self.assertIn("-", exec_dict)
        self.assertIn("*", exec_dict)
        self.assertIn("/", exec_dict)
        self.assertEqual(exec_dict["+"](3, 4), 7)
        self.assertEqual(exec_dict["/"](8, 2), 4.0)

    def test_zero_division(self):
        with self.assertRaises(ZeroDivisionError):
            self.calc.div(10, 0)
        with self.assertRaises(ZeroDivisionError):
            div(10, 0)

    def test_extended_calculation_methods(self):
        self.assertEqual(self.calc.pow(2, 3), 8)
        self.assertEqual(self.calc.mod(10, 3), 1)
        self.assertEqual(self.calc.floordiv(10, 3), 3)

    def test_chained_calculation(self):
        c = Calculation(10)
        c.add(5).mul(2).sub(6).div(4)
        # ((10 + 5) * 2 - 6) / 4 = (30 - 6) / 4 = 24 / 4 = 6.0
        self.assertEqual(c.value, 6.0)

    def test_standalone_functions(self):
        self.assertEqual(add(10, 20), 30)
        self.assertEqual(sub(20, 5), 15)
        self.assertEqual(mul(4, 5), 20)
        self.assertEqual(div(20, 4), 5.0)
        self.assertEqual(power(2, 4), 16)
        self.assertEqual(clamp(15, 0, 10), 10)
        self.assertEqual(clamp(-5, 0, 10), 0)
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(sign(42), 1)
        self.assertEqual(sign(-42), -1)
        self.assertEqual(sign(0), 0)
        self.assertTrue(is_close(0.1 + 0.2, 0.3))
        self.assertAlmostEqual(lerp(10, 20, 0.5), 15.0)
        self.assertAlmostEqual(remap(5, 0, 10, 0, 100), 50.0)


if __name__ == "__main__":
    unittest.main()
