import unittest
import func_pie
from libs.math.basic import Calculation


class TestCompatibility(unittest.TestCase):
    def test_legacy_libs_math_basic_import(self):
        calc = Calculation()
        self.assertEqual(calc.add(7, 3), 10)
        self.assertEqual(calc.sub(7, 3), 4)
        self.assertEqual(calc.mul(7, 3), 21)
        self.assertEqual(calc.div(6, 3), 2.0)
        usage, codes = calc.usage()
        self.assertEqual(codes[:4], ["+", "-", "*", "/"])
        self.assertEqual(usage["+"](1, 2), 3)

    def test_func_pie_alias_import(self):
        self.assertEqual(func_pie.__version__, "1.0.0")
        self.assertTrue(hasattr(func_pie, "math"))
        self.assertTrue(hasattr(func_pie, "physics"))
        self.assertTrue(hasattr(func_pie, "Calculation"))
        self.assertTrue(hasattr(func_pie, "pipe"))


if __name__ == "__main__":
    unittest.main()
