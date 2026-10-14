import unittest
from funcpie.core.pipeline import pipe, compose, curry, Pipeable, pipeable, tap


class TestPipeline(unittest.TestCase):
    def test_pipe(self):
        res = pipe(5, lambda x: int(x) + 1, lambda x: int(x) * 2, str)
        self.assertEqual(res, "12")

    def test_compose(self):
        double = lambda x: x * 2
        inc = lambda x: x + 1
        fn = compose(double, inc)
        self.assertEqual(fn(5), 12)

    def test_curry(self):
        @curry
        def add3(a: int, b: int, c: int) -> int:
            return a + b + c

        self.assertEqual(add3(1)(2)(3), 6)
        self.assertEqual(add3(1, 2)(3), 6)
        self.assertEqual(add3(1, 2, 3), 6)

    def test_pipeable_operator(self):
        inc = Pipeable(lambda x: x + 1)
        double = Pipeable(lambda x: x * 2)

        res = 5 | inc | double
        self.assertEqual(res, 12)

    def test_pipeable_decorator(self):
        @pipeable
        def square(x: int) -> int:
            return x ** 2

        self.assertEqual(4 | square, 16)

    def test_tap(self):
        logs = []
        res = pipe(
            10,
            tap(lambda x: logs.append(x)),
            lambda x: int(x) + 5
        )
        self.assertEqual(res, 15)
        self.assertEqual(logs, [10])


if __name__ == "__main__":
    unittest.main()
