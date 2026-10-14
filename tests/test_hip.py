import unittest
from funcpie.backend import (
    CPUBackend,
    HIPBackend,
    get_backend,
    get_device_properties,
    is_hip_available,
    set_backend,
)
from funcpie.math.linalg import Matrix
from funcpie.math.calculus import integrate
from funcpie.physics.mechanics import batch_projectile_motion


class TestHIPBackend(unittest.TestCase):
    def setUp(self) -> None:
        set_backend("cpu")

    def test_backend_detection(self) -> None:
        props = get_device_properties()
        self.assertIn("available", props)
        self.assertIn("backend", props)
        self.assertIn("device_count", props)
        self.assertIsInstance(is_hip_available(), bool)

    def test_cpu_backend_fallback_operations(self) -> None:
        # Matrix multiplication
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        res = CPUBackend.matmul(a, b)
        self.assertEqual(res, [[19.0, 22.0], [43.0, 50.0]])

        # Vector operations
        dot = CPUBackend.vector_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        self.assertEqual(dot, 32.0)

        add = CPUBackend.vector_add([1.0, 2.0], [3.0, 4.0])
        self.assertEqual(add, [4.0, 6.0])

        # Simpson integration
        # f(x) = x^2 from 0 to 2 with h=0.5 -> samples at 0, 0.5, 1.0, 1.5, 2.0
        samples = [0.0, 0.25, 1.0, 2.25, 4.0]
        area = CPUBackend.simpson_integrate_samples(samples, 0.5)
        self.assertAlmostEqual(area, 8.0 / 3.0, places=4)

        # Batch projectile
        results = CPUBackend.batch_projectile([50.0, 30.0], [45.0, 60.0])
        self.assertEqual(len(results), 2)
        h_max, t_flight, h_range = results[0]
        self.assertGreater(h_max, 0.0)
        self.assertGreater(t_flight, 0.0)
        self.assertGreater(h_range, 0.0)

    def test_hip_backend_interface(self) -> None:
        hip = HIPBackend()
        # Even if AMD GPU is unavailable, HIPBackend gracefully fallbacks to CPU
        a = [[2.0, 0.0], [0.0, 2.0]]
        b = [[3.0, 1.0], [1.0, 3.0]]
        res = hip.matmul(a, b)
        self.assertEqual(res, [[6.0, 2.0], [2.0, 6.0]])

        dot = hip.vector_dot([1.0, 2.0], [3.0, 4.0])
        self.assertEqual(dot, 11.0)

        vec_add = hip.vector_add([1.0, 2.0], [3.0, 4.0])
        self.assertEqual(vec_add, [4.0, 6.0])

        batch_proj = hip.batch_projectile([50.0], [45.0])
        self.assertEqual(len(batch_proj), 1)

    def test_matrix_and_calculus_integration(self) -> None:
        # Matrix using backend
        m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = Matrix([[2.0, 0.0], [1.0, 2.0]])
        m3 = m1 @ m2
        self.assertEqual(m3.data, [[4.0, 4.0], [10.0, 8.0]])

        # Calculus integrate using backend
        val = integrate(lambda x: float(x) ** 2, 0.0, 3.0, n=100)
        self.assertAlmostEqual(val, 9.0, places=4)

        # Batch projectiles
        batch = batch_projectile_motion([50.0, 100.0], [45.0, 45.0])
        self.assertEqual(len(batch), 2)
        self.assertAlmostEqual(batch[0][0], (50.0 * 0.70710678) ** 2 / (2.0 * 9.80665), places=2)

    def test_backend_switch(self) -> None:
        set_backend("cpu")
        self.assertEqual(get_backend().name, "cpu")

        with self.assertRaises(ValueError):
            set_backend("invalid_backend")

        if not is_hip_available():
            with self.assertRaises(RuntimeError):
                set_backend("hip")


if __name__ == "__main__":
    unittest.main()
