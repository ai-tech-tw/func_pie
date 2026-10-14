"""Tests for Apple Silicon Metal GPU acceleration backend and graceful fallback."""

from __future__ import annotations

import unittest

from funcpie.backend import (
    CPUBackend,
    MetalBackend,
    get_backend,
    get_device_properties,
    is_metal_available,
    set_backend,
)
from funcpie.backend.metal import (
    MSL_COLLATZ_KERNEL_SRC,
    MSL_HEAT_DIFFUSION_KERNEL_SRC,
    MSL_LORENZ_KERNEL_SRC,
    MSL_MANDELBROT_KERNEL_SRC,
    MSL_MATMUL_KERNEL_SRC,
    MSL_NBODY_KERNEL_SRC,
    MSL_VECTOR_ADD_KERNEL_SRC,
)


class TestMetalBackend(unittest.TestCase):
    """Test suite for MetalBackend interface, CPU fallback, and MSL kernels."""

    def test_backend_detection(self) -> None:
        metal = MetalBackend()
        self.assertEqual(metal.name, "metal")
        is_avail = metal.is_available()
        self.assertIsInstance(is_avail, bool)

        info = metal.get_device_info()
        self.assertIn("available", info)
        self.assertIn("device_name", info)
        self.assertIn("backend", info)
        if is_avail:
            self.assertEqual(info["backend"], "metal")
        else:
            self.assertEqual(info["backend"], "cpu_fallback")

    def test_metal_fallback_operations(self) -> None:
        metal = MetalBackend()
        # Matmul
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[2.0, 0.0], [1.0, 2.0]]
        c = metal.matmul(a, b)
        self.assertEqual(c, [[4.0, 4.0], [10.0, 8.0]])

        # Vector dot
        dot = metal.vector_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        self.assertEqual(dot, 32.0)

        # Collatz batch
        collatz = metal.collatz_batch(1, 3)
        self.assertEqual(collatz, [(1, 0, 1), (2, 1, 2), (3, 7, 16)])

        # Goldbach batch
        gb = metal.goldbach_batch(4, 1)
        self.assertEqual(gb, [(4, 2, 2)])

        # Lucas-Lehmer (M_5 = 31 is prime)
        self.assertTrue(metal.lucas_lehmer(5))

    def test_backend_switch_metal(self) -> None:
        avail = is_metal_available()
        if not avail:
            with self.assertRaises(RuntimeError):
                set_backend("metal")
        else:
            set_backend("metal")
            self.assertEqual(get_backend().name, "metal")

        # Always switchable back to cpu
        set_backend("cpu")
        self.assertEqual(get_backend().name, "cpu")

        # Device properties check
        props = get_device_properties()
        self.assertIn("active_backend", props)
        self.assertIn("hip", props)
        self.assertIn("metal", props)

    def test_msl_kernel_templates(self) -> None:
        kernels = [
            MSL_MATMUL_KERNEL_SRC,
            MSL_VECTOR_ADD_KERNEL_SRC,
            MSL_COLLATZ_KERNEL_SRC,
            MSL_NBODY_KERNEL_SRC,
            MSL_LORENZ_KERNEL_SRC,
            MSL_MANDELBROT_KERNEL_SRC,
            MSL_HEAT_DIFFUSION_KERNEL_SRC,
        ]
        for src in kernels:
            self.assertIn("#include <metal_stdlib>", src)
            self.assertIn("using namespace metal;", src)
            self.assertIn("kernel void", src)
            self.assertIn("[[thread_position_in_grid]]", src)


if __name__ == "__main__":
    unittest.main()
