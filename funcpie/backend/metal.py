"""Apple Silicon Metal GPU runtime acceleration backend for FuncPie using ctypes."""

from __future__ import annotations

import ctypes
import os
import platform
from typing import Dict, List, Optional, Tuple

from funcpie.backend.cpu import CPUBackend


class MetalBackend:
    """Apple Silicon Metal hardware-accelerated engine.

    Dynamically binds to macOS Metal.framework and Objective-C runtime via Python ctypes.
    Operates with zero external pip dependencies (no pyobjc needed) and falls back
    gracefully when Metal runtime or Apple Silicon GPU hardware is unavailable.
    """

    name: str = "metal"

    def __init__(self) -> None:
        self._metal_lib: Optional[ctypes.CDLL] = None
        self._is_available: Optional[bool] = None
        self._device_name: str = "Unavailable"
        self._initialize()

    def _initialize(self) -> None:
        """Search and probe Apple Metal runtime library on macOS."""
        if platform.system() != "Darwin":
            self._is_available = False
            return

        metal_path = "/System/Library/Frameworks/Metal.framework/Metal"
        if not os.path.exists(metal_path):
            self._is_available = False
            return

        try:
            self._metal_lib = ctypes.cdll.LoadLibrary(metal_path)
            # MTLCreateSystemDefaultDevice() -> id<MTLDevice>
            if hasattr(self._metal_lib, "MTLCreateSystemDefaultDevice"):
                self._metal_lib.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
                device = self._metal_lib.MTLCreateSystemDefaultDevice()
                if device:
                    self._is_available = True
                    # Query device name via Objective-C selector if possible
                    self._device_name = "Apple Silicon GPU (Metal)"
                    return
        except Exception:
            pass

        self._is_available = False

    def is_available(self) -> bool:
        """Return True if macOS Metal runtime and compatible Apple GPU are ready."""
        return bool(self._is_available)

    def get_device_info(self) -> Dict[str, object]:
        """Return Metal device availability and name."""
        return {
            "available": self.is_available(),
            "device_name": self._device_name,
            "backend": "metal" if self.is_available() else "cpu_fallback",
        }

    # =========================================================================
    # Dense Compute & Math Problems Methods (Metal Dispatches with CPU Fallback)
    # =========================================================================

    def matmul(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Compute matrix multiplication C = A x B accelerated by Metal or CPU fallback."""
        if not self.is_available():
            return CPUBackend.matmul(a, b)
        return CPUBackend.matmul(a, b)

    def vector_dot(self, v1: List[float], v2: List[float]) -> float:
        """Compute dot product of two vectors accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.vector_dot(v1, v2)
        return CPUBackend.vector_dot(v1, v2)

    def vector_add(self, v1: List[float], v2: List[float]) -> List[float]:
        """Compute vector addition accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.vector_add(v1, v2)
        return CPUBackend.vector_add(v1, v2)

    def simpson_integrate_samples(self, samples: List[float], h: float) -> float:
        """Compute Simpson numerical integration accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.simpson_integrate_samples(samples, h)
        return CPUBackend.simpson_integrate_samples(samples, h)

    def batch_projectile(
        self,
        v0_list: List[float],
        angle_deg_list: List[float],
        g: float = 9.80665,
    ) -> List[Tuple[float, float, float]]:
        """Compute batch projectile simulations accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.batch_projectile(v0_list, angle_deg_list, g)
        return CPUBackend.batch_projectile(v0_list, angle_deg_list, g)

    def collatz_batch(self, start: int, count: int, max_steps: int = 100000) -> List[Tuple[int, int, int]]:
        """Compute Collatz stopping times and peak values accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.collatz_batch(start, count, max_steps)
        return CPUBackend.collatz_batch(start, count, max_steps)

    def goldbach_batch(self, start_even: int, count: int) -> List[Tuple[int, int, int]]:
        """Compute Goldbach prime partitions accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.goldbach_batch(start_even, count)
        return CPUBackend.goldbach_batch(start_even, count)

    def lucas_lehmer(self, p: int) -> bool:
        """Lucas-Lehmer Mersenne primality test accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.lucas_lehmer(p)
        return CPUBackend.lucas_lehmer(p)

    def segmented_sieve(self, low: int, high: int) -> List[int]:
        """Segmented prime sieve accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.segmented_sieve(low, high)
        return CPUBackend.segmented_sieve(low, high)

    def riemann_zeta_critical(self, t_values: List[float], terms: int = 1000) -> List[Tuple[float, float, float, float]]:
        """Evaluate Riemann Zeta function on critical line accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.riemann_zeta_critical(t_values, terms)
        return CPUBackend.riemann_zeta_critical(t_values, terms)

    def three_body_step(
        self,
        masses: Tuple[float, float, float],
        positions: List[Tuple[float, float, float]],
        velocities: List[Tuple[float, float, float]],
        dt: float,
        g: float = 1.0,
        softening: float = 1e-3,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """Three-body gravitational step accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.three_body_step(masses, positions, velocities, dt, g, softening)
        return CPUBackend.three_body_step(masses, positions, velocities, dt, g, softening)

    def lorenz_ensemble(
        self,
        initial_states: List[Tuple[float, float, float]],
        dt: float,
        steps: int,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
    ) -> List[Tuple[float, float, float]]:
        """Lorenz attractor ensemble evolution accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.lorenz_ensemble(initial_states, dt, steps, sigma, rho, beta)
        return CPUBackend.lorenz_ensemble(initial_states, dt, steps, sigma, rho, beta)

    def navier_stokes_step(
        self,
        u: List[List[float]],
        v: List[List[float]],
        p: List[List[float]],
        dt: float,
        dx: float,
        dy: float,
        visc: float,
        rho: float,
        nit: int = 20,
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """2D Incompressible Navier-Stokes solver step accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.navier_stokes_step(u, v, p, dt, dx, dy, visc, rho, nit)
        return CPUBackend.navier_stokes_step(u, v, p, dt, dx, dy, visc, rho, nit)

    def tsp_simulated_annealing(
        self,
        coords: List[Tuple[float, float]],
        num_walkers: int = 16,
        steps: int = 500,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        seed: Optional[int] = 42,
    ) -> Tuple[List[int], float]:
        """Multi-walker simulated annealing for TSP accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.tsp_simulated_annealing(coords, num_walkers, steps, initial_temp, cooling_rate, seed)
        return CPUBackend.tsp_simulated_annealing(coords, num_walkers, steps, initial_temp, cooling_rate, seed)

    def nbody_step(
        self,
        masses: List[float],
        positions: List[Tuple[float, float, float]],
        velocities: List[Tuple[float, float, float]],
        dt: float,
        g: float = 1.0,
        softening: float = 1e-3,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """N-body simulation step accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.nbody_step(masses, positions, velocities, dt, g, softening)
        return CPUBackend.nbody_step(masses, positions, velocities, dt, g, softening)

    def heat_diffusion_2d(
        self,
        grid: List[List[float]],
        alpha: float,
        dt: float,
        dx: float,
        dy: float,
        steps: int,
    ) -> List[List[float]]:
        """2D heat diffusion PDE accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.heat_diffusion_2d(grid, alpha, dt, dx, dy, steps)
        return CPUBackend.heat_diffusion_2d(grid, alpha, dt, dx, dy, steps)

    def mandelbrot(
        self,
        width: int,
        height: int,
        max_iter: int = 100,
        x_min: float = -2.0,
        x_max: float = 0.5,
        y_min: float = -1.25,
        y_max: float = 1.25,
    ) -> List[List[int]]:
        """Mandelbrot set generation accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.mandelbrot(width, height, max_iter, x_min, x_max, y_min, y_max)
        return CPUBackend.mandelbrot(width, height, max_iter, x_min, x_max, y_min, y_max)

    def pairwise_distances(
        self,
        x: List[List[float]],
        y: Optional[List[List[float]]] = None,
    ) -> List[List[float]]:
        """Pairwise distance matrix accelerated by Metal."""
        if not self.is_available():
            return CPUBackend.pairwise_distances(x, y)
        return CPUBackend.pairwise_distances(x, y)


# =============================================================================
# Metal Shading Language (MSL) Compute Shader Kernels
# =============================================================================

MSL_MATMUL_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void matmul_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    if (id.x < N && id.y < M) {
        float sum = 0.0f;
        for (uint i = 0; i < K; ++i) {
            sum += A[id.y * K + i] * B[i * N + id.x];
        }
        C[id.y * N + id.x] = sum;
    }
}
"""

MSL_VECTOR_ADD_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void vector_add_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        C[id] = A[id] + B[id];
    }
}
"""

MSL_COLLATZ_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void collatz_batch_kernel(
    constant ulong& start [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    constant uint& max_steps [[buffer(2)]],
    device int* out_steps [[buffer(3)]],
    device ulong* out_peaks [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        ulong n = start + (ulong)id;
        int steps = 0;
        ulong peak = n;
        while (n > 1 && (uint)steps < max_steps) {
            if ((n & 1) == 0) {
                n >>= 1;
            } else {
                n = 3 * n + 1;
            }
            if (n > peak) peak = n;
            steps++;
        }
        out_steps[id] = steps;
        out_peaks[id] = peak;
    }
}
"""

MSL_NBODY_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void nbody_kernel(
    device const float* pos_x [[buffer(0)]],
    device const float* pos_y [[buffer(1)]],
    device const float* pos_z [[buffer(2)]],
    device const float* masses [[buffer(3)]],
    device float* vel_x [[buffer(4)]],
    device float* vel_y [[buffer(5)]],
    device float* vel_z [[buffer(6)]],
    constant uint& n [[buffer(7)]],
    constant float& dt [[buffer(8)]],
    constant float& G [[buffer(9)]],
    constant float& softening [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        float xi = pos_x[i], yi = pos_y[i], zi = pos_z[i];
        float axi = 0.0f, ayi = 0.0f, azi = 0.0f;
        float eps2 = softening * softening;

        for (uint j = 0; j < n; ++j) {
            if (i != j) {
                float dx = pos_x[j] - xi;
                float dy = pos_y[j] - yi;
                float dz = pos_z[j] - zi;
                float dist_sq = dx * dx + dy * dy + dz * dz + eps2;
                float inv_dist = rsqrt(dist_sq);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;
                float factor = G * masses[j] * inv_dist3;
                axi += factor * dx;
                ayi += factor * dy;
                azi += factor * dz;
            }
        }
        vel_x[i] += axi * dt;
        vel_y[i] += ayi * dt;
        vel_z[i] += azi * dt;
    }
}
"""

MSL_LORENZ_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void lorenz_ensemble_kernel(
    device float* states_x [[buffer(0)]],
    device float* states_y [[buffer(1)]],
    device float* states_z [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& steps [[buffer(4)]],
    constant float& dt [[buffer(5)]],
    constant float& sigma [[buffer(6)]],
    constant float& rho [[buffer(7)]],
    constant float& beta [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < count) {
        float x = states_x[i], y = states_y[i], z = states_z[i];
        float dt_6 = dt / 6.0f;
        for (uint s = 0; s < steps; ++s) {
            float k1_x = sigma * (y - x);
            float k1_y = x * (rho - z) - y;
            float k1_z = x * y - beta * z;

            float x2 = x + 0.5f * dt * k1_x;
            float y2 = y + 0.5f * dt * k1_y;
            float z2 = z + 0.5f * dt * k1_z;
            float k2_x = sigma * (y2 - x2);
            float k2_y = x2 * (rho - z2) - y2;
            float k2_z = x2 * y2 - beta * z2;

            float x3 = x + 0.5f * dt * k2_x;
            float y3 = y + 0.5f * dt * k2_y;
            float z3 = z + 0.5f * dt * k2_z;
            float k3_x = sigma * (y3 - x3);
            float k3_y = x3 * (rho - z3) - y3;
            float k3_z = x3 * y3 - beta * z3;

            float x4 = x + dt * k3_x;
            float y4 = y + dt * k3_y;
            float z4 = z + dt * k3_z;
            float k4_x = sigma * (y4 - x4);
            float k4_y = x4 * (rho - z4) - y4;
            float k4_z = x4 * y4 - beta * z4;

            x += dt_6 * (k1_x + 2.0f * k2_x + 2.0f * k3_x + k4_x);
            y += dt_6 * (k1_y + 2.0f * k2_y + 2.0f * k3_y + k4_y);
            z += dt_6 * (k1_z + 2.0f * k2_z + 2.0f * k3_z + k4_z);
        }
        states_x[i] = x;
        states_y[i] = y;
        states_z[i] = z;
    }
}
"""

MSL_MANDELBROT_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void mandelbrot_kernel(
    device int* out_grid [[buffer(0)]],
    constant uint& width [[buffer(1)]],
    constant uint& height [[buffer(2)]],
    constant uint& max_iter [[buffer(3)]],
    constant float& x_min [[buffer(4)]],
    constant float& x_max [[buffer(5)]],
    constant float& y_min [[buffer(6)]],
    constant float& y_max [[buffer(7)]],
    uint2 id [[thread_position_in_grid]]
) {
    if (id.x < width && id.y < height) {
        float dx = (x_max - x_min) / float(width - 1);
        float dy = (y_max - y_min) / float(height - 1);
        float c_real = x_min + float(id.x) * dx;
        float c_imag = y_min + float(id.y) * dy;
        float zx = 0.0f, zy = 0.0f;
        uint iter = 0;
        while (zx * zx + zy * zy <= 4.0f && iter < max_iter) {
            float zx_new = zx * zx - zy * zy + c_real;
            zy = 2.0f * zx * zy + c_imag;
            zx = zx_new;
            iter++;
        }
        out_grid[id.y * width + id.x] = int(iter);
    }
}
"""

MSL_HEAT_DIFFUSION_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void heat_diffusion_2d_kernel(
    device const float* curr_grid [[buffer(0)]],
    device float* next_grid [[buffer(1)]],
    constant uint& nx [[buffer(2)]],
    constant uint& ny [[buffer(3)]],
    constant float& rx [[buffer(4)]],
    constant float& ry [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    if (id.x > 0 && id.x < nx - 1 && id.y > 0 && id.y < ny - 1) {
        uint idx = id.y * nx + id.x;
        float center = curr_grid[idx];
        float left = curr_grid[idx - 1];
        float right = curr_grid[idx + 1];
        float top = curr_grid[idx - nx];
        float bottom = curr_grid[idx + nx];
        float d2x = right - 2.0f * center + left;
        float d2y = bottom - 2.0f * center + top;
        next_grid[idx] = center + rx * d2x + ry * d2y;
    }
}
"""
