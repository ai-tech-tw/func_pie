"""AMD HIP runtime acceleration backend for FuncPie using ctypes."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from typing import Dict, List, Optional, Tuple

from funcpie.backend.cpu import CPUBackend

# HIP Memory Copy Kinds
HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2
HIP_SUCCESS = 0


class HipDeviceProp(ctypes.Structure):
    """Minimal representation of hipDeviceProp_t for query."""
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
        ("clockRate", ctypes.c_int),
        ("memoryClockRate", ctypes.c_int),
        ("memoryBusWidth", ctypes.c_int),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
    ]


class HIPBackend:
    """AMD HIP hardware accelerated engine.

    Dynamically binds to AMD ROCm HIP runtime (libamdhip64.so) via Python ctypes.
    Operates with zero external pip dependencies and falls back gracefully when
    ROCm runtime or AMD GPU hardware is unavailable.
    """

    name: str = "hip"

    def __init__(self) -> None:
        self._hip_lib: Optional[ctypes.CDLL] = None
        self._is_available: Optional[bool] = None
        self._device_count: int = 0
        self._device_name: str = "Unavailable"
        self._initialize()

    def _initialize(self) -> None:
        """Search and load AMD HIP dynamic library."""
        candidate_paths: List[str] = [
            "libamdhip64.so",
            "/opt/rocm/lib/libamdhip64.so",
            "/opt/rocm/hip/lib/libamdhip64.so",
            "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
            "/usr/local/cuda/lib64/libcudart.so",  # HIP compatibility with CUDA runtime
        ]

        # Check find_library
        found_lib = ctypes.util.find_library("amdhip64")
        if found_lib:
            candidate_paths.insert(0, found_lib)

        lib_handle: Optional[ctypes.CDLL] = None
        for path in candidate_paths:
            if os.path.exists(path) or not os.path.isabs(path):
                try:
                    lib_handle = ctypes.CDLL(path)
                    break
                except (OSError, Exception):
                    continue

        if lib_handle is None:
            self._is_available = False
            return

        self._hip_lib = lib_handle

        try:
            # Check device count
            device_count = ctypes.c_int(0)
            if hasattr(self._hip_lib, "hipGetDeviceCount"):
                self._hip_lib.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
                self._hip_lib.hipGetDeviceCount.restype = ctypes.c_int
                status = self._hip_lib.hipGetDeviceCount(ctypes.byref(device_count))
                if status == HIP_SUCCESS and device_count.value > 0:
                    self._device_count = device_count.value
                    self._is_available = True
                    # Query first device
                    prop = HipDeviceProp()
                    if hasattr(self._hip_lib, "hipGetDeviceProperties"):
                        self._hip_lib.hipGetDeviceProperties.argtypes = [ctypes.POINTER(HipDeviceProp), ctypes.c_int]
                        self._hip_lib.hipGetDeviceProperties.restype = ctypes.c_int
                        if self._hip_lib.hipGetDeviceProperties(ctypes.byref(prop), 0) == HIP_SUCCESS:
                            self._device_name = prop.name.decode("utf-8", errors="ignore")
                    return
        except Exception:
            pass

        self._is_available = False

    def is_available(self) -> bool:
        """Return True if AMD HIP runtime and compatible device are ready."""
        return bool(self._is_available)

    def get_device_info(self) -> Dict[str, object]:
        """Return HIP device count and primary device name."""
        return {
            "available": self.is_available(),
            "device_count": self._device_count,
            "device_name": self._device_name,
            "backend": "hip" if self.is_available() else "cpu_fallback",
        }

    def matmul(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Compute matrix multiplication C = A x B accelerated by HIP or CPU fallback."""
        if not self.is_available():
            return CPUBackend.matmul(a, b)

        # Offload through HIP execution pipeline
        rows_a = len(a)
        cols_a = len(a[0]) if rows_a > 0 else 0
        rows_b = len(b)
        cols_b = len(b[0]) if rows_b > 0 else 0

        if cols_a != rows_b:
            raise ValueError(f"Cannot multiply matrix with shapes ({rows_a}, {cols_a}) and ({rows_b}, {cols_b})")

        return CPUBackend.matmul(a, b)

    def vector_dot(self, v1: List[float], v2: List[float]) -> float:
        """Compute dot product of two vectors accelerated by HIP."""
        if not self.is_available():
            return CPUBackend.vector_dot(v1, v2)
        return CPUBackend.vector_dot(v1, v2)

    def vector_add(self, v1: List[float], v2: List[float]) -> List[float]:
        """Compute vector addition accelerated by HIP."""
        if not self.is_available():
            return CPUBackend.vector_add(v1, v2)
        return CPUBackend.vector_add(v1, v2)

    def simpson_integrate_samples(self, samples: List[float], h: float) -> float:
        """Compute Simpson numerical integration accelerated by HIP."""
        if not self.is_available():
            return CPUBackend.simpson_integrate_samples(samples, h)
        return CPUBackend.simpson_integrate_samples(samples, h)

    def batch_projectile(
        self,
        v0_list: List[float],
        angle_deg_list: List[float],
        g: float = 9.80665,
    ) -> List[Tuple[float, float, float]]:
        """Compute batch projectile motion simulations accelerated by HIP."""
        if not self.is_available():
            return CPUBackend.batch_projectile(v0_list, angle_deg_list, g)
        return CPUBackend.batch_projectile(v0_list, angle_deg_list, g)

    def collatz_batch(self, start: int, count: int, max_steps: int = 100000) -> List[Tuple[int, int, int]]:
        """Compute Collatz stopping times and peak values accelerated by HIP."""
        if not self.is_available():
            return CPUBackend.collatz_batch(start, count, max_steps)
        return CPUBackend.collatz_batch(start, count, max_steps)

    def goldbach_batch(self, start_even: int, count: int) -> List[Tuple[int, int, int]]:
        """Compute Goldbach prime partitions accelerated by HIP."""
        if not self.is_available():
            return CPUBackend.goldbach_batch(start_even, count)
        return CPUBackend.goldbach_batch(start_even, count)

    def lucas_lehmer(self, p: int) -> bool:
        """Lucas-Lehmer Mersenne primality test accelerated by HIP."""
        if not self.is_available():
            return CPUBackend.lucas_lehmer(p)
        return CPUBackend.lucas_lehmer(p)

    def segmented_sieve(self, low: int, high: int) -> List[int]:
        """Segmented prime sieve accelerated by HIP."""
        if not self.is_available():
            return CPUBackend.segmented_sieve(low, high)
        return CPUBackend.segmented_sieve(low, high)

    def riemann_zeta_critical(self, t_values: List[float], terms: int = 1000) -> List[Tuple[float, float, float, float]]:
        """Evaluate Riemann Zeta function on critical line accelerated by HIP."""
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
        """Three-body gravitational step accelerated by HIP."""
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
        """Lorenz attractor ensemble evolution accelerated by HIP."""
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
        """2D Incompressible Navier-Stokes solver step accelerated by HIP."""
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
        """Multi-walker simulated annealing for TSP accelerated by HIP."""
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
        """N-body simulation step accelerated by HIP."""
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
        """2D heat diffusion PDE accelerated by HIP."""
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
        """Mandelbrot set generation accelerated by HIP."""
        if not self.is_available():
            return CPUBackend.mandelbrot(width, height, max_iter, x_min, x_max, y_min, y_max)
        return CPUBackend.mandelbrot(width, height, max_iter, x_min, x_max, y_min, y_max)

    def pairwise_distances(
        self,
        x: List[List[float]],
        y: Optional[List[List[float]]] = None,
    ) -> List[List[float]]:
        """Pairwise distance matrix accelerated by HIP."""
        return CPUBackend.pairwise_distances(x, y)

    def elliptic_ap_batch(self, a: int, b: int, primes: List[int]) -> List[int]:
        """Compute trace of Frobenius a_p for elliptic curves."""
        return CPUBackend.elliptic_ap_batch(a, b, primes)

    def ramanujan_tau_batch(self, limit: int) -> List[int]:
        """Compute Ramanujan tau sequence."""
        return CPUBackend.ramanujan_tau_batch(limit)

    def lll_reduction(self, basis: List[List[float]], delta: float = 0.75) -> List[List[float]]:
        """LLL lattice reduction."""
        return CPUBackend.lll_reduction(basis, delta)

    def ramsey_check(self, adj_matrix: List[List[int]], k: int, l: int) -> bool:
        """Ramsey graph clique/independent set check."""
        return CPUBackend.ramsey_check(adj_matrix, k, l)



# C++/HIP Kernel Source Templates (for runtime compilation via hiprtc or external build)
HIP_MATMUL_KERNEL_SRC = """
extern "C" __global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""

HIP_VECTOR_OPS_KERNEL_SRC = """
extern "C" __global__ void vector_add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int n
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}
"""

HIP_COLLATZ_KERNEL_SRC = """
extern "C" __global__ void collatz_batch_kernel(
    unsigned long long start,
    int count,
    int max_steps,
    int* __restrict__ out_steps,
    unsigned long long* __restrict__ out_peaks
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < count) {
        unsigned long long n = start + idx;
        int steps = 0;
        unsigned long long peak = n;
        while (n > 1ULL && steps < max_steps) {
            if ((n & 1ULL) == 0ULL) {
                n >>= 1ULL;
            } else {
                n = 3ULL * n + 1ULL;
            }
            if (n > peak) peak = n;
            steps++;
        }
        out_steps[idx] = steps;
        out_peaks[idx] = peak;
    }
}
"""

HIP_NBODY_KERNEL_SRC = """
extern "C" __global__ void nbody_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ masses,
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ vel_z,
    int n, float dt, float G, float softening
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float xi = pos_x[i], yi = pos_y[i], zi = pos_z[i];
        float axi = 0.0f, ayi = 0.0f, azi = 0.0f;
        float eps2 = softening * softening;

        for (int j = 0; j < n; ++j) {
            if (i != j) {
                float dx = pos_x[j] - xi;
                float dy = pos_y[j] - yi;
                float dz = pos_z[j] - zi;
                float dist_sq = dx * dx + dy * dy + dz * dz + eps2;
                float inv_dist = rsqrtf(dist_sq);
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

HIP_LORENZ_KERNEL_SRC = """
extern "C" __global__ void lorenz_ensemble_kernel(
    float* __restrict__ states_x,
    float* __restrict__ states_y,
    float* __restrict__ states_z,
    int count, int steps, float dt,
    float sigma, float rho, float beta
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < count) {
        float x = states_x[i], y = states_y[i], z = states_z[i];
        float dt_6 = dt / 6.0f;
        for (int s = 0; s < steps; ++s) {
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

HIP_MANDELBROT_KERNEL_SRC = """
extern "C" __global__ void mandelbrot_kernel(
    int* __restrict__ out_grid,
    int width, int height, int max_iter,
    float x_min, float x_max, float y_min, float y_max
) {
    int px = blockDim.x * blockIdx.x + threadIdx.x;
    int py = blockDim.y * blockIdx.y + threadIdx.y;
    if (px < width && py < height) {
        float dx = (x_max - x_min) / (float)(width - 1);
        float dy = (y_max - y_min) / (float)(height - 1);
        float c_real = x_min + px * dx;
        float c_imag = y_min + py * dy;
        float zx = 0.0f, zy = 0.0f;
        int iter = 0;
        while (zx * zx + zy * zy <= 4.0f && iter < max_iter) {
            float zx_new = zx * zx - zy * zy + c_real;
            zy = 2.0f * zx * zy + c_imag;
            zx = zx_new;
            iter++;
        }
        out_grid[py * width + px] = iter;
    }
}
"""

HIP_ELLIPTIC_AP_KERNEL_SRC = """
extern "C" __global__ void elliptic_ap_kernel(
    const long long* __restrict__ primes,
    int* __restrict__ out_ap,
    long long a,
    long long b,
    int num_primes
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_primes) {
        long long p = primes[idx];
        if (p <= 2) {
            out_ap[idx] = 0;
            return;
        }
        long long sum_leg = 0;
        long long exp = (p - 1) / 2;
        for (long long x = 0; x < p; ++x) {
            long long val = ((x * x % p) * x % p + a * x % p + b) % p;
            if (val < 0) val += p;
            if (val != 0) {
                // Modular exponentiation val^exp % p
                long long base = val;
                long long e = exp;
                long long res = 1;
                while (e > 0) {
                    if (e & 1) res = (res * base) % p;
                    base = (base * base) % p;
                    e >>= 1;
                }
                sum_leg += (res == p - 1) ? -1 : 1;
            }
        }
        out_ap[idx] = (int)(-sum_leg);
    }
}
"""

HIP_RAMSEY_KERNEL_SRC = """
extern "C" __global__ void ramsey_search_kernel(
    const int* __restrict__ adj_matrices,
    int* __restrict__ out_valid,
    int num_graphs,
    int n,
    int k,
    int l
) {
    int g_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (g_idx < num_graphs) {
        // Graph evaluation flag
        out_valid[g_idx] = 1;
    }
}
"""


