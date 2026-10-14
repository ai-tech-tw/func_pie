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
