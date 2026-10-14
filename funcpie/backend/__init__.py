"""Compute backend abstraction layer supporting CPU, AMD HIP, and Apple Silicon Metal acceleration."""

from __future__ import annotations

from typing import Dict, Union

from funcpie.backend.cpu import CPUBackend
from funcpie.backend.hip import HIPBackend
from funcpie.backend.metal import MetalBackend

_hip_backend = HIPBackend()
_metal_backend = MetalBackend()
_cpu_backend = CPUBackend()

# Priority: HIP (AMD/NVIDIA on Linux/Win) -> Metal (Apple Silicon on macOS) -> CPU fallback
if _hip_backend.is_available():
    _active_backend: Union[HIPBackend, MetalBackend, CPUBackend] = _hip_backend
elif _metal_backend.is_available():
    _active_backend = _metal_backend
else:
    _active_backend = _cpu_backend


def is_hip_available() -> bool:
    """Check whether AMD HIP hardware acceleration is available."""
    return _hip_backend.is_available()


def is_metal_available() -> bool:
    """Check whether Apple Silicon Metal hardware acceleration is available."""
    return _metal_backend.is_available()


def get_backend() -> Union[HIPBackend, MetalBackend, CPUBackend]:
    """Get the currently active compute backend (HIPBackend, MetalBackend, or CPUBackend)."""
    return _active_backend


def set_backend(backend_name: str) -> None:
    """Set the active compute backend ('hip', 'metal', or 'cpu').

    Raises:
        ValueError: If backend_name is not supported.
        RuntimeError: If requested backend is unavailable in current environment.
    """
    global _active_backend
    name = backend_name.lower().strip()
    if name == "hip":
        if not _hip_backend.is_available():
            raise RuntimeError("AMD HIP acceleration is not available in the current environment.")
        _active_backend = _hip_backend
    elif name == "metal":
        if not _metal_backend.is_available():
            raise RuntimeError("Apple Metal acceleration is not available in the current environment.")
        _active_backend = _metal_backend
    elif name == "cpu":
        _active_backend = _cpu_backend
    else:
        raise ValueError(
            f"Unknown backend: {backend_name}. Supported backends are 'hip', 'metal', and 'cpu'."
        )


def get_device_properties() -> Dict[str, object]:
    """Return dictionary of properties for compute devices."""
    hip_info = _hip_backend.get_device_info()
    metal_info = _metal_backend.get_device_info()
    dev_name = "Unavailable"
    if is_hip_available():
        dev_name = str(hip_info.get("device_name", "Unavailable"))
    elif is_metal_available():
        dev_name = str(metal_info.get("device_name", "Unavailable"))

    return {
        "available": is_hip_available() or is_metal_available(),
        "backend": _active_backend.name,
        "device_count": hip_info.get("device_count", 0),
        "device_name": dev_name,
        "active_backend": _active_backend.name,
        "hip": hip_info,
        "metal": metal_info,
    }


__all__ = [
    "CPUBackend",
    "HIPBackend",
    "MetalBackend",
    "get_backend",
    "set_backend",
    "is_hip_available",
    "is_metal_available",
    "get_device_properties",
]
