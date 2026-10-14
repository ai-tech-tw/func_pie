"""Compute backend abstraction layer supporting CPU and AMD HIP acceleration."""

from __future__ import annotations

from typing import Dict, Union

from funcpie.backend.cpu import CPUBackend
from funcpie.backend.hip import HIPBackend

_hip_backend = HIPBackend()
_cpu_backend = CPUBackend()

# Default to HIP if available, otherwise CPU
_active_backend: Union[HIPBackend, CPUBackend] = (
    _hip_backend if _hip_backend.is_available() else _cpu_backend
)


def is_hip_available() -> bool:
    """Check whether AMD HIP hardware acceleration is available."""
    return _hip_backend.is_available()


def get_backend() -> Union[HIPBackend, CPUBackend]:
    """Get the currently active compute backend (HIPBackend or CPUBackend)."""
    return _active_backend


def set_backend(backend_name: str) -> None:
    """Set the active compute backend ('hip' or 'cpu').

    Raises:
        ValueError: If backend_name is not supported or requested HIP is unavailable.
    """
    global _active_backend
    name = backend_name.lower().strip()
    if name == "hip":
        if not _hip_backend.is_available():
            raise RuntimeError("AMD HIP acceleration is not available in the current environment.")
        _active_backend = _hip_backend
    elif name == "cpu":
        _active_backend = _cpu_backend
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Supported backends are 'hip' and 'cpu'.")


def get_device_properties() -> Dict[str, object]:
    """Return dictionary of properties for current compute devices."""
    return _hip_backend.get_device_info()


__all__ = [
    "CPUBackend",
    "HIPBackend",
    "get_backend",
    "set_backend",
    "is_hip_available",
    "get_device_properties",
]
