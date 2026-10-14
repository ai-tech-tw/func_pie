"""FuncPie - Modern Mathematics and Physics library for Python."""

__version__ = "1.0.0"
__author__ = "SuperSonic"
__license__ = "MIT"

from funcpie import backend, core, math, physics, problems
from funcpie.backend import (
    get_backend,
    get_device_properties,
    is_hip_available,
    set_backend,
)
from funcpie.core import Pipeable, compose, curry, pipe, pipeable, tap
from funcpie.math import Calculation, Matrix, Polynomial, Vector2D, Vector3D
from funcpie.physics import constants

__all__ = [
    "__version__",
    "math",
    "physics",
    "core",
    "backend",
    "problems",
    "is_hip_available",
    "get_backend",
    "set_backend",
    "get_device_properties",
    # Functional tools
    "pipe",
    "compose",
    "curry",
    "Pipeable",
    "pipeable",
    "tap",
    # Prominent classes
    "Calculation",
    "Vector2D",
    "Vector3D",
    "Matrix",
    "Polynomial",
    "constants",
]
