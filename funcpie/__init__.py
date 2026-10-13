"""FuncPie - Modern Mathematics and Physics library for Python."""

__version__ = "1.0.0"
__author__ = "SuperSonic"
__license__ = "MIT"

from funcpie import core, math, physics
from funcpie.core import Pipeable, compose, curry, pipe, pipeable, tap
from funcpie.math import Calculation, Matrix, Polynomial, Vector2D, Vector3D
from funcpie.physics import constants

__all__ = [
    "__version__",
    "math",
    "physics",
    "core",
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
