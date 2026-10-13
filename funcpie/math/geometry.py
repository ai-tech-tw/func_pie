"""Geometry: 2D/3D shapes, areas, volumes, and coordinate transformations."""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
Number = Union[int, float]


def distance_2d(p1: Sequence[Number], p2: Sequence[Number]) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def distance_3d(p1: Sequence[Number], p2: Sequence[Number]) -> float:
    """Euclidean distance between two 3D points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def triangle_area_heron(a: float, b: float, c: float) -> float:
    """Area of triangle given side lengths a, b, c using Heron's formula."""
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("Side lengths must be positive")
    if a + b <= c or a + c <= b or b + c <= a:
        raise ValueError("Triangle inequality violated")
    s = (a + b + c) / 2.0
    return math.sqrt(s * (s - a) * (s - b) * (s - c))


def triangle_area_points(p1: Sequence[Number], p2: Sequence[Number], p3: Sequence[Number]) -> float:
    """Area of triangle given 3 2D vertices."""
    return 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))


def circle_area(radius: float) -> float:
    """Area of a circle: pi * r^2."""
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return math.pi * radius * radius


def circle_circumference(radius: float) -> float:
    """Circumference of a circle: 2 * pi * r."""
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return 2.0 * math.pi * radius


def polygon_area(points: Sequence[Sequence[Number]]) -> float:
    """Area of an arbitrary 2D polygon using the Shoelace formula."""
    n = len(points)
    if n < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return 0.5 * abs(area)


def regular_polygon_area(n_sides: int, side_length: float) -> float:
    """Area of a regular n-sided polygon."""
    if n_sides < 3:
        raise ValueError("Polygon must have at least 3 sides")
    if side_length <= 0:
        raise ValueError("Side length must be positive")
    return (n_sides * side_length ** 2) / (4.0 * math.tan(math.pi / n_sides))


def sphere_volume(radius: float) -> float:
    """Volume of a sphere: (4/3) * pi * r^3."""
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return (4.0 / 3.0) * math.pi * (radius ** 3)


def sphere_surface_area(radius: float) -> float:
    """Surface area of a sphere: 4 * pi * r^2."""
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return 4.0 * math.pi * (radius ** 2)


def cylinder_volume(radius: float, height: float) -> float:
    """Volume of a cylinder: pi * r^2 * h."""
    if radius < 0 or height < 0:
        raise ValueError("Radius and height must be non-negative")
    return math.pi * (radius ** 2) * height


def cylinder_surface_area(radius: float, height: float) -> float:
    """Total surface area of a closed cylinder: 2*pi*r*h + 2*pi*r^2."""
    if radius < 0 or height < 0:
        raise ValueError("Radius and height must be non-negative")
    return 2.0 * math.pi * radius * (radius + height)


def cone_volume(radius: float, height: float) -> float:
    """Volume of a cone: (1/3) * pi * r^2 * h."""
    if radius < 0 or height < 0:
        raise ValueError("Radius and height must be non-negative")
    return (1.0 / 3.0) * math.pi * (radius ** 2) * height


def cone_surface_area(radius: float, height: float) -> float:
    """Total surface area of a cone: pi * r * (r + sqrt(h^2 + r^2))."""
    if radius < 0 or height < 0:
        raise ValueError("Radius and height must be non-negative")
    slant = math.hypot(radius, height)
    return math.pi * radius * (radius + slant)


def cuboid_volume(length: float, width: float, height: float) -> float:
    """Volume of a cuboid: l * w * h."""
    if length < 0 or width < 0 or height < 0:
        raise ValueError("Dimensions must be non-negative")
    return length * width * height


def cuboid_surface_area(length: float, width: float, height: float) -> float:
    """Total surface area of a cuboid: 2 * (lw + wh + hl)."""
    if length < 0 or width < 0 or height < 0:
        raise ValueError("Dimensions must be non-negative")
    return 2.0 * (length * width + width * height + height * length)


# Coordinate Transformations
def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
    """Convert Cartesian (x, y) to polar coordinates (r, theta) in radians."""
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    return (r, theta)


def polar_to_cartesian(r: float, theta: float) -> Tuple[float, float]:
    """Convert polar coordinates (r, theta) to Cartesian (x, y)."""
    return (r * math.cos(theta), r * math.sin(theta))


def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert Cartesian (x, y, z) to spherical (r, theta, phi) in radians.

    r >= 0, theta (polar/inclination angle from z-axis) in [0, pi],
    phi (azimuthal angle in xy-plane) in [-pi, pi].
    """
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0:
        return (0.0, 0.0, 0.0)
    theta = math.acos(max(-1.0, min(1.0, z / r)))
    phi = math.atan2(y, x)
    return (r, theta, phi)


def spherical_to_cartesian(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    """Convert spherical (r, theta, phi) to Cartesian (x, y, z)."""
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return (x, y, z)


def cartesian_to_cylindrical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert Cartesian (x, y, z) to cylindrical (r, theta, z)."""
    r, theta = cartesian_to_polar(x, y)
    return (r, theta, z)


def cylindrical_to_cartesian(r: float, theta: float, z: float) -> Tuple[float, float, float]:
    """Convert cylindrical (r, theta, z) to Cartesian (x, y, z)."""
    x, y = polar_to_cartesian(r, theta)
    return (x, y, z)
