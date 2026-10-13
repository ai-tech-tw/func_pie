import unittest
import math
from funcpie.math.geometry import (
    distance_2d, distance_3d, triangle_area_heron, triangle_area_points,
    circle_area, circle_circumference, polygon_area, regular_polygon_area,
    sphere_volume, sphere_surface_area, cylinder_volume, cylinder_surface_area,
    cone_volume, cone_surface_area, cuboid_volume, cuboid_surface_area,
    cartesian_to_polar, polar_to_cartesian,
    cartesian_to_spherical, spherical_to_cartesian,
    cartesian_to_cylindrical, cylindrical_to_cartesian
)


class TestMathGeometry(unittest.TestCase):
    def test_2d_shapes(self):
        self.assertAlmostEqual(distance_2d((0, 0), (3, 4)), 5.0)
        self.assertAlmostEqual(triangle_area_heron(3, 4, 5), 6.0)
        self.assertAlmostEqual(triangle_area_points((0, 0), (4, 0), (0, 3)), 6.0)
        self.assertAlmostEqual(circle_area(2), math.pi * 4)
        self.assertAlmostEqual(circle_circumference(2), 4 * math.pi)

        # Square of side 2 area = 4
        square = [(0, 0), (2, 0), (2, 2), (0, 2)]
        self.assertAlmostEqual(polygon_area(square), 4.0)
        self.assertAlmostEqual(regular_polygon_area(4, 2), 4.0)

    def test_3d_shapes(self):
        self.assertAlmostEqual(distance_3d((0, 0, 0), (1, 2, 2)), 3.0)
        self.assertAlmostEqual(sphere_volume(3), 36.0 * math.pi)
        self.assertAlmostEqual(sphere_surface_area(3), 36.0 * math.pi)
        self.assertAlmostEqual(cylinder_volume(2, 5), 20.0 * math.pi)
        self.assertAlmostEqual(cone_volume(3, 4), 12.0 * math.pi)
        self.assertAlmostEqual(cuboid_volume(2, 3, 4), 24.0)
        self.assertAlmostEqual(cuboid_surface_area(2, 3, 4), 52.0)

    def test_coordinate_transforms(self):
        r, theta = cartesian_to_polar(0, 5)
        self.assertAlmostEqual(r, 5.0)
        self.assertAlmostEqual(theta, math.pi / 2)
        x, y = polar_to_cartesian(r, theta)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 5.0)

        r3, th3, ph3 = cartesian_to_spherical(0, 0, 5)
        self.assertAlmostEqual(r3, 5.0)
        self.assertAlmostEqual(th3, 0.0)
        x3, y3, z3 = spherical_to_cartesian(r3, th3, ph3)
        self.assertAlmostEqual(z3, 5.0)


if __name__ == "__main__":
    unittest.main()
