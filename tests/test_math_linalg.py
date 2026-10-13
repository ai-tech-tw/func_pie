import unittest
import math
from funcpie.math.linalg import Vector2D, Vector3D, Matrix


class TestMathLinAlg(unittest.TestCase):
    def test_vector2d(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(1, 2)
        self.assertEqual(v1 + v2, Vector2D(4, 6))
        self.assertEqual(v1 - v2, Vector2D(2, 2))
        self.assertEqual(v1 * 2, Vector2D(6, 8))
        self.assertEqual(v1 / 2, Vector2D(1.5, 2.0))
        self.assertAlmostEqual(v1.magnitude, 5.0)
        self.assertEqual(v1.normalized(), Vector2D(0.6, 0.8))
        self.assertEqual(v1.dot(v2), 3 * 1 + 4 * 2)  # 11
        self.assertEqual(v1.cross(v2), 3 * 2 - 4 * 1)  # 2

    def test_vector3d(self):
        v1 = Vector3D(1, 0, 0)
        v2 = Vector3D(0, 1, 0)
        self.assertEqual(v1.cross(v2), Vector3D(0, 0, 1))
        self.assertEqual(v1.dot(v2), 0.0)
        self.assertAlmostEqual(v1.magnitude, 1.0)
        self.assertAlmostEqual(v1.angle_to(v2), math.pi / 2)

    def test_matrix_operations(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        self.assertEqual((m1 + m2).data, [[6.0, 8.0], [10.0, 12.0]])
        self.assertEqual((m1 * 2).data, [[2.0, 4.0], [6.0, 8.0]])
        self.assertEqual(m1.transpose().data, [[1.0, 3.0], [2.0, 4.0]])
        self.assertEqual(m1.trace(), 5.0)

        # Matmul: [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        m_prod = m1 @ m2
        self.assertEqual(m_prod.data, [[19.0, 22.0], [43.0, 50.0]])

        # Determinant: 1*4 - 2*3 = -2
        self.assertAlmostEqual(m1.determinant(), -2.0)

        # Inverse: A @ A^-1 = I
        inv_m1 = m1.inverse()
        ident = m1 @ inv_m1
        self.assertAlmostEqual(ident[0][0], 1.0)
        self.assertAlmostEqual(ident[0][1], 0.0)
        self.assertAlmostEqual(ident[1][0], 0.0)
        self.assertAlmostEqual(ident[1][1], 1.0)

        # Solve system:
        # x + 2y = 5
        # 3x + 4y = 11 -> solution x=1, y=2
        sol = m1.solve([5, 11])
        self.assertAlmostEqual(sol[0], 1.0)
        self.assertAlmostEqual(sol[1], 2.0)


if __name__ == "__main__":
    unittest.main()
