"""Linear algebra: 2D/3D vectors and matrices."""

from __future__ import annotations

import math
from typing import Iterator, List, Sequence, Tuple, Union

Number = Union[int, float]


class Vector2D:
    """2-dimensional vector with geometric and algebraic operations."""

    __slots__ = ("x", "y")

    def __init__(self, x: Number, y: Number):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self) -> str:
        return f"Vector2D({self.x:g}, {self.y:g})"

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    def __getitem__(self, idx: int) -> float:
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.y
        raise IndexError("Vector2D index out of range (0 or 1)")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector2D):
            return False
        return math.isclose(self.x, other.x, abs_tol=1e-9) and math.isclose(self.y, other.y, abs_tol=1e-9)

    def __add__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x - other.x, self.y - other.y)

    def __neg__(self) -> Vector2D:
        return Vector2D(-self.x, -self.y)

    def __mul__(self, scalar: Number) -> Vector2D:
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: Number) -> Vector2D:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> Vector2D:
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector2D(self.x / scalar, self.y / scalar)

    @property
    def magnitude(self) -> float:
        """Euclidean norm / length of the vector."""
        return math.hypot(self.x, self.y)

    norm = magnitude

    @property
    def magnitude_squared(self) -> float:
        """Squared length of the vector."""
        return self.x * self.x + self.y * self.y

    def normalized(self) -> Vector2D:
        """Return unit vector in the same direction."""
        m = self.magnitude
        if m == 0:
            raise ZeroDivisionError("Cannot normalize a zero vector")
        return self / m

    def dot(self, other: Vector2D) -> float:
        """Dot product of two 2D vectors."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: Vector2D) -> float:
        """2D cross product (scalar perpendicular magnitude: x1*y2 - x2*y1)."""
        return self.x * other.y - self.y * other.x

    def angle(self) -> float:
        """Angle of vector relative to positive x-axis in radians."""
        return math.atan2(self.y, self.x)

    def angle_to(self, other: Vector2D) -> float:
        """Angle between two vectors in radians."""
        m1 = self.magnitude
        m2 = other.magnitude
        if m1 == 0 or m2 == 0:
            raise ValueError("Cannot calculate angle with zero vector")
        cos_theta = self.dot(other) / (m1 * m2)
        # Clamp to [-1, 1] to prevent float precision domain error
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.acos(cos_theta)

    def rotate(self, angle_rad: float) -> Vector2D:
        """Rotate vector counter-clockwise by angle in radians."""
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return Vector2D(self.x * c - self.y * s, self.x * s + self.y * c)

    def project_onto(self, other: Vector2D) -> Vector2D:
        """Project this vector onto another vector."""
        denom = other.magnitude_squared
        if denom == 0:
            raise ValueError("Cannot project onto zero vector")
        return other * (self.dot(other) / denom)


class Vector3D:
    """3-dimensional vector with vector algebra operations."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: Number, y: Number, z: Number):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Vector3D({self.x:g}, {self.y:g}, {self.z:g})"

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, idx: int) -> float:
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.y
        elif idx == 2:
            return self.z
        raise IndexError("Vector3D index out of range (0, 1, or 2)")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3D):
            return False
        return (
            math.isclose(self.x, other.x, abs_tol=1e-9)
            and math.isclose(self.y, other.y, abs_tol=1e-9)
            and math.isclose(self.z, other.z, abs_tol=1e-9)
        )

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> Vector3D:
        return Vector3D(-self.x, -self.y, -self.z)

    def __mul__(self, scalar: Number) -> Vector3D:
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: Number) -> Vector3D:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> Vector3D:
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    @property
    def magnitude(self) -> float:
        """Euclidean norm / length of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    norm = magnitude

    @property
    def magnitude_squared(self) -> float:
        """Squared length of the vector."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> Vector3D:
        """Return unit vector in the same direction."""
        m = self.magnitude
        if m == 0:
            raise ZeroDivisionError("Cannot normalize a zero vector")
        return self / m

    def dot(self, other: Vector3D) -> float:
        """Dot product of two 3D vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3D) -> Vector3D:
        """Cross product of two 3D vectors."""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def angle_to(self, other: Vector3D) -> float:
        """Angle between two vectors in radians."""
        m1 = self.magnitude
        m2 = other.magnitude
        if m1 == 0 or m2 == 0:
            raise ValueError("Cannot calculate angle with zero vector")
        cos_theta = self.dot(other) / (m1 * m2)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.acos(cos_theta)

    def project_onto(self, other: Vector3D) -> Vector3D:
        """Project this vector onto another vector."""
        denom = other.magnitude_squared
        if denom == 0:
            raise ValueError("Cannot project onto zero vector")
        return other * (self.dot(other) / denom)


class Matrix:
    """2D Matrix representation supporting linear algebra operations."""

    def __init__(self, data: Sequence[Sequence[Number]]):
        if not data or not data[0]:
            raise ValueError("Matrix cannot be empty")
        cols = len(data[0])
        self.data: List[List[float]] = []
        for row in data:
            if len(row) != cols:
                raise ValueError("All rows in Matrix must have identical lengths")
            self.data.append([float(val) for val in row])
        self.rows = len(self.data)
        self.cols = cols

    @classmethod
    def identity(cls, n: int) -> Matrix:
        """Create an n x n identity matrix."""
        if n <= 0:
            raise ValueError("Dimension must be positive")
        return cls([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])

    @classmethod
    def zeros(cls, rows: int, cols: int) -> Matrix:
        """Create a rows x cols zero matrix."""
        return cls([[0.0] * cols for _ in range(rows)])

    def __repr__(self) -> str:
        rows_str = ", ".join(str(r) for r in self.data)
        return f"Matrix([{rows_str}])"

    def __getitem__(self, idx: int) -> List[float]:
        return self.data[idx]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    def is_square(self) -> bool:
        return self.rows == self.cols

    def transpose(self) -> Matrix:
        """Return the transpose of the matrix."""
        return Matrix([[self.data[r][c] for r in range(self.rows)] for c in range(self.cols)])

    def __add__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch for matrix addition: {self.shape} vs {other.shape}")
        return Matrix([[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(self.data, other.data)])

    def __sub__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch for matrix subtraction: {self.shape} vs {other.shape}")
        return Matrix([[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(self.data, other.data)])

    def __mul__(self, scalar: Number) -> Matrix:
        return Matrix([[val * scalar for val in row] for row in self.data])

    def __rmul__(self, scalar: Number) -> Matrix:
        return self.__mul__(scalar)

    def matmul(self, other: Union[Matrix, Vector2D, Vector3D]) -> Union[Matrix, List[float]]:
        """Matrix multiplication."""
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError(f"Incompatible dimensions for multiplication: {self.shape} x {other.shape}")
            from funcpie.backend import get_backend
            res = get_backend().matmul(self.data, other.data)
            return Matrix(res)
        elif isinstance(other, (Vector2D, Vector3D)):
            vec_list = list(other)
            if self.cols != len(vec_list):
                raise ValueError(f"Matrix cols {self.cols} != vector length {len(vec_list)}")
            res_vec = [sum(self.data[i][k] * vec_list[k] for k in range(self.cols)) for i in range(self.rows)]
            return res_vec
        raise TypeError(f"Unsupported operand type for matmul: {type(other)}")

    def __matmul__(self, other: Matrix) -> Matrix:
        res = self.matmul(other)
        if isinstance(res, Matrix):
            return res
        raise TypeError("Expected Matrix result")

    def trace(self) -> float:
        """Return trace (sum of diagonal elements) of square matrix."""
        if not self.is_square():
            raise ValueError("Trace is only defined for square matrices")
        return sum(self.data[i][i] for i in range(self.rows))

    def determinant(self) -> float:
        """Compute the determinant of a square matrix using Gaussian elimination."""
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices")
        n = self.rows
        # Make a copy of matrix elements
        m = [list(row) for row in self.data]
        det = 1.0

        for i in range(n):
            # Pivot selection
            max_row = i
            for k in range(i + 1, n):
                if abs(m[k][i]) > abs(m[max_row][i]):
                    max_row = k
            if max_row != i:
                m[i], m[max_row] = m[max_row], m[i]
                det = -det

            pivot = m[i][i]
            if abs(pivot) < 1e-12:
                return 0.0

            det *= pivot
            for k in range(i + 1, n):
                factor = m[k][i] / pivot
                for j in range(i + 1, n):
                    m[k][j] -= factor * m[i][j]

        return det

    def inverse(self) -> Matrix:
        """Compute the matrix inverse using Gauss-Jordan elimination."""
        if not self.is_square():
            raise ValueError("Only square matrices can be inverted")
        n = self.rows
        # Augmented matrix [A | I]
        aug = [list(self.data[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        for i in range(n):
            pivot_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[pivot_row][i]):
                    pivot_row = k
            if abs(aug[pivot_row][i]) < 1e-12:
                raise ValueError("Matrix is singular and cannot be inverted")
            aug[i], aug[pivot_row] = aug[pivot_row], aug[i]

            pivot = aug[i][i]
            for j in range(2 * n):
                aug[i][j] /= pivot

            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * n):
                        aug[k][j] -= factor * aug[i][j]

        inv_data = [[aug[i][n + j] for j in range(n)] for i in range(n)]
        return Matrix(inv_data)

    def solve(self, b: Sequence[Number]) -> List[float]:
        """Solve linear system Ax = b."""
        if not self.is_square():
            raise ValueError("System matrix must be square")
        if len(b) != self.rows:
            raise ValueError(f"Vector b length {len(b)} must match matrix rows {self.rows}")
        inv_a = self.inverse()
        return [sum(inv_a.data[i][j] * b[j] for j in range(self.rows)) for i in range(self.rows)]
