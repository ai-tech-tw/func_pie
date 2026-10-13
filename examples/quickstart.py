import os
import sys

# Ensure repository root is in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import funcpie as fp
from funcpie.core import pipe, pipeable
from funcpie.math import Calculation, Matrix, Polynomial, Vector2D
from funcpie.physics import constants
from funcpie.physics import mechanics, thermodynamics, electromagnetism, waves, modern, units


def main():
    print("==================================================")
    print(f"FuncPie v{fp.__version__} - Quickstart Demo")
    print("==================================================")

    # 1. Chained calculation & functional pipeline
    print("\n[1] Math: Calculation & Pipelines")
    calc = Calculation(10)
    calc.add(15).mul(2).sub(10).div(2)
    print(f"Chained Calculation: ((10 + 15) * 2 - 10) / 2 = {calc.value}")

    # Functional pipeline with | operator
    @pipeable
    def add_ten(x):
        return x + 10

    @pipeable
    def square(x):
        return x ** 2

    piped_result = 5 | add_ten | square
    print(f"Unix-style Pipeline: 5 | add_ten | square = {piped_result}")

    # 2. Linear Algebra & Polynomials
    print("\n[2] Math: Vectors, Matrices & Polynomials")
    v1 = Vector2D(3, 4)
    v2 = Vector2D(1, 2)
    print(f"Vector2D: {v1} magnitude = {v1.magnitude}, normalized = {v1.normalized()}")
    print(f"v1 dot v2 = {v1.dot(v2)}, angle between = {math.degrees(v1.angle_to(v2)):.2f} deg")

    m = Matrix([[2, 1], [1, 3]])
    print(f"Matrix:\n{m}")
    print(f"Determinant: {m.determinant()}")
    print(f"Inverse:\n{m.inverse()}")
    sol = m.solve([4, 7])
    print(f"Solve [2x + y = 4; x + 3y = 7] -> x = {sol[0]:.2f}, y = {sol[1]:.2f}")

    poly = Polynomial([1, -3, 2])  # x^2 - 3x + 2
    print(f"Polynomial: {poly}")
    print(f"Derivative: {poly.derivative()}")
    print(f"Roots: {poly.roots()}")

    # 3. Physics Mechanics
    print("\n[3] Physics: Mechanics")
    proj = mechanics.projectile_motion(v0=50.0, angle_deg=45.0)
    print(f"Projectile launch at 50 m/s @ 45 deg:")
    print(f"  Max Height:       {proj.max_height:.2f} m")
    print(f"  Flight Time:      {proj.flight_time:.2f} s")
    print(f"  Horizontal Range: {proj.horizontal_range:.2f} m")

    v_esc_earth = mechanics.escape_velocity(mass=5.972e24, radius=6.371e6)
    print(f"Earth Escape Velocity: {v_esc_earth / 1000.0:.2f} km/s")

    # 4. Modern Physics & Relativity
    print("\n[4] Physics: Modern Physics & Relativity")
    v_fast = 0.8 * constants.c
    gamma = modern.lorentz_factor(v_fast)
    print(f"Lorentz factor at 0.8c: {gamma:.4f}")
    dilated_t = modern.time_dilation(proper_time=1.0, velocity=v_fast)
    print(f"1.0 s proper time dilated to: {dilated_t:.4f} s")
    e_rest = modern.rest_energy(1.0)
    print(f"1 kg rest mass energy (E=mc^2): {e_rest:.2e} Joules")

    # 5. Unit Conversions
    print("\n[5] Unit Conversions")
    print(f"100 km/h in mph: {units.convert(100, 'km/h', 'mph'):.2f} mph")
    print(f"100 Celsius in Fahrenheit: {units.celsius_to_fahrenheit(100):.1f} °F")
    print(f"1 atm in psi: {units.convert(1, 'atm', 'psi'):.2f} psi")

    print("\n==================================================")
    print("Demo completed successfully!")
    print("==================================================")


if __name__ == "__main__":
    main()
