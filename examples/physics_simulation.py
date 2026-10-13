"""FuncPie Physics Simulation: Projectile Motion and Simple Harmonic Motion."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from funcpie.physics import mechanics, constants


def simulate_projectile():
    print("--- 1. Projectile Motion Simulation (v0=30 m/s, angle=60 deg) ---")
    summary = mechanics.projectile_motion(v0=30.0, angle_deg=60.0)
    print(f"Total time of flight: {summary.flight_time:.2f} s")
    print(f"Maximum height:       {summary.max_height:.2f} m")
    print(f"Horizontal range:     {summary.horizontal_range:.2f} m")

    trajectory = mechanics.projectile_trajectory(v0=30.0, angle_deg=60.0, num_points=10)
    print("\nTrajectory sample points (x in m, y in m):")
    for idx, (x, y) in enumerate(trajectory):
        print(f"  Step {idx:2d}: x = {x:6.2f} m, y = {y:6.2f} m")


def simulate_pendulum():
    print("\n--- 2. Simple Pendulum & Spring-Mass System ---")
    length = 2.0  # meters
    period = mechanics.simple_pendulum_period(length=length)
    frequency = 1.0 / period
    print(f"Pendulum of length {length} m:")
    print(f"  Period T:    {period:.3f} s")
    print(f"  Frequency f: {frequency:.3f} Hz")

    mass = 0.5  # kg
    k_spring = 50.0  # N/m
    spring_t = mechanics.spring_mass_period(mass=mass, spring_k=k_spring)
    print(f"\nSpring-mass oscillator (m = {mass} kg, k = {k_spring} N/m):")
    print(f"  Period T:    {spring_t:.3f} s")


if __name__ == "__main__":
    simulate_projectile()
    simulate_pendulum()
