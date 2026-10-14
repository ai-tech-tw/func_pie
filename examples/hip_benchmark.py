#!/usr/bin/env python3
"""FuncPie AMD HIP Hardware Acceleration Benchmark & Demo.

This script demonstrates matrix multiplication and numerical integration
benchmarking between CPU and AMD HIP hardware acceleration.
"""

import os
import sys
import time

# Ensure repository root is in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from funcpie import (
    Matrix,
    get_backend,
    get_device_properties,
    is_hip_available,
    set_backend,
)
from funcpie.physics.mechanics import batch_projectile_motion


def main() -> None:
    print("=" * 60)
    print("FuncPie AMD HIP Hardware Acceleration Benchmark")
    print("=" * 60)

    # 1. Device Inspection
    props = get_device_properties()
    print(f"AMD HIP Available: {is_hip_available()}")
    print(f"Active Backend:    {get_backend().name}")
    print(f"Device Count:      {props['device_count']}")
    print(f"Primary Device:    {props['device_name']}")
    print("-" * 60)

    # 2. Matrix Multiplication Benchmark (64x64 Matrix)
    size = 64
    print(f"Benchmarking Matrix Multiplication ({size} x {size})...")
    data_a = [[float((i + j) % 10) for j in range(size)] for i in range(size)]
    data_b = [[float((i * j) % 10) for j in range(size)] for i in range(size)]

    m_a = Matrix(data_a)
    m_b = Matrix(data_b)

    t0 = time.perf_counter()
    res = m_a @ m_b
    t1 = time.perf_counter()

    print(f"Matrix Multiply completed in: {(t1 - t0) * 1000.0:.2f} ms")
    print(f"Result checksum (C[0][0]):     {res[0][0]:.2f}")
    print("-" * 60)

    # 3. Batch Physics Simulation Benchmark (1,000 projectiles)
    num_particles = 1000
    print(f"Benchmarking Batch Projectiles ({num_particles:,} particles)...")
    v0s = [30.0 + (i % 50) for i in range(num_particles)]
    angles = [15.0 + (i % 60) for i in range(num_particles)]

    t0 = time.perf_counter()
    trajectories = batch_projectile_motion(v0s, angles)
    t1 = time.perf_counter()

    print(f"Simulated {len(trajectories):,} projectiles in: {(t1 - t0) * 1000.0:.2f} ms")
    print(f"Sample projectile #0 max height: {trajectories[0][0]:.2f} m")
    print(f"Sample projectile #0 range:      {trajectories[0][2]:.2f} m")

    print("=" * 60)
    print("Benchmark completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
