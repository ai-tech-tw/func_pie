"""Hardware GPU stress and load burst test for AMD HIP Backend.

Exercises HIPBackend with heavy batch workloads (Collatz batch, elliptic Frobenius a_p,
large matrix multiplication) to trigger observable GPU compute load and power burst in rocm-smi.
"""

from __future__ import annotations

import time
from funcpie.backend.hip import HIPBackend


def main() -> None:
    print("=" * 60)
    print("FuncPie AMD HIP Hardware Stress & Load Burst Benchmark")
    print("=" * 60)

    backend = HIPBackend()
    info = backend.get_device_info()
    print(f"[*] Backend Available : {info['available']}")
    print(f"[*] Device Name       : {info['device_name']}")
    print(f"[*] Device Count      : {info['device_count']}")
    raw_mem = info.get("total_memory_bytes")
    total_bytes = int(raw_mem) if isinstance(raw_mem, (int, float)) else 0
    total_gb = total_bytes / (1024**3)
    print(f"[*] VRAM Size         : {total_gb:.2f} GB")
    print("-" * 60)

    # 1. Collatz Batch Stress (50,000,000 points)
    n_points = 50_000_000
    print(f"[*] [Stage 1/3] Triggering Collatz Batch Stress ({n_points:,} numbers)...")
    t0 = time.perf_counter()
    res_collatz = backend.collatz_batch(start=100_000_000, count=n_points, max_steps=1000)
    t1 = time.perf_counter()
    dt_collatz = t1 - t0
    rate = n_points / dt_collatz / 1e6
    print(f"    Completed in {dt_collatz:.3f} s ({rate:.2f} M ops/sec)")
    print(f"    Sample result [0]: {res_collatz[0]}")
    print(f"    Sample result [-1]: {res_collatz[-1]}")
    print("-" * 60)

    # 2. Elliptic Frobenius a_p Point Count Stress
    primes_batch = [p for p in range(5, 50000) if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]
    print(f"[*] [Stage 2/3] Triggering Elliptic Curve Frobenius a_p Stress ({len(primes_batch):,} primes)...")
    t0 = time.perf_counter()
    ap_results = backend.elliptic_ap_batch(a=0, b=-432, primes=primes_batch)
    t1 = time.perf_counter()
    dt_ap = t1 - t0
    print(f"    Completed in {dt_ap:.3f} s ({len(primes_batch)/dt_ap:.1f} curves/sec)")
    print(f"    First 5 a_p: {ap_results[:5]}")
    print("-" * 60)

    # 3. Large Matrix Multiplication Stress (1024 x 1024)
    size = 1024
    print(f"[*] [Stage 3/3] Triggering Matrix Multiplication ({size} x {size})...")
    mat_a = [[1.0 for _ in range(size)] for _ in range(size)]
    mat_b = [[2.0 for _ in range(size)] for _ in range(size)]
    t0 = time.perf_counter()
    mat_c = backend.matmul(mat_a, mat_b)
    t1 = time.perf_counter()
    dt_matmul = t1 - t0
    gflops = (2.0 * size * size * size) / (dt_matmul * 1e9)
    print(f"    Completed in {dt_matmul:.3f} s ({gflops:.2f} GFLOPS)")
    print(f"    Sample mat_c[0][0] = {mat_c[0][0]} (Expected {float(size * 2)})")
    print("=" * 60)
    print("Stress test completed successfully!")


if __name__ == "__main__":
    main()
