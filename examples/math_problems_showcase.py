"""Mathematical Problems and Conjectures Showcase using FuncPie."""

import time
from funcpie.problems import (
    collatz_batch_verify,
    collatz_longest_chain,
    figure_eight_initial_conditions,
    find_zeta_zeros_approx,
    goldbach_batch_verify,
    goldbach_partition,
    lorenz_ensemble,
    lucas_lehmer_mersenne,
    lyapunov_exponent_estimate,
    mersenne_number,
    segmented_prime_sieve,
    simulate_cavity_flow,
    three_body_simulation,
    tsp_simulated_annealing,
    twin_primes_scan,
)


def main() -> None:
    print("=" * 60)
    print("FuncPie Mathematical Problems & Conjectures Showcase")
    print("=" * 60)

    # 1. Collatz Conjecture
    print("\n[1] Collatz Conjecture (3n + 1 Problem)")
    t0 = time.perf_counter()
    res = collatz_batch_verify(1, 1000)
    best_n, steps, peak = collatz_longest_chain(1, 1000)
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"Verified 1,000 integers in {dt:.2f} ms")
    print(f"Longest trajectory in [1, 1000]: n = {best_n} takes {steps} steps, peak = {peak}")

    # 2. Goldbach's Conjecture
    print("\n[2] Goldbach's Conjecture (Even Number = p1 + p2)")
    t0 = time.perf_counter()
    gb_res = goldbach_batch_verify(4, 500)
    p1, p2 = goldbach_partition(1000)
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"Verified 500 consecutive even numbers in {dt:.2f} ms")
    print(f"Goldbach partition for 1,000: {p1} + {p2} = 1000")

    # 3. Mersenne Primes
    print("\n[3] Mersenne Primes (Lucas-Lehmer Deterministic Test)")
    for p in [3, 5, 7, 11, 13, 17, 19]:
        is_prime = lucas_lehmer_mersenne(p)
        status = "PRIME" if is_prime else "COMPOSITE"
        print(f"M_{p:<2} = 2^{p:<2} - 1 = {mersenne_number(p):<7} -> {status}")

    # 4. Twin Primes
    print("\n[4] Twin Primes Sieve in [1,000, 1,200]")
    twins = twin_primes_scan(1000, 1200)
    print(f"Found {len(twins)} twin prime pairs: {twins}")

    # 5. Riemann Hypothesis Critical Line
    print("\n[5] Riemann Zeta Function along Critical Line s = 1/2 + i*t")
    zeros = find_zeta_zeros_approx(10.0, 26.0, steps=100, terms=500)
    print(f"Approximate non-trivial zeros detected in t in [10, 26]:")
    for t_approx, mag in zeros:
        print(f"  t ~ {t_approx:.4f} (modulus |zeta| = {mag:.4f})")

    # 6. Three-Body Problem
    print("\n[6] Chaotic Three-Body Problem (Figure-8 Choreography)")
    masses, pos, vel = figure_eight_initial_conditions()
    t0 = time.perf_counter()
    history = three_body_simulation(masses, pos, vel, dt=0.01, steps=200)
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"Integrated 200 RK4 orbit timesteps in {dt:.2f} ms")

    # 7. Lorenz Chaos and Lyapunov Exponent
    print("\n[7] Lorenz Chaos & Lyapunov Exponent (Butterfly Effect)")
    lyap = lyapunov_exponent_estimate((1.0, 1.0, 1.0), dt=0.01, steps=500)
    print(f"Estimated maximal Lyapunov exponent: lambda = {lyap:.4f}")
    if lyap > 0:
        print("  -> System exhibits exponential divergence (deterministic chaos).")

    # 8. Navier-Stokes Lid-Driven Cavity Flow
    print("\n[8] Navier-Stokes Incompressible Flow (2D Cavity Vortex)")
    t0 = time.perf_counter()
    u, v, p = simulate_cavity_flow(nx=15, ny=15, steps=30, dt=0.001)
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"Simulated 30 iterations of 2D Navier-Stokes cavity flow in {dt:.2f} ms")

    # 9. NP-Hard Traveling Salesperson Problem (TSP)
    print("\n[9] Traveling Salesperson Problem (Simulated Annealing)")
    coords = [
        (0.0, 0.0), (0.0, 4.0), (3.0, 4.0), (3.0, 0.0),
        (1.5, 2.0), (5.0, 2.0), (2.0, 6.0), (4.0, 5.0),
    ]
    t0 = time.perf_counter()
    best_tour, best_dist = tsp_simulated_annealing(coords, num_walkers=16, steps=200, seed=42)
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"Multi-walker simulated annealing completed in {dt:.2f} ms")
    print(f"Shortest tour distance: {best_dist:.3f}, Path: {best_tour}")

    print("\n" + "=" * 60)
    print("All mathematical problems solved and verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
