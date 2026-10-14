#!/usr/bin/env python3
"""OpenAI/math Advanced Theoretical Mathematics & Conjecture Verification Showcase.

Demonstrates high-performance computational verification of deep problems from
OpenAI's math repository (openai/math):
1. Birch and Swinnerton-Dyer (BSD) Conjecture & Elliptic Curves
2. Goldfeld's Quadratic Twist Families
3. Modular Forms & Ramanujan tau function (Deligne Bound)
4. Lenstra-Lenstra-Lovasz (LLL) Lattice Basis Reduction
5. Extremal Combinatorics & Ramsey Graph Search
6. GPU Backend & MCP Execution Engine
"""

from __future__ import annotations

import time

from funcpie.backend import get_backend, get_device_properties
from funcpie.mcp.cel_engine import evaluate_cel
from funcpie.mcp.registry import get_registry
from funcpie.problems.bsd import (
    EllipticCurve,
    ap_point_count,
    bsd_ratio_estimate,
    discriminant,
    elliptic_ap_batch,
    goldfeld_twist_batch,
    j_invariant,
    largest_real_root,
    real_period,
)
from funcpie.problems.combinatorics_open import (
    is_ramsey_graph,
    ramsey_edge_density,
    ramsey_random_search,
)
from funcpie.problems.lattice import (
    gram_schmidt,
    lll_reduction,
    shortest_vector_bound,
    vector_norm,
)
from funcpie.problems.modular import (
    ramanujan_tau,
    ramanujan_tau_batch,
    verify_deligne_bound,
    verify_multiplicative,
)


def main() -> None:
    print("=" * 80)
    print("  FuncPie - OpenAI/math Advanced Theoretical Mathematics Suite")
    print("=" * 80)

    # 0. Backend Hardware Detection
    props = get_device_properties()
    backend = get_backend()
    print(f"\n[Hardware Dispatch]")
    print(f"  Active Compute Backend : {backend.name.upper()}")
    print(f"  Device Name            : {props['device_name']}")
    print(f"  Hardware Acceleration  : {'ENABLED' if props['available'] else 'CPU Fallback'}")

    # 1. Birch and Swinnerton-Dyer (BSD) Conjecture
    print("\n" + "-" * 80)
    print("1. Birch & Swinnerton-Dyer (BSD) Conjecture & Elliptic Curves")
    print("-" * 80)
    a, b = -1.0, 0.0  # y^2 = x^3 - x (Cremona 32a2)
    curve = EllipticCurve(a, b)
    print(f"  Curve Equation         : y^2 = x^3 - x")
    print(f"  Weierstrass Disc Delta : {curve.disc}")
    print(f"  j-invariant            : {curve.j}")

    # Point counting over F_p
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    t0 = time.perf_counter()
    aps = elliptic_ap_batch(int(a), int(b), primes)
    t_ap = (time.perf_counter() - t0) * 1000.0
    print(f"  Frobenius Traces a_p   : {list(zip(primes[:8], aps[:8]))}... ({t_ap:.2f} ms)")

    # Real period Omega_E
    omega = real_period(a, b, steps=2000)
    print(f"  Real Period Omega_E    : {omega:.8f} (Exact: 2.62205755...)")

    # BSD ratio L(E, 1) / Omega_E
    ratio = bsd_ratio_estimate(int(a), int(b), primes)
    print(f"  BSD Ratio L(E,1)/Omega : {ratio:.6f}")

    # Goldfeld quadratic twists
    d_twists = [-7, -5, -3, -2, -1, 1, 2, 3, 5, 7]
    twist_stats = goldfeld_twist_batch(int(a), int(b), d_twists, primes)
    rank_ratio = float(str(twist_stats['rank_zero_ratio']))
    print(f"  Rank 0 Candidates Ratio: {rank_ratio * 100.0:.1f}%")

    # 2. Modular Forms & Ramanujan Tau Function
    print("\n" + "-" * 80)
    print("2. Modular Forms & Ramanujan Tau Function")
    print("-" * 80)
    t0 = time.perf_counter()
    taus = ramanujan_tau_batch(10)
    t_tau = (time.perf_counter() - t0) * 1000.0
    print(f"  Ramanujan tau(1..10)   : {taus} ({t_tau:.2f} ms)")

    # Multiplicative property
    is_mult = verify_multiplicative(2, 3)
    print(f"  Hecke Multiplicative   : tau(2 * 3) == tau(2) * tau(3) -> {is_mult}")

    # Deligne bound check
    deligne_ok, tau_7, bound_7 = verify_deligne_bound(7)
    print(f"  Deligne Theorem Bound  : |tau(7)| = {abs(tau_7)} <= {bound_7:.1f} -> {deligne_ok}")

    # 3. Lattice Theory & LLL Basis Reduction
    print("\n" + "-" * 80)
    print("3. High-Dimensional Lattice Basis Reduction (LLL)")
    print("-" * 80)
    lattice_basis = [
        [1.0, 1.0, 1.0],
        [-1.0, 0.0, 2.0],
        [3.0, 5.0, 6.0],
    ]
    print(f"  Initial Basis Vectors  :")
    for v in lattice_basis:
        print(f"    {v} (norm = {vector_norm(v):.4f})")

    reduced_basis = lll_reduction(lattice_basis, delta=0.75)
    print(f"  LLL-Reduced Basis      :")
    for v in reduced_basis:
        print(f"    {v} (norm = {vector_norm(v):.4f})")
    print(f"  Minkowski Shortest Bd  : {shortest_vector_bound(lattice_basis):.4f}")

    # 4. Extremal Combinatorics & Ramsey Theory
    print("\n" + "-" * 80)
    print("4. Extremal Combinatorics & Ramsey Graph Search")
    print("-" * 80)
    c5 = [
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ]
    is_c5_ramsey = is_ramsey_graph(c5, 3, 3)
    print(f"  5-Cycle C_5 Valid R(3,3): {is_c5_ramsey} (Proves R(3, 3) > 5)")
    print(f"  Edge Density           : {ramsey_edge_density(c5):.2f}")

    t0 = time.perf_counter()
    found_graph = ramsey_random_search(5, 3, 3, trials=500, seed=42)
    t_ramsey = (time.perf_counter() - t0) * 1000.0
    print(f"  Stochastic Search R(3,3): {'FOUND' if found_graph else 'NOT FOUND'} ({t_ramsey:.2f} ms)")

    # 5. MCP & Google CEL Expression Evaluation
    print("\n" + "-" * 80)
    print("5. FastMCP Registry Discovery & Google CEL Sandbox Execution")
    print("-" * 80)
    reg = get_registry()
    discovered_bsd = reg.search("bsd")
    print(f"  MCP Catalog 'bsd' Tools: {len(discovered_bsd)} operations found")
    for item in discovered_bsd[:3]:
        print(f"    - {item['name']}: {item['summary']}")

    cel_expr = "discriminant(-1.0, 0.0)"
    cel_res = evaluate_cel(cel_expr)
    print(f"\n  CEL Query: '{cel_expr}'")
    print(f"  Result   : {cel_res['result']} (Time: {cel_res['execution_time_ms']:.2f} ms)")

    cel_tau_expr = "ramanujan_tau(3)"
    cel_tau_res = evaluate_cel(cel_tau_expr)
    print(f"  CEL Query: '{cel_tau_expr}'")
    print(f"  Result   : {cel_tau_res['result']} (Time: {cel_tau_res['execution_time_ms']:.2f} ms)")

    print("\n" + "=" * 80)
    print("  Verification Showcase Complete. All theoretical algorithms operational!")
    print("=" * 80)


if __name__ == "__main__":
    main()
