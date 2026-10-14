"""Mathematical Problems and Famous Conjectures Suite for FuncPie."""

from funcpie.problems.bsd import (
    EllipticCurve,
    ap_point_count,
    bsd_ratio_estimate,
    discriminant,
    elliptic_ap_batch,
    goldfeld_twist_batch,
    j_invariant,
    l_function_estimate,
    largest_real_root,
    real_period,
)
from funcpie.problems.collatz import (
    collatz_batch_verify,
    collatz_longest_chain,
    collatz_sequence,
)
from funcpie.problems.combinatorics_open import (
    is_ramsey_graph,
    ramsey_edge_density,
    ramsey_random_search,
)
from funcpie.problems.goldbach import (
    goldbach_batch_verify,
    goldbach_count_partitions,
    goldbach_partition,
)
from funcpie.problems.lattice import (
    dot_product,
    gram_schmidt,
    lattice_determinant,
    lll_reduction,
    shortest_vector_bound,
    vector_norm,
)
from funcpie.problems.lorenz import (
    lorenz_ensemble,
    lorenz_step,
    lyapunov_exponent_estimate,
)
from funcpie.problems.modular import (
    gcd,
    ramanujan_tau,
    ramanujan_tau_batch,
    verify_deligne_bound,
    verify_multiplicative,
)
from funcpie.problems.navier_stokes import (
    navier_stokes_2d_step,
    simulate_cavity_flow,
)
from funcpie.problems.primes import (
    lucas_lehmer_mersenne,
    mersenne_number,
    segmented_prime_sieve,
    twin_primes_scan,
)
from funcpie.problems.riemann import (
    find_zeta_zeros_approx,
    riemann_zeta_critical_line,
)
from funcpie.problems.three_body import (
    figure_eight_initial_conditions,
    three_body_simulation,
    three_body_step,
)
from funcpie.problems.tsp import (
    tsp_simulated_annealing,
    tsp_tour_distance,
)


__all__ = [
    # Collatz Conjecture
    "collatz_sequence",
    "collatz_batch_verify",
    "collatz_longest_chain",
    # Goldbach's Conjecture
    "goldbach_partition",
    "goldbach_batch_verify",
    "goldbach_count_partitions",
    # Mersenne & Twin Primes
    "mersenne_number",
    "lucas_lehmer_mersenne",
    "segmented_prime_sieve",
    "twin_primes_scan",
    # Riemann Hypothesis
    "riemann_zeta_critical_line",
    "find_zeta_zeros_approx",
    # Three-Body Problem
    "three_body_step",
    "three_body_simulation",
    "figure_eight_initial_conditions",
    # Lorenz Chaos & Dynamical Systems
    "lorenz_step",
    "lorenz_ensemble",
    "lyapunov_exponent_estimate",
    # Navier-Stokes Equations
    "navier_stokes_2d_step",
    "simulate_cavity_flow",
    # NP-Hard Optimization (TSP)
    "tsp_simulated_annealing",
    "tsp_tour_distance",
    # Birch and Swinnerton-Dyer (BSD) Conjecture & Elliptic Curves
    "EllipticCurve",
    "discriminant",
    "j_invariant",
    "ap_point_count",
    "elliptic_ap_batch",
    "largest_real_root",
    "real_period",
    "l_function_estimate",
    "bsd_ratio_estimate",
    "goldfeld_twist_batch",
    # Modular Forms & Ramanujan Tau
    "gcd",
    "ramanujan_tau",
    "ramanujan_tau_batch",
    "verify_multiplicative",
    "verify_deligne_bound",
    # Lattice Theory & LLL Basis Reduction
    "dot_product",
    "vector_norm",
    "gram_schmidt",
    "lll_reduction",
    "lattice_determinant",
    "shortest_vector_bound",
    # Extremal Combinatorics & Ramsey Theory
    "is_ramsey_graph",
    "ramsey_edge_density",
    "ramsey_random_search",
]
