"""Mathematical Problems and Famous Conjectures Suite for FuncPie."""

from funcpie.problems.collatz import (
    collatz_batch_verify,
    collatz_longest_chain,
    collatz_sequence,
)
from funcpie.problems.goldbach import (
    goldbach_batch_verify,
    goldbach_count_partitions,
    goldbach_partition,
)
from funcpie.problems.lorenz import (
    lorenz_ensemble,
    lorenz_step,
    lyapunov_exponent_estimate,
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
]
