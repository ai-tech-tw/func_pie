"""Tests for the Mathematical Problems and Famous Conjectures suite."""

from __future__ import annotations

import unittest

from funcpie.mcp.registry import get_registry
from funcpie.problems import (
    collatz_batch_verify,
    collatz_longest_chain,
    collatz_sequence,
    figure_eight_initial_conditions,
    find_zeta_zeros_approx,
    goldbach_batch_verify,
    goldbach_count_partitions,
    goldbach_partition,
    lorenz_ensemble,
    lorenz_step,
    lucas_lehmer_mersenne,
    lyapunov_exponent_estimate,
    mersenne_number,
    navier_stokes_2d_step,
    riemann_zeta_critical_line,
    segmented_prime_sieve,
    simulate_cavity_flow,
    three_body_simulation,
    three_body_step,
    tsp_simulated_annealing,
    tsp_tour_distance,
    twin_primes_scan,
)


class TestMathematicalProblems(unittest.TestCase):
    """Test suite for mathematical problems and conjecture solvers."""

    def test_collatz(self) -> None:
        seq13 = collatz_sequence(13)
        self.assertEqual(seq13, [13, 40, 20, 10, 5, 16, 8, 4, 2, 1])

        batch = collatz_batch_verify(1, 5)
        # (n, steps, peak)
        self.assertEqual(len(batch), 5)
        self.assertEqual(batch[0][0], 1)
        self.assertEqual(batch[0][1], 0)  # 1 takes 0 steps
        self.assertEqual(batch[1][0], 2)
        self.assertEqual(batch[1][1], 1)  # 2 takes 1 step (2 -> 1)
        self.assertEqual(batch[2][0], 3)
        self.assertEqual(batch[2][1], 7)  # 3 takes 7 steps

        best_n, steps, peak = collatz_longest_chain(1, 10)
        self.assertEqual(best_n, 9)  # 9 takes 19 steps
        self.assertEqual(steps, 19)
        self.assertEqual(peak, 52)

    def test_goldbach(self) -> None:
        p1, p2 = goldbach_partition(4)
        self.assertEqual((p1, p2), (2, 2))

        p1_10, p2_10 = goldbach_partition(10)
        self.assertEqual(p1_10 + p2_10, 10)
        self.assertIn((p1_10, p2_10), [(3, 7), (5, 5)])

        batch = goldbach_batch_verify(4, 5)
        self.assertEqual(len(batch), 5)
        for even_num, q1, q2 in batch:
            self.assertEqual(q1 + q2, even_num)

        # 10 has two partitions: 3+7 and 5+5
        self.assertEqual(goldbach_count_partitions(10), 2)

    def test_mersenne_and_twin_primes(self) -> None:
        self.assertEqual(mersenne_number(3), 7)
        self.assertEqual(mersenne_number(5), 31)

        # Known Mersenne primes
        self.assertTrue(lucas_lehmer_mersenne(3))   # 2^3 - 1 = 7
        self.assertTrue(lucas_lehmer_mersenne(5))   # 2^5 - 1 = 31
        self.assertTrue(lucas_lehmer_mersenne(7))   # 2^7 - 1 = 127
        self.assertTrue(lucas_lehmer_mersenne(13))  # 2^13 - 1 = 8191

        # Composite Mersenne number (2^11 - 1 = 2047 = 23 * 89)
        self.assertFalse(lucas_lehmer_mersenne(11))

        # Segmented sieve
        primes = segmented_prime_sieve(10, 30)
        self.assertEqual(primes, [11, 13, 17, 19, 23, 29])

        # Twin primes
        twins = twin_primes_scan(10, 30)
        self.assertEqual(twins, [(11, 13), (17, 19)])

    def test_riemann_zeta(self) -> None:
        # First non-trivial zero is around t = 14.134725
        samples = riemann_zeta_critical_line([14.134725], terms=800)
        self.assertEqual(len(samples), 1)
        t, real, imag, mag = samples[0]
        self.assertAlmostEqual(t, 14.134725, places=4)
        # Modulus at zero should be very close to 0
        self.assertLess(mag, 0.2)

        # Non-trivial zero approximation
        zeros = find_zeta_zeros_approx(12.0, 16.0, steps=50, terms=500)
        self.assertGreaterEqual(len(zeros), 1)
        approx_t, min_mag = zeros[0]
        self.assertAlmostEqual(approx_t, 14.13, delta=0.5)

    def test_three_body(self) -> None:
        masses, pos, vel = figure_eight_initial_conditions()
        self.assertEqual(len(pos), 3)
        self.assertEqual(len(vel), 3)

        new_pos, new_vel = three_body_step(masses, pos, vel, dt=0.001)
        self.assertEqual(len(new_pos), 3)
        self.assertEqual(len(new_vel), 3)

        sim = three_body_simulation(masses, pos, vel, dt=0.001, steps=5)
        self.assertEqual(len(sim), 6)  # Initial + 5 steps

    def test_lorenz(self) -> None:
        init_state = (1.0, 1.0, 1.0)
        next_state = lorenz_step(init_state, dt=0.01)
        self.assertEqual(len(next_state), 3)
        self.assertNotEqual(init_state, next_state)

        ensemble = lorenz_ensemble([init_state, (1.001, 1.0, 1.0)], dt=0.01, steps=20)
        self.assertEqual(len(ensemble), 2)

        lyap = lyapunov_exponent_estimate(init_state, dt=0.01, steps=100)
        self.assertIsInstance(lyap, float)

    def test_navier_stokes(self) -> None:
        nx, ny = 11, 11
        u = [[0.0 for _ in range(nx)] for _ in range(ny)]
        v = [[0.0 for _ in range(nx)] for _ in range(ny)]
        p = [[0.0 for _ in range(nx)] for _ in range(ny)]

        u_next, v_next, p_next = navier_stokes_2d_step(u, v, p, dt=0.001, dx=0.2, dy=0.2, nit=5)
        self.assertEqual(len(u_next), ny)
        self.assertEqual(len(u_next[0]), nx)
        self.assertEqual(u_next[ny - 1][5], 1.0)  # Moving top boundary

        u_cav, v_cav, p_cav = simulate_cavity_flow(nx=11, ny=11, steps=5, dt=0.001, nit=5)
        self.assertEqual(len(u_cav), 11)

    def test_tsp(self) -> None:
        # 4 cities arranged on a 1x1 square
        square_cities = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        dist = tsp_tour_distance(square_cities, [0, 1, 2, 3])
        self.assertAlmostEqual(dist, 4.0, places=5)

        best_tour, best_dist = tsp_simulated_annealing(
            square_cities,
            num_walkers=4,
            steps=50,
            seed=123,
        )
        self.assertEqual(len(best_tour), 4)
        self.assertAlmostEqual(best_dist, 4.0, places=3)

    def test_mcp_registry_problems_discovery(self) -> None:
        reg = get_registry()
        search_collatz = reg.search("collatz")
        self.assertGreaterEqual(len(search_collatz), 1)
        self.assertTrue(any(item["name"] == "collatz_batch_verify" for item in search_collatz))

        search_goldbach = reg.search("goldbach")
        self.assertGreaterEqual(len(search_goldbach), 1)

        search_three_body = reg.search("three_body")
        self.assertGreaterEqual(len(search_three_body), 1)

        desc = reg.describe("lucas_lehmer_mersenne")
        self.assertIsNotNone(desc)
        if desc is not None:
            self.assertEqual(desc["category"], "problems.primes")

    def test_cel_evaluation_of_problems(self) -> None:
        try:
            from funcpie.mcp.cel_engine import evaluate_cel
            res = evaluate_cel("mersenne_number(5)")
            self.assertTrue(res["success"])
            self.assertEqual(res["result"], 31)

            res_ll = evaluate_cel("lucas_lehmer_mersenne(5)")
            self.assertTrue(res_ll["success"])
            self.assertEqual(res_ll["result"], True)
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
