"""Unit tests for the OpenAI/math verification suite (BSD, Modular Forms, LLL, Ramsey)."""

from __future__ import annotations

import math
import unittest

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
    dot_product,
    gram_schmidt,
    lattice_determinant,
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


class TestOpenAIMathSuite(unittest.TestCase):
    """Test verification suite for OpenAI math problems and theoretical conjectures."""

    def test_elliptic_curve_algebra(self) -> None:
        """Test Weierstrass curve geometry, discriminant, and j-invariant."""
        # y^2 = x^3 - x (Delta = -16 * (4*(-1)^3) = 64)
        delta = discriminant(-1.0, 0.0)
        self.assertAlmostEqual(delta, 64.0)

        j = j_invariant(-1.0, 0.0)
        self.assertAlmostEqual(j, 1728.0)

        curve = EllipticCurve(-1.0, 0.0)
        self.assertTrue(curve.is_smooth)
        self.assertTrue(curve.contains_point(0.0, 0.0))
        self.assertTrue(curve.contains_point(1.0, 0.0))
        self.assertTrue(curve.contains_point(-1.0, 0.0))

        # Test singular curve raises error
        with self.assertRaises(ValueError):
            EllipticCurve(0.0, 0.0)

    def test_elliptic_point_addition(self) -> None:
        """Test group law addition and scalar multiplication on elliptic curve."""
        curve = EllipticCurve(-1.0, 0.0)
        # Point (0, 0) has order 2, so 2 * (0, 0) = O
        p = (0.0, 0.0)
        two_p = curve.point_multiply(p, 2)
        self.assertIsNone(two_p)

        # Addition with identity
        self.assertEqual(curve.point_add(None, p), p)
        self.assertEqual(curve.point_add(p, None), p)

    def test_ap_point_counting_and_bsd(self) -> None:
        """Test Frobenius trace a_p, real period Omega_E, and BSD leading-term ratio."""
        # For y^2 = x^3 - x (LMFDB 32.a2 / 32.a1)
        # a_2 = 0, a_3 = 0, a_5 = -2, a_7 = 0, a_11 = 0, a_13 = 6
        primes = [2, 3, 5, 7, 11, 13]
        aps = elliptic_ap_batch(-1, 0, primes)
        self.assertEqual(aps, [0, 0, -2, 0, 0, 6])
        self.assertEqual(ap_point_count(-1, 0, 5), -2)
        self.assertEqual(ap_point_count(-1, 0, 13), 6)

        # Largest root of x^3 - x = 0 is 1.0
        e1 = largest_real_root(-1.0, 0.0)
        self.assertAlmostEqual(e1, 1.0)

        # Real period Omega_E is approx 2.6220575
        omega = real_period(-1.0, 0.0, steps=2000)
        self.assertAlmostEqual(omega, 2.6220575, places=5)

        # BSD ratio estimate is positive
        ratio = bsd_ratio_estimate(-1, 0, [3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
        self.assertGreater(ratio, 0.0)

    def test_goldfeld_quadratic_twists(self) -> None:
        """Test Goldfeld quadratic twist families evaluation."""
        d_list = [-3, -2, -1, 1, 2, 3, 5, 7]
        primes = [3, 5, 7, 11, 13, 17, 19]
        summary = goldfeld_twist_batch(-1, 0, d_list, primes)
        self.assertEqual(summary["total_twists"], len(d_list))
        self.assertIn("rank_zero_ratio", summary)
        self.assertGreaterEqual(float(str(summary["rank_zero_ratio"])), 0.0)

    def test_ramanujan_tau_and_deligne(self) -> None:
        """Test Ramanujan's tau function, multiplicativity, and Deligne bound."""
        # Known Ramanujan tau values:
        # tau(1)=1, tau(2)=-24, tau(3)=252, tau(4)=-1472, tau(5)=4830, tau(6)=-6048
        batch = ramanujan_tau_batch(6)
        expected = [1, -24, 252, -1472, 4830, -6048]
        self.assertEqual(batch, expected)
        self.assertEqual(ramanujan_tau(2), -24)
        self.assertEqual(ramanujan_tau(3), 252)
        self.assertEqual(ramanujan_tau(6), -6048)

        # Hecke Multiplicative property: gcd(2, 3) = 1 => tau(6) == tau(2) * tau(3)
        self.assertTrue(verify_multiplicative(2, 3))

        # Deligne's bound: |tau(p)| <= 2 * p^(11/2)
        for p in [2, 3, 5, 7]:
            valid, val, bound = verify_deligne_bound(p)
            self.assertTrue(valid)
            self.assertLessEqual(abs(val), bound)

    def test_lattice_lll_reduction(self) -> None:
        """Test Gram-Schmidt orthogonalization and LLL lattice basis reduction."""
        basis = [
            [1.0, 1.0, 1.0],
            [-1.0, 0.0, 2.0],
            [3.0, 5.0, 6.0],
        ]
        b_star, mu = gram_schmidt(basis)
        self.assertEqual(len(b_star), 3)

        # Check orthogonality of Gram-Schmidt vectors
        self.assertAlmostEqual(dot_product(b_star[0], b_star[1]), 0.0, places=6)
        self.assertAlmostEqual(dot_product(b_star[0], b_star[2]), 0.0, places=6)
        self.assertAlmostEqual(dot_product(b_star[1], b_star[2]), 0.0, places=6)

        # LLL reduction
        reduced = lll_reduction(basis, delta=0.75)
        self.assertEqual(len(reduced), 3)
        norm_orig = vector_norm(basis[2])
        norm_red = vector_norm(reduced[0])
        self.assertLess(norm_red, norm_orig)

        # Shortest vector bound
        bound = shortest_vector_bound(basis)
        self.assertGreater(bound, 0.0)

    def test_ramsey_graph_verification(self) -> None:
        """Test Ramsey graph clique and independent set property checker."""
        # 5-cycle graph C_5: contains no triangle (K_3) and no independent set of size 3 (I_3)
        c5 = [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ]
        self.assertTrue(is_ramsey_graph(c5, 3, 3))
        self.assertAlmostEqual(ramsey_edge_density(c5), 0.5)

        # Triangle graph K_3: has a 3-clique, so is NOT a valid (3, 3)-Ramsey graph
        k3 = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
        self.assertFalse(is_ramsey_graph(k3, 3, 3))

        # Stochastic search for R(3, 3) graph of order 5
        found = ramsey_random_search(5, 3, 3, trials=300, seed=42)
        self.assertIsNotNone(found)
        if found is not None:
            self.assertTrue(is_ramsey_graph(found, 3, 3))

    def test_mcp_discovery_and_cel(self) -> None:
        """Test FastMCP registry discovery and CEL execution for OpenAI math operations."""
        from funcpie.mcp.registry import get_registry
        reg = get_registry()

        # Discovery
        bsd_results = reg.search("bsd")
        self.assertGreaterEqual(len(bsd_results), 1)
        self.assertTrue(any(item["name"] == "bsd_ratio_estimate" for item in bsd_results))

        modular_results = reg.search("ramanujan")
        self.assertGreaterEqual(len(modular_results), 1)
        self.assertTrue(any(item["name"] == "ramanujan_tau" for item in modular_results))

        lattice_results = reg.search("lll")
        self.assertGreaterEqual(len(lattice_results), 1)

        ramsey_results = reg.search("ramsey")
        self.assertGreaterEqual(len(ramsey_results), 1)

        # CEL Evaluation
        try:
            from funcpie.mcp.cel_engine import evaluate_cel
            res_disc = evaluate_cel("discriminant(-1.0, 0.0)")
            self.assertTrue(res_disc["success"])
            self.assertEqual(res_disc["result"], 64.0)

            res_tau = evaluate_cel("ramanujan_tau(2)")
            self.assertTrue(res_tau["success"])
            self.assertEqual(res_tau["result"], -24)
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()

