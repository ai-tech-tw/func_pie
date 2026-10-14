"""CPU fallback backend implementing numeric operations in pure Python."""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple


class CPUBackend:
    """Pure Python CPU execution engine for numeric, scientific, and hard math operations."""

    name: str = "cpu"

    @staticmethod
    def is_available() -> bool:
        """CPU backend is always available."""
        return True

    @staticmethod
    def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Compute matrix multiplication C = A x B on CPU."""
        rows_a = len(a)
        cols_a = len(a[0]) if rows_a > 0 else 0
        rows_b = len(b)
        cols_b = len(b[0]) if rows_b > 0 else 0

        if cols_a != rows_b:
            raise ValueError(f"Cannot multiply matrix with shapes ({rows_a}, {cols_a}) and ({rows_b}, {cols_b})")

        b_t: List[List[float]] = [[b[r][c] for r in range(rows_b)] for c in range(cols_b)]
        result: List[List[float]] = []
        for row_a in a:
            res_row: List[float] = [
                sum(x * y for x, y in zip(row_a, col_b))
                for col_b in b_t
            ]
            result.append(res_row)
        return result

    @staticmethod
    def vector_dot(v1: List[float], v2: List[float]) -> float:
        """Compute dot product of two vectors on CPU."""
        if len(v1) != len(v2):
            raise ValueError(f"Vector length mismatch: {len(v1)} != {len(v2)}")
        return sum(x * y for x, y in zip(v1, v2))

    @staticmethod
    def vector_add(v1: List[float], v2: List[float]) -> List[float]:
        """Compute vector addition on CPU."""
        if len(v1) != len(v2):
            raise ValueError(f"Vector length mismatch: {len(v1)} != {len(v2)}")
        return [x + y for x, y in zip(v1, v2)]

    @staticmethod
    def simpson_integrate_samples(samples: List[float], h: float) -> float:
        """Composite Simpson's rule over evenly spaced sample points on CPU."""
        n = len(samples) - 1
        if n <= 0 or n % 2 != 0:
            raise ValueError("Simpson's rule requires an even number of intervals (odd number of samples)")

        total = samples[0] + samples[-1]
        for i in range(1, n, 2):
            total += 4.0 * samples[i]
        for i in range(2, n, 2):
            total += 2.0 * samples[i]
        return total * (h / 3.0)

    @staticmethod
    def batch_projectile(
        v0_list: List[float],
        angle_deg_list: List[float],
        g: float = 9.80665,
    ) -> List[Tuple[float, float, float]]:
        """Compute projectile motion results (max_height, flight_time, range) for batches."""
        if len(v0_list) != len(angle_deg_list):
            raise ValueError("Mismatched batch lengths for v0 and angles")

        results: List[Tuple[float, float, float]] = []
        for v0, angle_deg in zip(v0_list, angle_deg_list):
            rad = math.radians(angle_deg)
            vy0 = v0 * math.sin(rad)
            vx0 = v0 * math.cos(rad)
            t_flight = 2.0 * vy0 / g if vy0 > 0 else 0.0
            h_max = (vy0 ** 2) / (2.0 * g) if vy0 > 0 else 0.0
            h_range = vx0 * t_flight
            results.append((h_max, t_flight, h_range))
        return results

    # =========================================================================
    # Mathematical Problems & Conjectures Suite (CPU Reference Implementations)
    # =========================================================================

    @staticmethod
    def collatz_batch(start: int, count: int, max_steps: int = 100000) -> List[Tuple[int, int, int]]:
        """Compute Collatz stopping time and maximum trajectory value for a range of integers."""
        if start < 1 or count < 1:
            raise ValueError("Start and count must be positive integers.")

        results: List[Tuple[int, int, int]] = []
        for n in range(start, start + count):
            curr = n
            steps = 0
            peak = curr
            while curr > 1 and steps < max_steps:
                if curr % 2 == 0:
                    curr //= 2
                else:
                    curr = 3 * curr + 1
                if curr > peak:
                    peak = curr
                steps += 1
            results.append((n, steps, peak))
        return results

    @staticmethod
    def goldbach_batch(start_even: int, count: int) -> List[Tuple[int, int, int]]:
        """Verify Goldbach's conjecture for consecutive even numbers >= 4."""
        if start_even < 4 or start_even % 2 != 0:
            raise ValueError("start_even must be an even integer >= 4.")

        max_val = start_even + 2 * count
        # Simple sieve up to max_val
        is_prime_arr = [True] * (max_val + 1)
        is_prime_arr[0] = is_prime_arr[1] = False
        limit = int(math.isqrt(max_val))
        for p in range(2, limit + 1):
            if is_prime_arr[p]:
                for mult in range(p * p, max_val + 1, p):
                    is_prime_arr[mult] = False

        results: List[Tuple[int, int, int]] = []
        for idx in range(count):
            even_n = start_even + 2 * idx
            found = False
            for p in range(2, even_n // 2 + 1):
                if is_prime_arr[p] and is_prime_arr[even_n - p]:
                    results.append((even_n, p, even_n - p))
                    found = True
                    break
            if not found:
                results.append((even_n, -1, -1))
        return results

    @staticmethod
    def lucas_lehmer(p: int) -> bool:
        """Lucas-Lehmer primality test for Mersenne number M_p = 2^p - 1."""
        if p < 2:
            return False
        if p == 2:
            return True
        # Mersenne number
        m_p = (1 << p) - 1
        s = 4
        for _ in range(p - 2):
            s = (s * s - 2) % m_p
        return s == 0

    @staticmethod
    def segmented_sieve(low: int, high: int) -> List[int]:
        """Segmented Sieve of Eratosthenes to find primes in [low, high]."""
        if high < 2 or low > high:
            return []
        effective_low = max(2, low)
        limit = int(math.isqrt(high)) + 1

        # Small primes up to sqrt(high)
        is_prime_small = [True] * (limit + 1)
        is_prime_small[0] = is_prime_small[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime_small[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime_small[j] = False
        primes_small = [i for i, prime in enumerate(is_prime_small) if prime]

        # Segment array
        size = high - effective_low + 1
        is_prime_seg = [True] * size
        for p in primes_small:
            start = max(p * p, ((effective_low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                is_prime_seg[j - effective_low] = False

        return [effective_low + idx for idx, prime in enumerate(is_prime_seg) if prime]

    @staticmethod
    def riemann_zeta_critical(t_values: List[float], terms: int = 1000) -> List[Tuple[float, float, float, float]]:
        """Evaluate Riemann Zeta function along the critical line s = 1/2 + i*t using Dirichlet Eta expansion."""
        results: List[Tuple[float, float, float, float]] = []
        for t in t_values:
            # s = 0.5 + i*t
            # eta(s) = sum_{n=1}^terms (-1)^(n-1) / n^s
            # n^(-s) = n^(-0.5) * (cos(t * ln n) - i * sin(t * ln n))
            eta_real = 0.0
            eta_imag = 0.0
            sign = 1.0
            for n in range(1, terms + 1):
                inv_sqrt_n = 1.0 / math.sqrt(n)
                theta = -t * math.log(n)
                eta_real += sign * inv_sqrt_n * math.cos(theta)
                eta_imag += sign * inv_sqrt_n * math.sin(theta)
                sign = -sign

            # factor = 1 - 2^(1 - s) = 1 - 2^(0.5 - i*t)
            # 2^(0.5 - i*t) = sqrt(2) * (cos(t * ln 2) - i * sin(t * ln 2))
            sqrt2 = math.sqrt(2.0)
            ln2_t = t * math.log(2.0)
            factor_real = 1.0 - sqrt2 * math.cos(-ln2_t)
            factor_imag = -sqrt2 * math.sin(-ln2_t)

            denom = factor_real * factor_real + factor_imag * factor_imag
            if denom == 0.0:
                zeta_real = 0.0
                zeta_imag = 0.0
            else:
                # zeta = eta / factor
                zeta_real = (eta_real * factor_real + eta_imag * factor_imag) / denom
                zeta_imag = (eta_imag * factor_real - eta_real * factor_imag) / denom

            mag = math.hypot(zeta_real, zeta_imag)
            results.append((t, zeta_real, zeta_imag, mag))
        return results

    @staticmethod
    def three_body_step(
        masses: Tuple[float, float, float],
        positions: List[Tuple[float, float, float]],
        velocities: List[Tuple[float, float, float]],
        dt: float,
        g: float = 1.0,
        softening: float = 1e-3,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """RK4 numerical integration step for the classical Three-Body gravitational problem."""
        if len(positions) != 3 or len(velocities) != 3 or len(masses) != 3:
            raise ValueError("three_body_step requires exactly 3 bodies.")

        def compute_acc(pos: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
            acc = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
            acc_list = [[0.0, 0.0, 0.0] for _ in range(3)]
            for i in range(3):
                xi, yi, zi = pos[i]
                for j in range(3):
                    if i != j:
                        xj, yj, zj = pos[j]
                        dx = xj - xi
                        dy = yj - yi
                        dz = zj - zi
                        dist_sq = dx * dx + dy * dy + dz * dz + softening * softening
                        inv_dist3 = (dist_sq ** -1.5)
                        scale = g * masses[j] * inv_dist3
                        acc_list[i][0] += scale * dx
                        acc_list[i][1] += scale * dy
                        acc_list[i][2] += scale * dz
            return [(acc_list[i][0], acc_list[i][1], acc_list[i][2]) for i in range(3)]

        # RK4
        p0 = positions
        v0 = velocities
        a0 = compute_acc(p0)

        # k1
        k1_p = v0
        k1_v = a0

        # k2
        p1 = [(p0[i][c] + 0.5 * dt * k1_p[i][c]) for i in range(3) for c in range(3)]
        p1_tuples = [(p1[0], p1[1], p1[2]), (p1[3], p1[4], p1[5]), (p1[6], p1[7], p1[8])]
        v1 = [(v0[i][0] + 0.5 * dt * k1_v[i][0], v0[i][1] + 0.5 * dt * k1_v[i][1], v0[i][2] + 0.5 * dt * k1_v[i][2]) for i in range(3)]
        a1 = compute_acc(p1_tuples)
        k2_p = v1
        k2_v = a1

        # k3
        p2 = [(p0[i][c] + 0.5 * dt * k2_p[i][c]) for i in range(3) for c in range(3)]
        p2_tuples = [(p2[0], p2[1], p2[2]), (p2[3], p2[4], p2[5]), (p2[6], p2[7], p2[8])]
        v2 = [(v0[i][0] + 0.5 * dt * k2_v[i][0], v0[i][1] + 0.5 * dt * k2_v[i][1], v0[i][2] + 0.5 * dt * k2_v[i][2]) for i in range(3)]
        a2 = compute_acc(p2_tuples)
        k3_p = v2
        k3_v = a2

        # k4
        p3 = [(p0[i][c] + dt * k3_p[i][c]) for i in range(3) for c in range(3)]
        p3_tuples = [(p3[0], p3[1], p3[2]), (p3[3], p3[4], p3[5]), (p3[6], p3[7], p3[8])]
        v3 = [(v0[i][0] + dt * k3_v[i][0], v0[i][1] + dt * k3_v[i][1], v0[i][2] + dt * k3_v[i][2]) for i in range(3)]
        a3 = compute_acc(p3_tuples)
        k4_p = v3
        k4_v = a3

        # Update
        new_pos: List[Tuple[float, float, float]] = []
        new_vel: List[Tuple[float, float, float]] = []
        dt_6 = dt / 6.0
        for i in range(3):
            px = p0[i][0] + dt_6 * (k1_p[i][0] + 2.0 * k2_p[i][0] + 2.0 * k3_p[i][0] + k4_p[i][0])
            py = p0[i][1] + dt_6 * (k1_p[i][1] + 2.0 * k2_p[i][1] + 2.0 * k3_p[i][1] + k4_p[i][1])
            pz = p0[i][2] + dt_6 * (k1_p[i][2] + 2.0 * k2_p[i][2] + 2.0 * k3_p[i][2] + k4_p[i][2])
            vx = v0[i][0] + dt_6 * (k1_v[i][0] + 2.0 * k2_v[i][0] + 2.0 * k3_v[i][0] + k4_v[i][0])
            vy = v0[i][1] + dt_6 * (k1_v[i][1] + 2.0 * k2_v[i][1] + 2.0 * k3_v[i][1] + k4_v[i][1])
            vz = v0[i][2] + dt_6 * (k1_v[i][2] + 2.0 * k2_v[i][2] + 2.0 * k3_v[i][2] + k4_v[i][2])
            new_pos.append((px, py, pz))
            new_vel.append((vx, vy, vz))

        return new_pos, new_vel

    @staticmethod
    def lorenz_ensemble(
        initial_states: List[Tuple[float, float, float]],
        dt: float,
        steps: int,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
    ) -> List[Tuple[float, float, float]]:
        """Evolve an ensemble of Lorenz attractor states in parallel using RK4."""
        evolved: List[Tuple[float, float, float]] = []
        dt_6 = dt / 6.0
        for x, y, z in initial_states:
            curr_x, curr_y, curr_z = x, y, z
            for _ in range(steps):
                # k1
                k1_x = sigma * (curr_y - curr_x)
                k1_y = curr_x * (rho - curr_z) - curr_y
                k1_z = curr_x * curr_y - beta * curr_z

                # k2
                x2 = curr_x + 0.5 * dt * k1_x
                y2 = curr_y + 0.5 * dt * k1_y
                z2 = curr_z + 0.5 * dt * k1_z
                k2_x = sigma * (y2 - x2)
                k2_y = x2 * (rho - z2) - y2
                k2_z = x2 * y2 - beta * z2

                # k3
                x3 = curr_x + 0.5 * dt * k2_x
                y3 = curr_y + 0.5 * dt * k2_y
                z3 = curr_z + 0.5 * dt * k2_z
                k3_x = sigma * (y3 - x3)
                k3_y = x3 * (rho - z3) - y3
                k3_z = x3 * y3 - beta * z3

                # k4
                x4 = curr_x + dt * k3_x
                y4 = curr_y + dt * k3_y
                z4 = curr_z + dt * k3_z
                k4_x = sigma * (y4 - x4)
                k4_y = x4 * (rho - z4) - y4
                k4_z = x4 * y4 - beta * z4

                curr_x += dt_6 * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
                curr_y += dt_6 * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
                curr_z += dt_6 * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
            evolved.append((curr_x, curr_y, curr_z))
        return evolved

    @staticmethod
    def navier_stokes_step(
        u: List[List[float]],
        v: List[List[float]],
        p: List[List[float]],
        dt: float,
        dx: float,
        dy: float,
        visc: float,
        rho: float,
        nit: int = 20,
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """2D Incompressible Navier-Stokes finite difference step with Jacobi pressure Poisson solver."""
        ny = len(u)
        nx = len(u[0]) if ny > 0 else 0

        # Build RHS of pressure Poisson equation
        b = [[0.0 for _ in range(nx)] for _ in range(ny)]
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                du_dx = (u[j][i + 1] - u[j][i - 1]) / (2.0 * dx)
                dv_dy = (v[j + 1][i] - v[j - 1][i]) / (2.0 * dy)
                du_dy = (u[j + 1][i] - u[j - 1][i]) / (2.0 * dy)
                dv_dx = (v[j][i + 1] - v[j][i - 1]) / (2.0 * dx)

                b[j][i] = (rho * (1.0 / dt * (du_dx + dv_dy) - (du_dx ** 2 + 2.0 * du_dy * dv_dx + dv_dy ** 2)))

        # Jacobi iteration for pressure
        p_curr = [[p[j][i] for i in range(nx)] for j in range(ny)]
        for _ in range(nit):
            p_next = [[p_curr[j][i] for i in range(nx)] for j in range(ny)]
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    p_next[j][i] = (((p_curr[j][i + 1] + p_curr[j][i - 1]) * dy ** 2 +
                                     (p_curr[j + 1][i] + p_curr[j - 1][i]) * dx ** 2 -
                                     b[j][i] * dx ** 2 * dy ** 2) /
                                    (2.0 * (dx ** 2 + dy ** 2)))

            # Boundary conditions for pressure (Neumann: dp/dn = 0)
            for j in range(ny):
                p_next[j][nx - 1] = p_next[j][nx - 2]
                p_next[j][0] = p_next[j][1]
            for i in range(nx):
                p_next[0][i] = p_next[1][i]
                p_next[ny - 1][i] = 0.0  # Dirichlet at top
            p_curr = p_next

        # Velocity update
        u_next = [[u[j][i] for i in range(nx)] for j in range(ny)]
        v_next = [[v[j][i] for i in range(nx)] for j in range(ny)]
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u_adv = u[j][i] * (u[j][i] - u[j][i - 1]) / dx + v[j][i] * (u[j][i] - u[j - 1][i]) / dy
                v_adv = u[j][i] * (v[j][i] - v[j][i - 1]) / dx + v[j][i] * (v[j][i] - v[j - 1][i]) / dy

                u_diff = visc * ((u[j][i + 1] - 2.0 * u[j][i] + u[j][i - 1]) / dx ** 2 +
                                 (u[j + 1][i] - 2.0 * u[j][i] + u[j - 1][i]) / dy ** 2)
                v_diff = visc * ((v[j][i + 1] - 2.0 * v[j][i] + v[j][i - 1]) / dx ** 2 +
                                 (v[j + 1][i] - 2.0 * v[j][i] + v[j - 1][i]) / dy ** 2)

                dp_dx = (p_curr[j][i + 1] - p_curr[j][i - 1]) / (2.0 * dx)
                dp_dy = (p_curr[j + 1][i] - p_curr[j - 1][i]) / (2.0 * dy)

                u_next[j][i] = u[j][i] - dt * (u_adv + 1.0 / rho * dp_dx - u_diff)
                v_next[j][i] = v[j][i] - dt * (v_adv + 1.0 / rho * dp_dy - v_diff)

        # Cavity boundary conditions (lid-driven cavity: top u = 1.0)
        for i in range(nx):
            u_next[0][i] = 0.0
            u_next[ny - 1][i] = 1.0
            v_next[0][i] = 0.0
            v_next[ny - 1][i] = 0.0
        for j in range(ny):
            u_next[j][0] = 0.0
            u_next[j][nx - 1] = 0.0
            v_next[j][0] = 0.0
            v_next[j][nx - 1] = 0.0

        return u_next, v_next, p_curr

    @staticmethod
    def tsp_simulated_annealing(
        coords: List[Tuple[float, float]],
        num_walkers: int = 16,
        steps: int = 500,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        seed: Optional[int] = 42,
    ) -> Tuple[List[int], float]:
        """Multi-walker parallel simulated annealing for the Traveling Salesperson Problem (TSP)."""
        num_cities = len(coords)
        if num_cities < 2:
            return list(range(num_cities)), 0.0

        rnd = random.Random(seed)

        def tour_distance(tour: List[int]) -> float:
            d = 0.0
            for idx in range(num_cities):
                c1 = coords[tour[idx]]
                c2 = coords[tour[(idx + 1) % num_cities]]
                d += math.hypot(c1[0] - c2[0], c1[1] - c2[1])
            return d

        best_tour = list(range(num_cities))
        best_dist = tour_distance(best_tour)

        for _ in range(num_walkers):
            # Each walker starts with a random permutation
            current_tour = list(range(num_cities))
            rnd.shuffle(current_tour)
            current_dist = tour_distance(current_tour)
            if current_dist < best_dist:
                best_dist = current_dist
                best_tour = list(current_tour)

            temp = initial_temp
            for _ in range(steps):
                # 2-opt swap
                i, j = rnd.sample(range(num_cities), 2)
                if i > j:
                    i, j = j, i
                neighbor_tour = current_tour[:i] + current_tour[i : j + 1][::-1] + current_tour[j + 1 :]
                neighbor_dist = tour_distance(neighbor_tour)

                delta = neighbor_dist - current_dist
                if delta < 0.0 or rnd.random() < math.exp(-delta / max(1e-9, temp)):
                    current_tour = neighbor_tour
                    current_dist = neighbor_dist
                    if current_dist < best_dist:
                        best_dist = current_dist
                        best_tour = list(current_tour)

                temp *= cooling_rate

        return best_tour, best_dist

    @staticmethod
    def nbody_step(
        masses: List[float],
        positions: List[Tuple[float, float, float]],
        velocities: List[Tuple[float, float, float]],
        dt: float,
        g: float = 1.0,
        softening: float = 1e-3,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """Compute single-step N-body gravitational force updates for all particles."""
        n = len(masses)
        new_pos: List[Tuple[float, float, float]] = []
        new_vel: List[Tuple[float, float, float]] = []

        for i in range(n):
            xi, yi, zi = positions[i]
            vxi, vyi, vzi = velocities[i]
            axi, ayi, azi = 0.0, 0.0, 0.0

            for j in range(n):
                if i != j:
                    dx = positions[j][0] - xi
                    dy = positions[j][1] - yi
                    dz = positions[j][2] - zi
                    dist_sq = dx * dx + dy * dy + dz * dz + softening * softening
                    inv_dist3 = dist_sq ** -1.5
                    factor = g * masses[j] * inv_dist3
                    axi += factor * dx
                    ayi += factor * dy
                    azi += factor * dz

            vxi_next = vxi + axi * dt
            vyi_next = vyi + ayi * dt
            vzi_next = vzi + azi * dt

            xi_next = xi + vxi_next * dt
            yi_next = yi + vyi_next * dt
            zi_next = zi + vzi_next * dt

            new_vel.append((vxi_next, vyi_next, vzi_next))
            new_pos.append((xi_next, yi_next, zi_next))

        return new_pos, new_vel

    @staticmethod
    def heat_diffusion_2d(
        grid: List[List[float]],
        alpha: float,
        dt: float,
        dx: float,
        dy: float,
        steps: int,
    ) -> List[List[float]]:
        """2D Finite Difference Heat Diffusion solver."""
        ny = len(grid)
        nx = len(grid[0]) if ny > 0 else 0
        curr = [[grid[j][i] for i in range(nx)] for j in range(ny)]

        rx = alpha * dt / (dx * dx)
        ry = alpha * dt / (dy * dy)

        for _ in range(steps):
            next_grid = [[curr[j][i] for i in range(nx)] for j in range(ny)]
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    d2x = curr[j][i + 1] - 2.0 * curr[j][i] + curr[j][i - 1]
                    d2y = curr[j + 1][i] - 2.0 * curr[j][i] + curr[j - 1][i]
                    next_grid[j][i] = curr[j][i] + rx * d2x + ry * d2y
            curr = next_grid
        return curr

    @staticmethod
    def mandelbrot(
        width: int,
        height: int,
        max_iter: int = 100,
        x_min: float = -2.0,
        x_max: float = 0.5,
        y_min: float = -1.25,
        y_max: float = 1.25,
    ) -> List[List[int]]:
        """Generate Mandelbrot set escape-time 2D grid."""
        grid: List[List[int]] = []
        dx = (x_max - x_min) / max(1, width - 1)
        dy = (y_max - y_min) / max(1, height - 1)

        for j in range(height):
            c_imag = y_min + j * dy
            row: List[int] = []
            for i in range(width):
                c_real = x_min + i * dx
                zx = 0.0
                zy = 0.0
                iteration = 0
                while zx * zx + zy * zy <= 4.0 and iteration < max_iter:
                    zx_temp = zx * zx - zy * zy + c_real
                    zy = 2.0 * zx * zy + c_imag
                    zx = zx_temp
                    iteration += 1
                row.append(iteration)
            grid.append(row)
        return grid

    @staticmethod
    def pairwise_distances(
        x: List[List[float]],
        y: Optional[List[List[float]]] = None,
    ) -> List[List[float]]:
        """Compute pairwise Euclidean distance matrix between points."""
        target_y = x if y is None else y
        dist_matrix: List[List[float]] = []

        for pt_x in x:
            row: List[float] = []
            for pt_y in target_y:
                sq_sum = sum((a - b) ** 2 for a, b in zip(pt_x, pt_y))
                row.append(math.sqrt(sq_sum))
            dist_matrix.append(row)
        return dist_matrix

    @staticmethod
    def elliptic_ap_batch(a: int, b: int, primes: List[int]) -> List[int]:
        """Compute trace of Frobenius a_p = p + 1 - #E(F_p) for elliptic curve y^2 = x^3 + ax + b across primes."""
        results: List[int] = []
        for p in primes:
            if p <= 1:
                results.append(0)
                continue
            if p == 2:
                # E(F_2) calculation
                pts = 1  # Point at infinity O
                for x in (0, 1):
                    rhs = (x * x * x + a * x + b) % 2
                    # y in {0, 1}: y^2 mod 2 is y
                    if rhs == 0:
                        pts += 1  # y=0
                    elif rhs == 1:
                        pts += 1  # y=1
                results.append(2 + 1 - pts)
                continue

            # For odd primes, a_p = - sum_{x=0}^{p-1} ( (x^3 + ax + b) / p )
            s = 0
            exp = (p - 1) // 2
            for x in range(p):
                val = (x * x * x + a * x + b) % p
                if val != 0:
                    leg = pow(val, exp, p)
                    s += -1 if leg == p - 1 else 1
            results.append(-s)
        return results

    @staticmethod
    def ramanujan_tau_batch(limit: int) -> List[int]:
        """Compute Ramanujan tau values tau(1), ..., tau(limit) using logarithmic derivative recurrence."""
        if limit < 1:
            return []
        size = limit + 1
        sigma = [0] * size
        for i in range(1, size):
            for j in range(i, size, i):
                sigma[j] += i

        c = [0] * size
        c[0] = 1
        for n in range(1, size):
            s = sum(sigma[k] * c[n - k] for k in range(1, n + 1))
            c[n] = (-24 * s) // n

        return [c[i - 1] for i in range(1, size)]

    @staticmethod
    def lll_reduction(basis: List[List[float]], delta: float = 0.75) -> List[List[float]]:
        """Lenstra-Lenstra-Lovasz (LLL) lattice basis reduction."""
        n = len(basis)
        if n <= 1:
            return [[float(x) for x in v] for v in basis]
        m = len(basis[0])
        b = [[float(x) for x in v] for v in basis]

        def _dot(v1: List[float], v2: List[float]) -> float:
            return sum(x * y for x, y in zip(v1, v2))

        def _gram_schmidt() -> Tuple[List[List[float]], List[List[float]]]:
            b_star: List[List[float]] = []
            mu: List[List[float]] = [[0.0] * n for _ in range(n)]
            for i in range(n):
                v = list(b[i])
                for j in range(i):
                    denom = _dot(b_star[j], b_star[j])
                    if abs(denom) > 1e-15:
                        mu[i][j] = _dot(b[i], b_star[j]) / denom
                    else:
                        mu[i][j] = 0.0
                    for d in range(m):
                        v[d] -= mu[i][j] * b_star[j][d]
                b_star.append(v)
                mu[i][i] = 1.0
            return b_star, mu

        b_star, mu = _gram_schmidt()
        k = 1
        while k < n:
            for j in range(k - 1, -1, -1):
                if abs(mu[k][j]) > 0.5:
                    q = round(mu[k][j])
                    for d in range(m):
                        b[k][d] -= q * b[j][d]
                    b_star, mu = _gram_schmidt()

            # Lovasz condition: ||b_k*||^2 >= (delta - mu_{k, k-1}^2) * ||b_{k-1}*||^2
            norm_k = _dot(b_star[k], b_star[k])
            norm_prev = _dot(b_star[k - 1], b_star[k - 1])
            if norm_k >= (delta - mu[k][k - 1] ** 2) * norm_prev:
                k += 1
            else:
                b[k], b[k - 1] = b[k - 1], b[k]
                b_star, mu = _gram_schmidt()
                k = max(k - 1, 1)

        return b

    @staticmethod
    def ramsey_check(adj_matrix: List[List[int]], k: int, l: int) -> bool:
        """Verify whether the graph contains no clique of size k and no independent set of size l."""
        n = len(adj_matrix)
        if n == 0:
            return True

        def _has_clique(adj: List[List[int]], target: int) -> bool:
            if target <= 1:
                return True if n >= target else False

            def _search(candidates: List[int], current_size: int) -> bool:
                if current_size == target:
                    return True
                while candidates:
                    if current_size + len(candidates) < target:
                        return False
                    v = candidates.pop()
                    next_cand = [u for u in candidates if adj[v][u]]
                    if _search(next_cand, current_size + 1):
                        return True
                return False

            return _search(list(range(n)), 0)

        # Check k-clique
        if _has_clique(adj_matrix, k):
            return False

        # Check l-independent set (clique in complement graph)
        complement = [[0 if i == j or adj_matrix[i][j] else 1 for j in range(n)] for i in range(n)]
        if _has_clique(complement, l):
            return False

        return True

