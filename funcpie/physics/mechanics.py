"""Classical mechanics: kinematics, dynamics, gravitation, and oscillations."""

from __future__ import annotations

import math
from typing import List, NamedTuple, Tuple

from funcpie.physics.constants import G, g


class ProjectileSummary(NamedTuple):
    flight_time: float
    max_height: float
    horizontal_range: float


def velocity(displacement: float, time: float) -> float:
    """Average velocity: v = s / t."""
    if time <= 0:
        raise ValueError("time must be strictly positive")
    return displacement / time


def acceleration(v_initial: float, v_final: float, time: float) -> float:
    """Average acceleration: a = (v_final - v_initial) / t."""
    if time <= 0:
        raise ValueError("time must be strictly positive")
    return (v_final - v_initial) / time


def displacement_constant_acc(v0: float, a: float, t: float) -> float:
    """Displacement under constant acceleration: s = v0*t + 0.5*a*t^2."""
    if t < 0:
        raise ValueError("time cannot be negative")
    return v0 * t + 0.5 * a * (t ** 2)


def final_velocity(v0: float, a: float, t: float) -> float:
    """Final velocity under constant acceleration: v = v0 + a*t."""
    if t < 0:
        raise ValueError("time cannot be negative")
    return v0 + a * t


def final_velocity_squared(v0: float, a: float, s: float) -> float:
    """Final velocity squared: v^2 = v0^2 + 2*a*s."""
    val = (v0 ** 2) + 2.0 * a * s
    if val < 0:
        raise ValueError("v^2 cannot be negative for real velocity")
    return math.sqrt(val)


def projectile_motion(v0: float, angle_deg: float, gravity: float = g) -> ProjectileSummary:
    """Compute projectile motion summary parameters (assuming launch from ground level y=0)."""
    if v0 < 0:
        raise ValueError("Initial velocity cannot be negative")
    if gravity <= 0:
        raise ValueError("Gravity must be positive")
    theta = math.radians(angle_deg)
    v0y = v0 * math.sin(theta)
    v0x = v0 * math.cos(theta)
    if v0y <= 0:
        return ProjectileSummary(flight_time=0.0, max_height=0.0, horizontal_range=0.0)

    flight_time = 2.0 * v0y / gravity
    max_height = (v0y ** 2) / (2.0 * gravity)
    horizontal_range = v0x * flight_time
    return ProjectileSummary(
        flight_time=flight_time,
        max_height=max_height,
        horizontal_range=horizontal_range,
    )


def projectile_trajectory(
    v0: float,
    angle_deg: float,
    num_points: int = 50,
    gravity: float = g,
) -> List[Tuple[float, float]]:
    """Compute (x, y) coordinates along the projectile trajectory."""
    summary = projectile_motion(v0, angle_deg, gravity)
    if summary.flight_time == 0.0 or num_points <= 1:
        return [(0.0, 0.0)]

    theta = math.radians(angle_deg)
    v0x = v0 * math.cos(theta)
    v0y = v0 * math.sin(theta)
    dt = summary.flight_time / (num_points - 1)
    points = []
    for i in range(num_points):
        t = i * dt
        x = v0x * t
        y = max(0.0, v0y * t - 0.5 * gravity * (t ** 2))
        points.append((x, y))
    return points


def batch_projectile_motion(
    v0_list: List[float],
    angle_deg_list: List[float],
    gravity: float = g,
) -> List[Tuple[float, float, float]]:
    """Compute (max_height, flight_time, range) for a batch of projectiles using accelerated backend."""
    from funcpie.backend import get_backend
    return get_backend().batch_projectile(v0_list, angle_deg_list, gravity)


def force(mass: float, acc: float) -> float:
    """Newton's second law: F = m * a."""
    if mass < 0:
        raise ValueError("mass cannot be negative")
    return mass * acc


def weight(mass: float, gravity: float = g) -> float:
    """Weight force of an object: W = m * g."""
    return force(mass, gravity)


def friction_force(friction_coeff: float, normal_force: float) -> float:
    """Frictional force: F_f = mu * N."""
    if friction_coeff < 0 or normal_force < 0:
        raise ValueError("friction_coeff and normal_force must be non-negative")
    return friction_coeff * normal_force


def centripetal_acceleration(v: float, r: float) -> float:
    """Centripetal acceleration: a_c = v^2 / r."""
    if r <= 0:
        raise ValueError("radius must be positive")
    return (v ** 2) / r


def centripetal_force(mass: float, v: float, r: float) -> float:
    """Centripetal force: F_c = m * v^2 / r."""
    return mass * centripetal_acceleration(v, r)


def gravitational_force(m1: float, m2: float, r: float) -> float:
    """Newton's universal law of gravitation: F = G * m1 * m2 / r^2."""
    if m1 < 0 or m2 < 0:
        raise ValueError("masses cannot be negative")
    if r <= 0:
        raise ValueError("distance r must be positive")
    return G * m1 * m2 / (r ** 2)


def gravitational_potential_energy(mass: float, height: float, gravity: float = g) -> float:
    """Uniform field gravitational potential energy: U = m * g * h."""
    if mass < 0:
        raise ValueError("mass cannot be negative")
    return mass * gravity * height


def universal_gravitational_pe(m1: float, m2: float, r: float) -> float:
    """Universal gravitational potential energy: U = -G * m1 * m2 / r."""
    if m1 < 0 or m2 < 0:
        raise ValueError("masses cannot be negative")
    if r <= 0:
        raise ValueError("distance r must be positive")
    return -G * m1 * m2 / r


def escape_velocity(mass: float, radius: float) -> float:
    """Escape velocity from a spherical body: v_esc = sqrt(2 * G * M / R)."""
    if mass <= 0 or radius <= 0:
        raise ValueError("mass and radius must be positive")
    return math.sqrt(2.0 * G * mass / radius)


def orbital_velocity(mass: float, radius: float) -> float:
    """Circular orbital velocity: v_orb = sqrt(G * M / R)."""
    if mass <= 0 or radius <= 0:
        raise ValueError("mass and radius must be positive")
    return math.sqrt(G * mass / radius)


def kinetic_energy(mass: float, vel: float) -> float:
    """Kinetic energy: E_k = 0.5 * m * v^2."""
    if mass < 0:
        raise ValueError("mass cannot be negative")
    return 0.5 * mass * (vel ** 2)


def work(f: float, displacement: float, angle_deg: float = 0.0) -> float:
    """Mechanical work: W = F * d * cos(theta)."""
    theta = math.radians(angle_deg)
    return f * displacement * math.cos(theta)


def power(w: float, t: float) -> float:
    """Mechanical power: P = W / t."""
    if t <= 0:
        raise ValueError("time must be strictly positive")
    return w / t


def momentum(mass: float, vel: float) -> float:
    """Linear momentum: p = m * v."""
    if mass < 0:
        raise ValueError("mass cannot be negative")
    return mass * vel


def impulse(f: float, t: float) -> float:
    """Impulse: J = F * delta_t."""
    return f * t


def elastic_collision_1d(m1: float, v1: float, m2: float, v2: float) -> Tuple[float, float]:
    """1D perfectly elastic collision between two masses."""
    total_m = m1 + m2
    if total_m <= 0:
        raise ValueError("Sum of masses must be positive")
    v1_prime = ((m1 - m2) * v1 + 2.0 * m2 * v2) / total_m
    v2_prime = (2.0 * m1 * v1 + (m2 - m1) * v2) / total_m
    return (v1_prime, v2_prime)


def inelastic_collision_1d(m1: float, v1: float, m2: float, v2: float) -> float:
    """1D completely inelastic collision (masses stick together)."""
    total_m = m1 + m2
    if total_m <= 0:
        raise ValueError("Sum of masses must be positive")
    return (m1 * v1 + m2 * v2) / total_m


def simple_pendulum_period(length: float, gravity: float = g) -> float:
    """Period of a simple pendulum: T = 2 * pi * sqrt(L / g)."""
    if length <= 0 or gravity <= 0:
        raise ValueError("length and gravity must be positive")
    return 2.0 * math.pi * math.sqrt(length / gravity)


def spring_mass_period(mass: float, spring_k: float) -> float:
    """Period of a spring-mass oscillator: T = 2 * pi * sqrt(m / k)."""
    if mass <= 0 or spring_k <= 0:
        raise ValueError("mass and spring_k must be positive")
    return 2.0 * math.pi * math.sqrt(mass / spring_k)


def hookes_law(k: float, x: float) -> float:
    """Restoring force of spring (Hooke's law): F = -k * x."""
    return -k * x


def spring_potential_energy(k: float, x: float) -> float:
    """Elastic potential energy: U = 0.5 * k * x^2."""
    if k < 0:
        raise ValueError("spring constant k cannot be negative")
    return 0.5 * k * (x ** 2)
