"""Chaotic Three-Body Gravitational Problem Solver and Orbit Integrator."""

from __future__ import annotations

from typing import List, Tuple

from funcpie.backend import get_backend


def three_body_step(
    masses: Tuple[float, float, float],
    positions: List[Tuple[float, float, float]],
    velocities: List[Tuple[float, float, float]],
    dt: float,
    g: float = 1.0,
    softening: float = 1e-3,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """Single RK4 integration step for three bodies under gravitational interaction.

    Args:
        masses: Tuple of 3 body masses (m1, m2, m3).
        positions: List of 3 coordinates [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)].
        velocities: List of 3 velocity vectors [(vx1,vy1,vz1), (vx2,vy2,vz2), (vx3,vy3,vz3)].
        dt: Integration time step.
        g: Gravitational constant.
        softening: Numerical softening parameter to avoid division by zero.

    Returns:
        Tuple of (new_positions, new_velocities).
    """
    return get_backend().three_body_step(masses, positions, velocities, dt, g, softening)


def three_body_simulation(
    masses: Tuple[float, float, float],
    positions: List[Tuple[float, float, float]],
    velocities: List[Tuple[float, float, float]],
    dt: float,
    steps: int,
    g: float = 1.0,
    softening: float = 1e-3,
) -> List[List[Tuple[float, float, float]]]:
    """Simulate trajectories for 3 bodies over multiple time steps.

    Args:
        masses: Tuple of masses (m1, m2, m3).
        positions: Initial coordinates of the 3 bodies.
        velocities: Initial velocities of the 3 bodies.
        dt: Timestep.
        steps: Number of integration steps.
        g: Gravitational constant.
        softening: Softening length.

    Returns:
        List of positions at each time step (length == steps + 1).
    """
    history: List[List[Tuple[float, float, float]]] = [list(positions)]
    curr_pos = list(positions)
    curr_vel = list(velocities)

    for _ in range(steps):
        curr_pos, curr_vel = three_body_step(masses, curr_pos, curr_vel, dt, g, softening)
        history.append(list(curr_pos))

    return history


def figure_eight_initial_conditions() -> Tuple[
    Tuple[float, float, float],
    List[Tuple[float, float, float]],
    List[Tuple[float, float, float]],
]:
    """Return initial conditions for the famous stable Chenciner-Montgomery Figure-8 three-body orbit.

    Returns:
        Tuple of (masses, positions, velocities).
    """
    masses = (1.0, 1.0, 1.0)
    # Positions
    x1, y1 = 0.97000436, -0.24308753
    positions = [
        (x1, y1, 0.0),
        (-x1, -y1, 0.0),
        (0.0, 0.0, 0.0),
    ]
    # Velocities
    vx3, vy3 = -2.0 * -0.93240737 / -2.0, -2.0 * -0.86473146 / -2.0  # Normalized
    vx1, vy1 = 0.46620531, 0.43236573
    velocities = [
        (vx1, vy1, 0.0),
        (vx1, vy1, 0.0),
        (-2.0 * vx1, -2.0 * vy1, 0.0),
    ]
    return masses, positions, velocities
