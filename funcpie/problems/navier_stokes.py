"""2D Incompressible Navier-Stokes Equations Numerical Solver."""

from __future__ import annotations

from typing import List, Tuple

from funcpie.backend import get_backend


def navier_stokes_2d_step(
    u: List[List[float]],
    v: List[List[float]],
    p: List[List[float]],
    dt: float,
    dx: float,
    dy: float,
    visc: float = 0.1,
    rho: float = 1.0,
    nit: int = 20,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """Single finite-difference step solving the 2D incompressible Navier-Stokes equations.

    Solves momentum convection-diffusion coupled with pressure Poisson equation.

    Args:
        u: 2D horizontal velocity field (ny x nx).
        v: 2D vertical velocity field (ny x nx).
        p: 2D pressure field (ny x nx).
        dt: Time increment.
        dx: Grid spacing in x direction.
        dy: Grid spacing in y direction.
        visc: Kinematic viscosity nu.
        rho: Fluid density.
        nit: Number of Jacobi pressure Poisson relaxation iterations.

    Returns:
        Tuple of updated (u_next, v_next, p_next).
    """
    return get_backend().navier_stokes_step(u, v, p, dt, dx, dy, visc, rho, nit)


def simulate_cavity_flow(
    nx: int = 21,
    ny: int = 21,
    steps: int = 50,
    dt: float = 0.001,
    visc: float = 0.1,
    rho: float = 1.0,
    nit: int = 20,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """Simulate 2D lid-driven cavity flow benchmark.

    A fluid cavity where the top lid moves horizontally with velocity u = 1.0,
    inducing vortex recirculations inside the cavity.

    Args:
        nx: Grid points along x.
        ny: Grid points along y.
        steps: Total time iterations.
        dt: Timestep.
        visc: Kinematic viscosity.
        rho: Fluid density.
        nit: Pressure Poisson iterations per step.

    Returns:
        Tuple of final fields (u, v, p).
    """
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    u = [[0.0 for _ in range(nx)] for _ in range(ny)]
    v = [[0.0 for _ in range(nx)] for _ in range(ny)]
    p = [[0.0 for _ in range(nx)] for _ in range(ny)]

    for _ in range(steps):
        u, v, p = navier_stokes_2d_step(u, v, p, dt, dx, dy, visc, rho, nit)

    return u, v, p
