"""Modern physics: special relativity, quantum mechanics, and nuclear physics."""

from __future__ import annotations

import math
from typing import Optional

from funcpie.physics.constants import c, h


def lorentz_factor(velocity: float, c_speed: float = c) -> float:
    """Lorentz factor gamma: gamma = 1 / sqrt(1 - (v/c)^2)."""
    if abs(velocity) >= c_speed:
        raise ValueError("Velocity cannot equal or exceed the speed of light")
    beta = velocity / c_speed
    return 1.0 / math.sqrt(1.0 - beta * beta)


def time_dilation(proper_time: float, velocity: float, c_speed: float = c) -> float:
    """Relativistic time dilation: delta_t = gamma * delta_t0."""
    gamma = lorentz_factor(velocity, c_speed)
    return gamma * proper_time


def length_contraction(proper_length: float, velocity: float, c_speed: float = c) -> float:
    """Relativistic length contraction: L = L0 / gamma."""
    gamma = lorentz_factor(velocity, c_speed)
    return proper_length / gamma


def rest_energy(mass: float, c_speed: float = c) -> float:
    """Rest mass energy: E_0 = m * c^2 (in Joules)."""
    if mass < 0:
        raise ValueError("Mass cannot be negative")
    return mass * (c_speed ** 2)


def relativistic_total_energy(rest_mass: float, velocity: float, c_speed: float = c) -> float:
    """Total relativistic energy: E = gamma * m * c^2."""
    gamma = lorentz_factor(velocity, c_speed)
    return gamma * rest_energy(rest_mass, c_speed)


def relativistic_momentum(rest_mass: float, velocity: float, c_speed: float = c) -> float:
    """Relativistic linear momentum: p = gamma * m * v."""
    gamma = lorentz_factor(velocity, c_speed)
    return gamma * rest_mass * velocity


def photon_energy_from_freq(frequency: float, h_const: float = h) -> float:
    """Photon energy from frequency: E = h * f (in Joules)."""
    if frequency < 0:
        raise ValueError("Frequency cannot be negative")
    return h_const * frequency


def photon_energy_from_wavelength(wavelength: float, h_const: float = h, c_speed: float = c) -> float:
    """Photon energy from wavelength: E = h * c / lambda (in Joules)."""
    if wavelength <= 0:
        raise ValueError("Wavelength must be strictly positive")
    return (h_const * c_speed) / wavelength


def de_broglie_wavelength(momentum: float, h_const: float = h) -> float:
    """Matter wave de Broglie wavelength: lambda = h / p."""
    if momentum == 0:
        raise ZeroDivisionError("Momentum cannot be zero")
    return h_const / abs(momentum)


def photoelectric_max_kinetic_energy(photon_energy: float, work_function: float) -> float:
    """Maximum kinetic energy of photoelectrons: K_max = max(0, E_photon - phi)."""
    if photon_energy < 0 or work_function < 0:
        raise ValueError("Energies cannot be negative")
    return max(0.0, photon_energy - work_function)


def radioactive_decay_remaining(initial_quantity: float, half_life: float, elapsed_time: float) -> float:
    """Remaining quantity after radioactive decay: N(t) = N0 * (1/2)^(t / t_half)."""
    if half_life <= 0:
        raise ValueError("Half-life must be strictly positive")
    if initial_quantity < 0 or elapsed_time < 0:
        raise ValueError("Initial quantity and elapsed time cannot be negative")
    return initial_quantity * (0.5 ** (elapsed_time / half_life))


def decay_constant(half_life: float) -> float:
    """Radioactive decay constant lambda = ln(2) / t_half."""
    if half_life <= 0:
        raise ValueError("Half-life must be strictly positive")
    return math.log(2.0) / half_life


def activity(num_nuclei: float, half_life: float) -> float:
    """Decay activity A = lambda * N (in Becquerels)."""
    if num_nuclei < 0:
        raise ValueError("Number of nuclei cannot be negative")
    lam = decay_constant(half_life)
    return lam * num_nuclei
