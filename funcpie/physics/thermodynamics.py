"""Thermodynamics and statistical heat physics."""

from __future__ import annotations

from funcpie.physics.constants import R, sigma, wien_b


def ideal_gas_pressure(n: float, temperature_k: float, volume_m3: float, gas_constant: float = R) -> float:
    """Ideal gas pressure: P = n * R * T / V (in Pascals)."""
    if volume_m3 <= 0:
        raise ValueError("Volume must be strictly positive")
    if temperature_k < 0:
        raise ValueError("Temperature in Kelvin cannot be negative")
    return (n * gas_constant * temperature_k) / volume_m3


def ideal_gas_volume(n: float, temperature_k: float, pressure_pa: float, gas_constant: float = R) -> float:
    """Ideal gas volume: V = n * R * T / P (in m^3)."""
    if pressure_pa <= 0:
        raise ValueError("Pressure must be strictly positive")
    if temperature_k < 0:
        raise ValueError("Temperature in Kelvin cannot be negative")
    return (n * gas_constant * temperature_k) / pressure_pa


def ideal_gas_temperature(pressure_pa: float, volume_m3: float, n: float, gas_constant: float = R) -> float:
    """Ideal gas absolute temperature: T = P * V / (n * R) (in Kelvin)."""
    if n <= 0:
        raise ValueError("Moles n must be positive")
    return (pressure_pa * volume_m3) / (n * gas_constant)


def ideal_gas_moles(pressure_pa: float, volume_m3: float, temperature_k: float, gas_constant: float = R) -> float:
    """Number of moles: n = P * V / (R * T)."""
    if temperature_k <= 0:
        raise ValueError("Temperature in Kelvin must be strictly positive")
    return (pressure_pa * volume_m3) / (gas_constant * temperature_k)


def heat_energy(mass: float, specific_heat: float, delta_temp: float) -> float:
    """Heat energy transferred: Q = m * c * delta_T (in Joules)."""
    if mass < 0 or specific_heat < 0:
        raise ValueError("Mass and specific heat capacity must be non-negative")
    return mass * specific_heat * delta_temp


def latent_heat(mass: float, latent_heat_coeff: float) -> float:
    """Heat required for phase change: Q = m * L (in Joules)."""
    if mass < 0 or latent_heat_coeff < 0:
        raise ValueError("Mass and latent heat must be non-negative")
    return mass * latent_heat_coeff


def thermal_expansion_linear(initial_length: float, alpha: float, delta_temp: float) -> float:
    """Linear thermal expansion change in length: delta_L = L0 * alpha * delta_T."""
    return initial_length * alpha * delta_temp


def thermal_expansion_volume(initial_volume: float, beta: float, delta_temp: float) -> float:
    """Volumetric thermal expansion change in volume: delta_V = V0 * beta * delta_T."""
    return initial_volume * beta * delta_temp


def carnot_efficiency(t_cold_k: float, t_hot_k: float) -> float:
    """Maximum theoretical thermodynamic efficiency of a heat engine: eta = 1 - T_c / T_h."""
    if t_cold_k < 0 or t_hot_k <= 0:
        raise ValueError("Absolute temperatures must be positive")
    if t_cold_k > t_hot_k:
        raise ValueError("Cold reservoir temperature cannot exceed hot reservoir temperature")
    return 1.0 - (t_cold_k / t_hot_k)


def stefan_boltzmann_radiation(area: float, emissivity: float, temperature_k: float) -> float:
    """Radiated power from a blackbody or greybody: P = epsilon * sigma * A * T^4 (in Watts)."""
    if area < 0:
        raise ValueError("Area must be non-negative")
    if not (0.0 <= emissivity <= 1.0):
        raise ValueError("Emissivity must be in range [0, 1]")
    if temperature_k < 0:
        raise ValueError("Temperature in Kelvin cannot be negative")
    return emissivity * sigma * area * (temperature_k ** 4)


def wien_displacement_peak_wavelength(temperature_k: float) -> float:
    """Peak emission wavelength for blackbody radiation: lambda_max = b / T (in meters)."""
    if temperature_k <= 0:
        raise ValueError("Temperature in Kelvin must be positive")
    return wien_b / temperature_k


def heat_conduction_rate(thermal_conductivity: float, area: float, delta_temp: float, thickness: float) -> float:
    """Rate of heat conduction: Q/t = k * A * delta_T / d (in Watts)."""
    if thickness <= 0:
        raise ValueError("Thickness must be strictly positive")
    if area < 0 or thermal_conductivity < 0:
        raise ValueError("Conductivity and area must be non-negative")
    return thermal_conductivity * area * delta_temp / thickness
