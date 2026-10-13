"""Wave physics, acoustics, and geometric optics."""

from __future__ import annotations

import math
from typing import Optional


def wave_speed(frequency: float, wavelength: float) -> float:
    """Wave propagation speed: v = f * lambda."""
    if frequency < 0 or wavelength < 0:
        raise ValueError("Frequency and wavelength must be non-negative")
    return frequency * wavelength


def wave_frequency(speed: float, wavelength: float) -> float:
    """Wave frequency: f = v / lambda."""
    if wavelength <= 0:
        raise ValueError("Wavelength must be strictly positive")
    return speed / wavelength


def wave_length(speed: float, frequency: float) -> float:
    """Wavelength: lambda = v / f."""
    if frequency <= 0:
        raise ValueError("Frequency must be strictly positive")
    return speed / frequency


def sound_speed_air(temperature_celsius: float = 20.0) -> float:
    """Speed of sound in dry air at given Celsius temperature: v ~ 331.3 * sqrt(1 + T / 273.15)."""
    t_kelvin = temperature_celsius + 273.15
    if t_kelvin <= 0:
        raise ValueError("Absolute temperature must be positive")
    return 331.3 * math.sqrt(t_kelvin / 273.15)


def doppler_effect(
    source_frequency: float,
    v_wave: float,
    v_observer: float,
    v_source: float,
    observer_towards: bool = True,
    source_towards: bool = True,
) -> float:
    """Observed frequency due to Doppler effect for sound/mechanical waves:

    f_obs = f_src * (v_wave +- v_obs) / (v_wave -+ v_src)
    """
    if source_frequency <= 0 or v_wave <= 0:
        raise ValueError("Frequency and wave speed must be positive")
    numerator = v_wave + (v_observer if observer_towards else -v_observer)
    denominator = v_wave + (-v_source if source_towards else v_source)
    if denominator <= 0:
        raise ValueError("Source speed cannot exceed wave speed in towards direction (shockwave)")
    return source_frequency * (numerator / denominator)


def snells_law_angle(n1: float, theta1_deg: float, n2: float) -> float:
    """Calculate angle of refraction theta2 in degrees according to Snell's law: n1*sin(th1) = n2*sin(th2).

    Raises:
        ValueError: If total internal reflection occurs (sin(theta2) > 1).
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Refractive indices must be positive")
    sin_theta1 = math.sin(math.radians(theta1_deg))
    sin_theta2 = (n1 / n2) * sin_theta1
    if abs(sin_theta2) > 1.0:
        raise ValueError("Total internal reflection occurred; no real refracted angle exists")
    return math.degrees(math.asin(sin_theta2))


def critical_angle(n1: float, n2: float) -> float:
    """Critical angle for total internal reflection in degrees (n1 > n2)."""
    if n1 <= n2:
        raise ValueError("Total internal reflection only occurs when light travels from higher to lower index (n1 > n2)")
    return math.degrees(math.asin(n2 / n1))


def thin_lens_focal_length(d_object: float, d_image: float) -> float:
    """Focal length of thin lens: 1/f = 1/d_o + 1/d_i."""
    if d_object == 0 or d_image == 0:
        raise ValueError("Distances cannot be zero")
    return 1.0 / (1.0 / d_object + 1.0 / d_image)


def thin_lens_image_distance(d_object: float, focal_length: float) -> float:
    """Image distance: 1/d_i = 1/f - 1/d_o."""
    if d_object == focal_length:
        raise ValueError("Object at focal point; image is at infinity")
    return 1.0 / (1.0 / focal_length - 1.0 / d_object)


def thin_lens_magnification(d_object: float, d_image: float) -> float:
    """Linear magnification: m = - d_i / d_o."""
    if d_object == 0:
        raise ValueError("Object distance cannot be zero")
    return -d_image / d_object


def sound_level_decibels(intensity: float, reference_intensity: float = 1e-12) -> float:
    """Sound intensity level in decibels (dB): beta = 10 * log10(I / I_0)."""
    if intensity <= 0 or reference_intensity <= 0:
        raise ValueError("Intensities must be strictly positive")
    return 10.0 * math.log10(intensity / reference_intensity)
