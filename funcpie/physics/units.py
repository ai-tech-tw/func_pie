"""Physical unit conversions across multiple measurement systems."""

from __future__ import annotations

import math
from typing import Dict, Optional

# Base units conversion tables (values relative to SI base units)
LENGTH_TO_METERS: Dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "um": 1e-6,
    "micrometer": 1e-6,
    "nm": 1e-9,
    "nanometer": 1e-9,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "yd": 0.9144,
    "yard": 0.9144,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
    "nautical_mile": 1852.0,
    "au": 1.495978707e11,
    "light_year": 9.4607304725808e15,
}

MASS_TO_KILOGRAMS: Dict[str, float] = {
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
    "mg": 1e-6,
    "milligram": 1e-6,
    "ug": 1e-9,
    "microgram": 1e-9,
    "lb": 0.45359237,
    "pound": 0.45359237,
    "pounds": 0.45359237,
    "oz": 0.028349523125,
    "ounce": 0.028349523125,
    "ton": 907.18474,  # US short ton
    "metric_ton": 1000.0,
    "tonne": 1000.0,
}

TIME_TO_SECONDS: Dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 0.001,
    "millisecond": 0.001,
    "us": 1e-6,
    "microsecond": 1e-6,
    "ns": 1e-9,
    "nanosecond": 1e-9,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "day": 86400.0,
    "days": 86400.0,
    "year": 31557600.0,  # Julian year (365.25 days)
    "years": 31557600.0,
}

PRESSURE_TO_PASCALS: Dict[str, float] = {
    "pa": 1.0,
    "pascal": 1.0,
    "kpa": 1000.0,
    "mpa": 1e6,
    "bar": 100000.0,
    "mbar": 100.0,
    "atm": 101325.0,
    "psi": 6894.757293168,
    "torr": 133.322368421,
    "mmhg": 133.322387415,
}

ENERGY_TO_JOULES: Dict[str, float] = {
    "j": 1.0,
    "joule": 1.0,
    "joules": 1.0,
    "kj": 1000.0,
    "cal": 4.184,
    "calorie": 4.184,
    "kcal": 4184.0,
    "ev": 1.602176634e-19,
    "kev": 1.602176634e-16,
    "mev": 1.602176634e-13,
    "wh": 3600.0,
    "kwh": 3.6e6,
    "btu": 1055.05585262,
}

SPEED_TO_MPS: Dict[str, float] = {
    "m/s": 1.0,
    "mps": 1.0,
    "km/h": 1.0 / 3.6,
    "kmh": 1.0 / 3.6,
    "mph": 0.44704,
    "knot": 0.514444444,
    "c": 299792458.0,
}

ANGLE_TO_RADIANS: Dict[str, float] = {
    "rad": 1.0,
    "radian": 1.0,
    "radians": 1.0,
    "deg": math.pi / 180.0,
    "degree": math.pi / 180.0,
    "degrees": math.pi / 180.0,
    "arcmin": math.pi / (180.0 * 60.0),
    "arcsec": math.pi / (180.0 * 3600.0),
    "rev": 2.0 * math.pi,
    "turn": 2.0 * math.pi,
}

CATEGORIES = {
    "length": LENGTH_TO_METERS,
    "mass": MASS_TO_KILOGRAMS,
    "time": TIME_TO_SECONDS,
    "pressure": PRESSURE_TO_PASCALS,
    "energy": ENERGY_TO_JOULES,
    "speed": SPEED_TO_MPS,
    "angle": ANGLE_TO_RADIANS,
}


def _normalize_unit(unit: str) -> str:
    return unit.strip().lower()


def convert_temperature(val: float, from_unit: str, to_unit: str) -> float:
    """Convert between temperature scales (C, F, K, R)."""
    f_u = _normalize_unit(from_unit)
    t_u = _normalize_unit(to_unit)

    # Convert to Kelvin
    if f_u in ("c", "celsius"):
        k = val + 273.15
    elif f_u in ("f", "fahrenheit"):
        k = (val - 32.0) * (5.0 / 9.0) + 273.15
    elif f_u in ("k", "kelvin"):
        k = val
    elif f_u in ("r", "rankine"):
        k = val * (5.0 / 9.0)
    else:
        raise ValueError(f"Unknown temperature unit: '{from_unit}'")

    # Convert from Kelvin to target
    if t_u in ("c", "celsius"):
        return k - 273.15
    elif t_u in ("f", "fahrenheit"):
        return (k - 273.15) * (9.0 / 5.0) + 32.0
    elif t_u in ("k", "kelvin"):
        return k
    elif t_u in ("r", "rankine"):
        return k * (9.0 / 5.0)
    else:
        raise ValueError(f"Unknown temperature unit: '{to_unit}'")


def convert(val: float, from_unit: str, to_unit: str, category: Optional[str] = None) -> float:
    """Universal unit converter supporting length, mass, time, pressure, energy, speed, angle, and temperature."""
    f_u = _normalize_unit(from_unit)
    t_u = _normalize_unit(to_unit)

    if f_u == t_u:
        return float(val)

    if category == "temperature" or f_u in ("c", "celsius", "f", "fahrenheit", "k", "kelvin", "r", "rankine"):
        if t_u in ("c", "celsius", "f", "fahrenheit", "k", "kelvin", "r", "rankine"):
            return convert_temperature(val, f_u, t_u)

    # Find matching category table
    tables = [CATEGORIES[category]] if category in CATEGORIES else CATEGORIES.values()
    for table in tables:
        if f_u in table and t_u in table:
            # Convert to base SI then to target
            base_si = val * table[f_u]
            return base_si / table[t_u]

    raise ValueError(f"Cannot convert from '{from_unit}' to '{to_unit}' (incompatible units or unknown)")


# Convenient direct helper functions
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return convert_temperature(celsius, "C", "F")


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return convert_temperature(fahrenheit, "F", "C")


def celsius_to_kelvin(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return convert_temperature(celsius, "C", "K")


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert Kelvin to Celsius."""
    return convert_temperature(kelvin, "K", "C")


def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * (math.pi / 180.0)


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return rad * (180.0 / math.pi)
