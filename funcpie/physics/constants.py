"""Standard CODATA physical constants."""

from __future__ import annotations

import math
from typing import Dict, NamedTuple


class PhysicalConstant(NamedTuple):
    symbol: str
    name: str
    value: float
    unit: str

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"{self.symbol} = {self.value:g} {self.unit} ({self.name})"


# Standard CODATA Values
SPEED_OF_LIGHT = PhysicalConstant("c", "Speed of light in vacuum", 299_792_458.0, "m/s")
GRAVITATIONAL_CONSTANT = PhysicalConstant("G", "Newtonian constant of gravitation", 6.67430e-11, "m^3/(kg s^2)")
PLANCK_CONSTANT = PhysicalConstant("h", "Planck constant", 6.62607015e-34, "J s")
REDUCED_PLANCK_CONSTANT = PhysicalConstant("hbar", "Reduced Planck constant", 6.62607015e-34 / (2.0 * math.pi), "J s")
ELEMENTARY_CHARGE = PhysicalConstant("e", "Elementary charge", 1.602176634e-19, "C")
BOLTZMANN_CONSTANT = PhysicalConstant("k_B", "Boltzmann constant", 1.380649e-23, "J/K")
AVOGADRO_CONSTANT = PhysicalConstant("N_A", "Avogadro constant", 6.02214076e23, "mol^-1")
GAS_CONSTANT = PhysicalConstant("R", "Molar gas constant", 8.314462618, "J/(mol K)")
VACUUM_PERMITTIVITY = PhysicalConstant("epsilon_0", "Vacuum electric permittivity", 8.8541878128e-12, "F/m")
VACUUM_PERMEABILITY = PhysicalConstant("mu_0", "Vacuum magnetic permeability", 1.25663706212e-6, "N/A^2")
STANDARD_GRAVITY = PhysicalConstant("g_0", "Standard acceleration of gravity", 9.80665, "m/s^2")
ELECTRON_MASS = PhysicalConstant("m_e", "Electron mass", 9.1093837015e-31, "kg")
PROTON_MASS = PhysicalConstant("m_p", "Proton mass", 1.67262192369e-27, "kg")
NEUTRON_MASS = PhysicalConstant("m_n", "Neutron mass", 1.67492749804e-27, "kg")
STEFAN_BOLTZMANN_CONSTANT = PhysicalConstant("sigma", "Stefan-Boltzmann constant", 5.670374419e-8, "W/(m^2 K^4)")
ATOMIC_MASS_UNIT = PhysicalConstant("u", "Unified atomic mass unit", 1.66053906660e-27, "kg")
WIEN_DISPLACEMENT_CONSTANT = PhysicalConstant("b", "Wien displacement law constant", 2.897771955e-3, "m K")

# Convenient short aliases
c = SPEED_OF_LIGHT.value
G = GRAVITATIONAL_CONSTANT.value
h = PLANCK_CONSTANT.value
hbar = REDUCED_PLANCK_CONSTANT.value
e = ELEMENTARY_CHARGE.value
k_B = BOLTZMANN_CONSTANT.value
N_A = AVOGADRO_CONSTANT.value
R = GAS_CONSTANT.value
epsilon_0 = VACUUM_PERMITTIVITY.value
mu_0 = VACUUM_PERMEABILITY.value
g = STANDARD_GRAVITY.value
m_e = ELECTRON_MASS.value
m_p = PROTON_MASS.value
m_n = NEUTRON_MASS.value
sigma = STEFAN_BOLTZMANN_CONSTANT.value
u = ATOMIC_MASS_UNIT.value
wien_b = WIEN_DISPLACEMENT_CONSTANT.value

ALL_CONSTANTS: Dict[str, PhysicalConstant] = {
    "c": SPEED_OF_LIGHT,
    "G": GRAVITATIONAL_CONSTANT,
    "h": PLANCK_CONSTANT,
    "hbar": REDUCED_PLANCK_CONSTANT,
    "e": ELEMENTARY_CHARGE,
    "k_B": BOLTZMANN_CONSTANT,
    "N_A": AVOGADRO_CONSTANT,
    "R": GAS_CONSTANT,
    "epsilon_0": VACUUM_PERMITTIVITY,
    "mu_0": VACUUM_PERMEABILITY,
    "g": STANDARD_GRAVITY,
    "m_e": ELECTRON_MASS,
    "m_p": PROTON_MASS,
    "m_n": NEUTRON_MASS,
    "sigma": STEFAN_BOLTZMANN_CONSTANT,
    "u": ATOMIC_MASS_UNIT,
    "b": WIEN_DISPLACEMENT_CONSTANT,
}
