"""Dynamic catalog and discovery registry for FuncPie tools."""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Callable

import funcpie
from funcpie import backend, core, math, physics
from funcpie.physics import constants


@dataclass(frozen=True)
class ItemMetadata:
    """Metadata describing a discoverable function, class, or constant."""

    name: str
    qualified_name: str
    category: str
    item_type: str  # "function" | "class" | "constant"
    signature: str
    summary: str
    docstring: str


class FuncPieRegistry:
    """Registry maintaining a searchable catalog of all FuncPie operations."""

    def __init__(self) -> None:
        self._items: dict[str, ItemMetadata] = {}
        self._categories: dict[str, list[str]] = {}
        self._build_catalog()

    def _register(
        self,
        name: str,
        qualname: str,
        category: str,
        item: object,
        item_type: str | None = None,
    ) -> None:
        if item_type is None:
            if inspect.isclass(item):
                item_type = "class"
            elif inspect.isfunction(item) or inspect.isbuiltin(item) or callable(item):
                item_type = "function"
            else:
                item_type = "constant"

        # Determine signature
        sig_str = ""
        if callable(item):
            try:
                sig_str = str(inspect.signature(item))
            except (ValueError, TypeError):
                sig_str = "(...)"
        else:
            sig_str = f"= {repr(item)}"

        doc = inspect.getdoc(item) or ""
        summary = doc.split("\n\n")[0].split("\n")[0].strip() if doc else ""

        meta = ItemMetadata(
            name=name,
            qualified_name=qualname,
            category=category,
            item_type=item_type,
            signature=sig_str,
            summary=summary,
            docstring=doc,
        )
        self._items[name] = meta
        self._items[qualname] = meta

        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)

    def _build_catalog(self) -> None:
        # 1. Math Basic
        from funcpie.math import basic
        self._register("Calculation", "funcpie.Calculation", "math.basic", basic.Calculation)
        for fn_name in ["add", "sub", "mul", "div", "power", "clamp", "sign", "is_close", "lerp", "remap"]:
            if hasattr(basic, fn_name):
                self._register(fn_name, f"funcpie.math.{fn_name}", "math.basic", getattr(basic, fn_name))

        # 2. Math Algebra
        from funcpie.math import algebra
        for attr in [
            "Polynomial", "factorial", "fibonacci", "gcd", "lcm", "is_prime",
            "prime_factors", "primes_sieve", "solve_linear", "solve_quadratic",
            "combinations_count", "permutations_count", "binomial_coefficient",
        ]:
            if hasattr(algebra, attr):
                self._register(attr, f"funcpie.math.{attr}", "math.algebra", getattr(algebra, attr))

        # 3. Math Calculus
        from funcpie.math import calculus
        for attr in [
            "derivative", "second_derivative", "integrate", "simpsons_rule",
            "monte_carlo_integrate", "gradient", "find_root_bisection",
            "find_root_newton", "find_root_secant",
        ]:
            if hasattr(calculus, attr):
                self._register(attr, f"funcpie.math.{attr}", "math.calculus", getattr(calculus, attr))

        # 4. Math Linear Algebra & Geometry
        from funcpie.math import geometry, linalg
        for attr in ["Vector2D", "Vector3D", "Matrix", "dot", "cross", "angle_between", "vector_projection"]:
            if hasattr(linalg, attr):
                self._register(attr, f"funcpie.math.{attr}", "math.linalg", getattr(linalg, attr))
        for attr in [
            "distance_2d", "distance_3d", "triangle_area", "circle_area", "circle_perimeter",
            "rectangle_area", "sphere_volume", "sphere_surface_area", "cylinder_volume", "cone_volume",
        ]:
            if hasattr(geometry, attr):
                self._register(attr, f"funcpie.math.{attr}", "math.geometry", getattr(geometry, attr))

        # 5. Math Statistics & Special
        from funcpie.math import special, statistics
        for attr in [
            "mean", "median", "mode", "variance", "standard_deviation",
            "covariance", "correlation", "percentile", "z_score",
        ]:
            if hasattr(statistics, attr):
                self._register(attr, f"funcpie.math.{attr}", "math.statistics", getattr(statistics, attr))
        for attr in ["gamma", "log_gamma", "beta", "erf", "erfc", "sigmoid", "softmax", "relu"]:
            if hasattr(special, attr):
                self._register(attr, f"funcpie.math.{attr}", "math.special", getattr(special, attr))

        # 6. Physics Mechanics
        from funcpie.physics import mechanics
        for attr in [
            "projectile_motion", "gravitational_force", "kinetic_energy",
            "gravitational_potential_energy", "momentum", "elastic_potential_energy",
            "work", "power_output", "friction_force", "centripetal_acceleration",
            "centripetal_force", "simple_pendulum_period", "spring_period",
            "escape_velocity", "orbital_velocity",
        ]:
            if hasattr(mechanics, attr):
                self._register(attr, f"funcpie.physics.mechanics.{attr}", "physics.mechanics", getattr(mechanics, attr))

        # 7. Physics Electromagnetism
        from funcpie.physics import electromagnetism
        for attr in [
            "coulomb_force", "electric_field_point_charge", "electric_potential_energy",
            "ohms_law_voltage", "ohms_law_current", "electrical_power",
            "resistors_series", "resistors_parallel", "capacitors_series",
            "capacitors_parallel", "lorentz_force", "magnetic_force_wire",
        ]:
            if hasattr(electromagnetism, attr):
                self._register(attr, f"funcpie.physics.electromagnetism.{attr}", "physics.electromagnetism", getattr(electromagnetism, attr))

        # 8. Physics Thermodynamics
        from funcpie.physics import thermodynamics
        for attr in [
            "celsius_to_kelvin", "kelvin_to_celsius", "fahrenheit_to_celsius",
            "celsius_to_fahrenheit", "ideal_gas_pressure", "ideal_gas_volume",
            "ideal_gas_temperature", "heat_energy", "carnot_efficiency",
            "entropy_change_isothermal", "work_isothermal", "heat_conduction",
        ]:
            if hasattr(thermodynamics, attr):
                self._register(attr, f"funcpie.physics.thermodynamics.{attr}", "physics.thermodynamics", getattr(thermodynamics, attr))

        # 9. Physics Waves & Optics
        from funcpie.physics import waves
        for attr in [
            "wave_speed", "wave_frequency", "wave_wavelength", "doppler_effect",
            "snells_law", "sound_intensity_level", "standing_wave_wavelength",
        ]:
            if hasattr(waves, attr):
                self._register(attr, f"funcpie.physics.waves.{attr}", "physics.waves", getattr(waves, attr))

        # 10. Physics Modern
        from funcpie.physics import modern
        for attr in [
            "mass_energy_equivalence", "photoelectric_kinetic_energy", "de_broglie_wavelength",
            "lorentz_factor", "time_dilation", "length_contraction", "relativistic_momentum",
            "compton_wavelength_shift", "heisenberg_position_momentum", "radioactive_decay",
        ]:
            if hasattr(modern, attr):
                self._register(attr, f"funcpie.physics.modern.{attr}", "physics.modern", getattr(modern, attr))

        # 11. Physics Units
        from funcpie.physics import units
        for attr in [
            "convert_length", "convert_mass", "convert_force", "convert_energy",
            "convert_pressure", "convert_temperature", "convert_units",
        ]:
            if hasattr(units, attr):
                self._register(attr, f"funcpie.physics.units.{attr}", "physics.units", getattr(units, attr))

        # 12. Physics Constants
        for attr in dir(constants):
            if attr.isupper() and not attr.startswith("_"):
                val = getattr(constants, attr)
                self._register(attr, f"funcpie.physics.constants.{attr}", "physics.constants", val, item_type="constant")

        # 13. Core Pipeline
        from funcpie.core import pipeline
        for attr in ["pipe", "compose", "curry", "Pipeable", "pipeable", "tap"]:
            if hasattr(pipeline, attr):
                self._register(attr, f"funcpie.core.{attr}", "core.pipeline", getattr(pipeline, attr))

        # 14. Backend
        for attr in ["get_backend", "set_backend", "is_hip_available", "get_device_properties"]:
            if hasattr(backend, attr):
                self._register(attr, f"funcpie.backend.{attr}", "backend", getattr(backend, attr))

    def search(self, query: str, category: str | None = None) -> list[dict[str, object]]:
        """Search registered items by name, docstring, or category."""
        q = query.lower().strip()
        matched: list[ItemMetadata] = []
        seen_names: set[str] = set()

        for meta in self._items.values():
            if category and meta.category != category:
                continue
            if meta.name in seen_names:
                continue

            if (
                q in meta.name.lower()
                or q in meta.qualified_name.lower()
                or q in meta.summary.lower()
                or q in meta.category.lower()
            ):
                matched.append(meta)
                seen_names.add(meta.name)

        return [
            {
                "name": m.name,
                "qualified_name": m.qualified_name,
                "category": m.category,
                "item_type": m.item_type,
                "signature": m.signature,
                "summary": m.summary,
            }
            for m in matched
        ]

    def list_categories(self, category: str | None = None) -> dict[str, list[str]]:
        """List registered items grouped by category, or items for a specific category."""
        if category:
            return {category: sorted(self._categories.get(category, []))}
        return {cat: sorted(names) for cat, names in sorted(self._categories.items())}

    def describe(self, name: str) -> dict[str, object] | None:
        """Retrieve complete documentation and signature for a specific item."""
        meta = self._items.get(name)
        if not meta:
            # Try case-insensitive lookup
            for item_name, item_meta in self._items.items():
                if item_name.lower() == name.lower():
                    meta = item_meta
                    break

        if not meta:
            return None

        return asdict(meta)


# Global singleton instance
_GLOBAL_REGISTRY: FuncPieRegistry | None = None


def get_registry() -> FuncPieRegistry:
    """Retrieve or create the global singleton FuncPieRegistry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = FuncPieRegistry()
    return _GLOBAL_REGISTRY
