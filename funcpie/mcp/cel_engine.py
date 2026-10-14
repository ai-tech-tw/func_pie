"""Google Common Expression Language (CEL) execution engine for FuncPie MCP.

Provides a non-Turing complete, memory-safe, and injection-immune expression
evaluation environment for scientific mathematics and physics computations.
"""

from __future__ import annotations

import time
from typing import Callable

try:
    import celpy
    from celpy import celtypes

    HAS_CEL = True
except ImportError:
    celpy = None  # pyright: ignore[reportConstantRedefinition]
    celtypes = None  # pyright: ignore[reportConstantRedefinition]
    HAS_CEL = False

from funcpie import math, physics
from funcpie.physics import constants


def _wrap_return_value(val: object) -> object:
    """Wrap return values so that nested attributes can be accessed via dot or bracket notation in CEL."""
    if val is None:
        return None
    if isinstance(val, (int, float, bool, str)):
        return val
    asdict_fn = getattr(val, "_asdict", None)
    if callable(asdict_fn):
        res_dict = asdict_fn()
        if isinstance(res_dict, dict):
            d = {str(k): _wrap_return_value(v) for k, v in res_dict.items()}
            if celtypes is not None:
                return celtypes.MessageType(d)  # pyright: ignore[reportArgumentType]
            return d
    import dataclasses
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        d = {str(k): _wrap_return_value(v) for k, v in dataclasses.asdict(val).items()}
        if celtypes is not None:
            return celtypes.MessageType(d)  # pyright: ignore[reportArgumentType]
        return d
    if isinstance(val, dict):
        d = {str(k): _wrap_return_value(v) for k, v in val.items()}
        if celtypes is not None:
            return celtypes.MessageType(d)  # pyright: ignore[reportArgumentType]
        return d
    if isinstance(val, (list, tuple)):
        return [_wrap_return_value(x) for x in val]
    return str(val)


def _make_safe_caller(fn: Callable[..., object]) -> Callable[..., object]:
    """Wrap a FuncPie function into a CEL-safe callable with automatic type unwrapping and wrapping."""

    def _caller(*args: object) -> object:
        # Unpack arguments to python primitives
        clean_args: list[object] = []
        for a in args:
            if celtypes is not None and isinstance(a, celtypes.DoubleType):
                clean_args.append(float(a))
            elif celtypes is not None and isinstance(a, celtypes.IntType):
                clean_args.append(int(a))
            elif isinstance(a, (int, float, str, bool, list, dict)):
                clean_args.append(a)
            else:
                clean_args.append(a)
        res = fn(*clean_args)
        return _wrap_return_value(res)

    return _caller


class CELEngine:
    """Security-hardened Common Expression Language evaluation engine for FuncPie."""

    def __init__(self) -> None:
        if not HAS_CEL or celpy is None:
            raise ImportError(
                "cel-python is required to use the CEL execution engine. "
                "Please install it with: pip install 'funcpie[mcp]' or pip install cel-python"
            )
        self.env = celpy.Environment()
        self.functions: dict[str, Callable[..., object]] = {}
        self.constants: dict[str, object] = {}
        self._init_bindings()

    def _init_bindings(self) -> None:
        # 1. Math Basic & Arithmetic
        from funcpie.math import basic
        for name in ["add", "sub", "mul", "div", "power", "clamp", "sign", "is_close", "lerp", "remap"]:
            if hasattr(basic, name):
                self.functions[name] = _make_safe_caller(getattr(basic, name))

        # 2. Math Algebra
        from funcpie.math import algebra
        for name in [
            "factorial", "fibonacci", "gcd", "lcm", "is_prime",
            "prime_factors", "combinations_count", "permutations_count", "binomial_coefficient",
        ]:
            if hasattr(algebra, name):
                self.functions[name] = _make_safe_caller(getattr(algebra, name))

        # 3. Math Geometry
        from funcpie.math import geometry
        for name in [
            "distance_2d", "distance_3d", "triangle_area", "circle_area", "circle_perimeter",
            "rectangle_area", "sphere_volume", "sphere_surface_area", "cylinder_volume", "cone_volume",
        ]:
            if hasattr(geometry, name):
                self.functions[name] = _make_safe_caller(getattr(geometry, name))

        # 4. Math Statistics & Special
        from funcpie.math import special, statistics
        for name in ["mean", "median", "variance", "standard_deviation"]:
            if hasattr(statistics, name):
                self.functions[name] = _make_safe_caller(getattr(statistics, name))
        for name in ["gamma", "erf", "sigmoid", "relu"]:
            if hasattr(special, name):
                self.functions[name] = _make_safe_caller(getattr(special, name))

        # 5. Physics Mechanics
        from funcpie.physics import mechanics
        for name in [
            "projectile_motion", "gravitational_force", "kinetic_energy",
            "gravitational_potential_energy", "momentum", "elastic_potential_energy",
            "work", "power_output", "friction_force", "centripetal_acceleration",
            "centripetal_force", "simple_pendulum_period", "spring_period",
            "escape_velocity", "orbital_velocity",
        ]:
            if hasattr(mechanics, name):
                self.functions[name] = _make_safe_caller(getattr(mechanics, name))

        # 6. Physics Electromagnetism
        from funcpie.physics import electromagnetism
        for name in [
            "coulomb_force", "electric_field_point_charge", "electric_potential_energy",
            "ohms_law_voltage", "ohms_law_current", "electrical_power",
            "lorentz_force", "magnetic_force_wire",
        ]:
            if hasattr(electromagnetism, name):
                self.functions[name] = _make_safe_caller(getattr(electromagnetism, name))

        # 7. Physics Thermodynamics
        from funcpie.physics import thermodynamics
        for name in [
            "celsius_to_kelvin", "kelvin_to_celsius", "fahrenheit_to_celsius",
            "celsius_to_fahrenheit", "ideal_gas_pressure", "ideal_gas_volume",
            "ideal_gas_temperature", "heat_energy", "carnot_efficiency",
            "heat_conduction",
        ]:
            if hasattr(thermodynamics, name):
                self.functions[name] = _make_safe_caller(getattr(thermodynamics, name))

        # 8. Physics Waves & Optics
        from funcpie.physics import waves
        for name in [
            "wave_speed", "wave_frequency", "wave_wavelength", "doppler_effect",
            "snells_law", "sound_intensity_level",
        ]:
            if hasattr(waves, name):
                self.functions[name] = _make_safe_caller(getattr(waves, name))

        # 9. Physics Modern
        from funcpie.physics import modern
        for name in [
            "mass_energy_equivalence", "photoelectric_kinetic_energy", "de_broglie_wavelength",
            "lorentz_factor", "time_dilation", "length_contraction", "relativistic_momentum",
        ]:
            if hasattr(modern, name):
                self.functions[name] = _make_safe_caller(getattr(modern, name))

        # 10. Physics Units
        from funcpie.physics import units
        if hasattr(units, "convert_units"):
            self.functions["convert_units"] = _make_safe_caller(getattr(units, "convert_units"))

        # 11. Physics Constants
        for attr in dir(constants):
            if attr.isupper() and not attr.startswith("_"):
                val = getattr(constants, attr)
                if isinstance(val, constants.PhysicalConstant):
                    self.constants[attr] = float(val.value)
                else:
                    self.constants[attr] = val

    def evaluate(
        self,
        expression: str,
        variables: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Compile and evaluate a CEL expression safely.

        Args:
            expression: CEL expression string (e.g., 'add(10, 20) * 2', 'projectile_motion(20.0, 45.0).max_height').
            variables: Optional variable dictionary to expose to the expression.

        Returns:
            Dictionary containing success, result, result_type, engine, execution_time_ms.
        """
        start_time = time.perf_counter()
        expr_clean = expression.strip()

        if not expr_clean:
            return {
                "success": True,
                "result": None,
                "result_type": "NoneType",
                "engine": "cel",
                "execution_time_ms": 0.0,
            }

        try:
            # Compile AST
            ast = self.env.compile(expr_clean)

            # Build execution program with bound functions
            prg = self.env.program(ast, functions=self.functions)  # pyright: ignore[reportArgumentType]

            # Build context (Constants + Variables)
            context: dict[str, object] = dict(self.constants)
            if variables:
                for k, v in variables.items():
                    context[k] = _wrap_return_value(v)

            # Evaluate
            raw_result = prg.evaluate(context)  # pyright: ignore[reportArgumentType]
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Convert CEL types to standard Python primitives
            final_result: object
            if celtypes is not None and isinstance(raw_result, celtypes.MessageType):
                final_result = dict(raw_result)
            elif celtypes is not None and isinstance(raw_result, celtypes.DoubleType):
                final_result = float(raw_result)
            elif celtypes is not None and isinstance(raw_result, celtypes.IntType):
                final_result = int(raw_result)
            else:
                final_result = raw_result

            return {
                "success": True,
                "result": final_result,
                "result_type": type(final_result).__name__,
                "engine": "cel",
                "execution_time_ms": round(elapsed_ms, 3),
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "success": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "engine": "cel",
                "execution_time_ms": round(elapsed_ms, 3),
            }


# Singleton CEL Engine
_GLOBAL_CEL_ENGINE: CELEngine | None = None


def get_cel_engine() -> CELEngine:
    """Retrieve or create the global singleton CELEngine."""
    global _GLOBAL_CEL_ENGINE
    if _GLOBAL_CEL_ENGINE is None:
        _GLOBAL_CEL_ENGINE = CELEngine()
    return _GLOBAL_CEL_ENGINE


def evaluate_cel(
    expression: str,
    variables: dict[str, object] | None = None,
) -> dict[str, object]:
    """Helper function to evaluate a CEL expression using the singleton engine."""
    return get_cel_engine().evaluate(expression=expression, variables=variables)
