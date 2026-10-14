"""FastMCP tools providing dynamic discovery and code execution for FuncPie."""

from __future__ import annotations

import ast
import dataclasses
import io
import time
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import funcpie
from funcpie import backend, core, math, physics
from funcpie.mcp.registry import get_registry
from funcpie.physics import constants

if TYPE_CHECKING:
    from fastmcp import FastMCP


def _format_result(val: object) -> object:
    """Format execution output safely for JSON transmission."""
    if val is None:
        return None
    if isinstance(val, (int, float, bool, str)):
        return val
    if isinstance(val, (list, tuple)):
        return [_format_result(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _format_result(v) for k, v in val.items()}
    # Handle objects with _asdict method (e.g. namedtuple)
    asdict_fn = getattr(val, "_asdict", None)
    if callable(asdict_fn):
        return _format_result(asdict_fn())
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        return _format_result(dataclasses.asdict(val))
    # Fallback to string representation
    return str(val)


def execute_funcpie(code: str, use_gpu: bool = False) -> dict[str, object]:
    """Execute arbitrary FuncPie Python code in a pre-configured scientific environment."""
    registry = get_registry()
    start_time = time.perf_counter()

    # Configure backend
    orig_backend_name = getattr(backend.get_backend(), "name", "cpu")
    if use_gpu:
        try:
            backend.set_backend("hip")
        except Exception:
            pass  # Fallback to current backend if HIP is unavailable

    # Prepare standard scope with FuncPie library components
    scope: dict[str, object] = {
        # Core namespace
        "fp": funcpie,
        "funcpie": funcpie,
        "math": math,
        "physics": physics,
        "core": core,
        "backend": backend,
        "constants": constants,
        # Functional tools
        "pipe": core.pipe,
        "compose": core.compose,
        "curry": core.curry,
        # Arithmetic & Math classes
        "Calculation": math.Calculation,
        "Vector2D": math.Vector2D,
        "Vector3D": math.Vector3D,
        "Matrix": math.Matrix,
        "Polynomial": math.Polynomial,
        # Basic arithmetic functions
        "add": math.add,
        "sub": math.sub,
        "mul": math.mul,
        "div": math.div,
        "power": math.power,
        "clamp": math.clamp,
    }

    stdout_buffer = io.StringIO()

    try:
        tree = ast.parse(code.strip())
        if not tree.body:
            current_backend_name = getattr(backend.get_backend(), "name", str(backend.get_backend()))
            return {
                "success": True,
                "result": None,
                "result_type": "NoneType",
                "stdout": "",
                "backend": current_backend_name,
                "execution_time_ms": 0.0,
            }

        last_stmt = tree.body[-1]
        exec_body = tree.body[:-1]

        with redirect_stdout(stdout_buffer):
            if exec_body:
                mod_exec = ast.Module(body=exec_body, type_ignores=[])
                exec(compile(mod_exec, "<funcpie-eval>", "exec"), scope)

            if isinstance(last_stmt, ast.Expr):
                val = eval(compile(ast.Expression(body=last_stmt.value), "<funcpie-eval>", "eval"), scope)
            else:
                mod_last = ast.Module(body=[last_stmt], type_ignores=[])
                exec(compile(mod_last, "<funcpie-eval>", "exec"), scope)
                val = scope.get("result", None)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        current_backend_name = getattr(backend.get_backend(), "name", str(backend.get_backend()))
        return {
            "success": True,
            "result": _format_result(val),
            "result_type": type(val).__name__,
            "stdout": stdout_buffer.getvalue(),
            "backend": current_backend_name,
            "execution_time_ms": round(elapsed_ms, 3),
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        current_backend_name = getattr(backend.get_backend(), "name", str(backend.get_backend()))
        return {
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "stdout": stdout_buffer.getvalue(),
            "backend": current_backend_name,
            "execution_time_ms": round(elapsed_ms, 3),
        }
    finally:
        if use_gpu and orig_backend_name != getattr(backend.get_backend(), "name", "cpu"):
            try:
                backend.set_backend(orig_backend_name)
            except Exception:
                pass


def register_runner_tools(mcp: FastMCP) -> None:
    """Register discovery and runner tools onto the given FastMCP instance."""
    registry = get_registry()

    @mcp.tool()
    def funcpie_search(query: str, category: str | None = None) -> list[dict[str, object]]:
        """Search available FuncPie math, physics, and calculation functions.

        Args:
            query: Keyword to search (e.g., 'add', 'projectile', 'energy', 'derivative').
            category: Optional category filter (e.g., 'math.basic', 'physics.mechanics').

        Returns:
            List of matching functions with name, category, signature, and summary.
        """
        return registry.search(query=query, category=category)

    @mcp.tool()
    def funcpie_list(category: str | None = None) -> dict[str, list[str]]:
        """List all available FuncPie function categories or functions within a category.

        Args:
            category: Optional specific category to filter (e.g., 'math.basic').

        Returns:
            Dictionary mapping category names to lists of function/class names.
        """
        return registry.list_categories(category=category)

    @mcp.tool()
    def funcpie_describe(name: str) -> dict[str, object]:
        """Get detailed documentation, signature, and parameters for a specific function.

        Args:
            name: Exact name or qualified name (e.g., 'projectile_motion', 'Calculation').

        Returns:
            Dictionary containing item_type, signature, category, and full docstring.
        """
        item = registry.describe(name)
        if not item:
            return {"error": f"Item '{name}' not found in FuncPie registry."}
        return item

    @mcp.tool()
    def funcpie_execute(code: str, use_gpu: bool = False) -> dict[str, object]:
        """Execute arbitrary FuncPie math and physics expressions or scripts.

        Available in environment:
            - fp / funcpie: Entire library
            - Calculation: Fluent arithmetic engine
            - pipe, compose, curry: Functional pipeline utilities
            - math, physics, constants, backend
            - add, sub, mul, div, power, clamp, Vector2D, Vector3D, Matrix, Polynomial

        Examples:
            - Single expression: "Calculation(10).add(5).mul(2).value" -> 30.0
            - Functional pipeline: "pipe(10, lambda x: add(x, 5), lambda x: mul(x, 2))" -> 30
            - Physics computation: "physics.mechanics.projectile_motion(v0=20, angle_deg=45)"
            - Multi-line script: Assign 'result = ...' or leave the final line as an expression.

        Args:
            code: Python code string using FuncPie.
            use_gpu: If True, attempts to use HIP GPU backend for execution.

        Returns:
            Dictionary containing success, result, result_type, stdout, backend, execution_time_ms.
        """
        return execute_funcpie(code=code, use_gpu=use_gpu)

    # Register resources
    @mcp.resource("funcpie://catalog")
    def funcpie_catalog_resource() -> str:
        """Full catalog of all available operations in FuncPie."""
        import json
        return json.dumps(registry.list_categories(), indent=2)

    @mcp.resource("funcpie://constants")
    def funcpie_constants_resource() -> str:
        """Standard physical constants defined in CODATA 2018."""
        import json
        const_dict = {}
        for attr in dir(constants):
            if attr.isupper() and not attr.startswith("_"):
                const_dict[attr] = getattr(constants, attr)
        return json.dumps(const_dict, indent=2)

    # Register quickstart prompt
    @mcp.prompt()
    def funcpie_quickstart() -> str:
        """Guidance on how to use FuncPie tools for math and physics problem solving."""
        return (
            "You have access to FuncPie via 4 core tools:\n"
            "1. funcpie_search(query): Find formulas, functions, and constants.\n"
            "2. funcpie_describe(name): Inspect full signatures and docstrings.\n"
            "3. funcpie_list(category): Browse available capabilities.\n"
            "4. funcpie_execute(code): Run calculations, chained arithmetic (Calculation), pipelines (pipe), and physics simulations.\n"
            "Always prefer funcpie_execute to compute accurate scientific and mathematical values."
        )
