"""Tests for FuncPie MCP dynamic discovery and runner execution architecture."""

from __future__ import annotations

import pytest

from funcpie.mcp.registry import FuncPieRegistry, get_registry
from funcpie.mcp.server import create_mcp_server
from funcpie.mcp.tools_runner import execute_funcpie


def test_registry_initialization():
    """Verify registry populates categories across math, physics, core, and backend."""
    reg = get_registry()
    categories = reg.list_categories()
    assert "math.basic" in categories
    assert "math.algebra" in categories
    assert "physics.mechanics" in categories
    assert "physics.constants" in categories
    assert "core.pipeline" in categories
    assert "backend" in categories

    # Verify math.basic items
    basic_items = categories["math.basic"]
    assert "Calculation" in basic_items
    assert "add" in basic_items
    assert "sub" in basic_items
    assert "mul" in basic_items
    assert "div" in basic_items


def test_registry_search():
    """Verify keyword searching in the registry."""
    reg = get_registry()
    results = reg.search("projectile")
    assert len(results) >= 1
    assert any(r["name"] == "projectile_motion" for r in results)

    # Search with category filter
    math_adds = reg.search("add", category="math.basic")
    assert len(math_adds) >= 1
    assert any(r["name"] == "add" for r in math_adds)


def test_registry_describe():
    """Verify retrieving detailed metadata for operations."""
    reg = get_registry()
    desc = reg.describe("Calculation")
    assert desc is not None
    assert desc["name"] == "Calculation"
    assert desc["category"] == "math.basic"
    assert desc["item_type"] == "class"

    desc_fn = reg.describe("add")
    assert desc_fn is not None
    assert desc_fn["name"] == "add"
    assert desc_fn["item_type"] == "function"
    assert "Return a + b" in str(desc_fn["summary"])

    # Non-existent item
    assert reg.describe("non_existent_symbol_xyz") is None


def test_execute_arithmetic_chaining():
    """Verify Calculation fluent chaining via execute_funcpie."""
    code = "Calculation(10).add(15).mul(2).sub(10).div(2).value"
    res = execute_funcpie(code)
    assert res["success"] is True
    assert res["result"] == 20.0
    assert res["result_type"] == "float"
    assert res["backend"] == "cpu"
    assert isinstance(res["execution_time_ms"], (int, float))
    assert res["execution_time_ms"] >= 0


def test_execute_standalone_functions():
    """Verify standalone arithmetic operations."""
    res = execute_funcpie("add(10, 25)")
    assert res["success"] is True
    assert res["result"] == 35

    res_mul = execute_funcpie("mul(6, 7)")
    assert res_mul["success"] is True
    assert res_mul["result"] == 42


def test_execute_functional_pipeline():
    """Verify pipe and compose in execute_funcpie."""
    code = "pipe(5, lambda x: add(x, 10), lambda x: mul(x, 2))"
    res = execute_funcpie(code)
    assert res["success"] is True
    assert res["result"] == 30


def test_execute_physics_motion():
    """Verify physics computation execution."""
    code = """
summary = physics.mechanics.projectile_motion(v0=20.0, angle_deg=45.0)
summary.max_height
"""
    res = execute_funcpie(code)
    assert res["success"] is True
    assert isinstance(res["result"], (int, float))
    assert round(res["result"], 2) == 10.20


def test_execute_multiline_with_stdout():
    """Verify multiline execution with print output capture."""
    code = """
print("Computing values...")
a = 100
b = 50
result = a - b
"""
    res = execute_funcpie(code)
    assert res["success"] is True
    assert res["result"] == 50
    assert "Computing values..." in str(res["stdout"])


def test_execute_error_handling():
    """Verify exception and syntax error handling."""
    res_div = execute_funcpie("div(10, 0)")
    assert res_div["success"] is False
    assert res_div["error_type"] == "ZeroDivisionError"

    res_syntax = execute_funcpie("def invalid_syntax(:")
    assert res_syntax["success"] is False
    assert res_syntax["error_type"] == "SyntaxError"


def test_mcp_server_runner_tools():
    """Verify create_mcp_server registers the 4 core tools and resources."""
    import asyncio

    mcp = create_mcp_server(name="test-runner-server")
    tools = asyncio.run(mcp.list_tools())
    tool_names = [t.name for t in tools]

    assert "funcpie_search" in tool_names
    assert "funcpie_list" in tool_names
    assert "funcpie_describe" in tool_names
    assert "funcpie_execute" in tool_names

    # By default, legacy 33 tools should NOT be registered
    assert len(tool_names) == 4
    assert "physics_projectile_motion" not in tool_names


def test_mcp_server_legacy_tools_flag():
    """Verify --legacy-tools flag registers granular tools when requested."""
    import asyncio

    mcp = create_mcp_server(name="test-legacy-server", legacy_tools=True)
    tools = asyncio.run(mcp.list_tools())
    tool_names = [t.name for t in tools]

    assert "funcpie_search" in tool_names
    assert "physics_projectile_motion" in tool_names
    assert len(tool_names) > 30
