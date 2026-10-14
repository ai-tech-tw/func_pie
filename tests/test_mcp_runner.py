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


def test_execute_cel_arithmetic():
    """Verify CEL arithmetic and nested functions via evaluate_cel."""
    from funcpie.mcp.cel_engine import evaluate_cel

    res = evaluate_cel("add(10, 20) * 2 - sub(30, 10)")
    assert res["success"] is True
    assert res["result"] == 40
    assert res["engine"] == "cel"
    assert isinstance(res["execution_time_ms"], (int, float))


def test_execute_cel_chaining():
    """Verify receiver method chaining in CEL."""
    from funcpie.mcp.cel_engine import evaluate_cel

    res = evaluate_cel("(10).add(15).mul(2)")
    assert res["success"] is True
    assert res["result"] == 50


def test_execute_cel_physics():
    """Verify CEL physics function call and dot attribute access."""
    from funcpie.mcp.cel_engine import evaluate_cel

    res = evaluate_cel("projectile_motion(20.0, 45.0).max_height")
    assert res["success"] is True
    assert isinstance(res["result"], (int, float))
    assert round(float(res["result"]), 2) == 10.20


def test_execute_cel_constants_and_variables():
    """Verify CEL evaluation with constants and custom variables."""
    from funcpie.mcp.cel_engine import evaluate_cel

    res_c = evaluate_cel("SPEED_OF_LIGHT")
    assert res_c["success"] is True
    assert res_c["result"] == 299792458.0

    res_ke = evaluate_cel("kinetic_energy(m, v)", variables={"m": 2.0, "v": 10.0})
    assert res_ke["success"] is True
    assert res_ke["result"] == 100.0


def test_cel_security_penetration_prevention():
    """Verify CEL completely blocks code injection and sandbox escape attempts."""
    from funcpie.mcp.cel_engine import evaluate_cel

    # Attack 1: OS / subprocess import
    r1 = evaluate_cel("__import__('os').system('ls')")
    assert r1["success"] is False
    assert "system" in str(r1["error"]) or r1["error_type"] in ("CELEvalError", "CELParseError")

    # Attack 2: File access via open
    r2 = evaluate_cel("open('/etc/passwd').read()")
    assert r2["success"] is False
    assert "open" in str(r2["error"]) or r2["error_type"] in ("CELEvalError", "CELParseError")

    # Attack 3: Python MRO sandbox escape
    r3 = evaluate_cel("().__class__.__base__.__subclasses__()")
    assert r3["success"] is False

    # Attack 4: Dynamic eval invocation
    r4 = evaluate_cel("eval('1 + 1')")
    assert r4["success"] is False

    # Attack 5: Statement and infinite loop rejection
    r5 = evaluate_cel("while True: pass")
    assert r5["success"] is False


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
