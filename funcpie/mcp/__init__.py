"""FastMCP server integration for FuncPie physics and mathematics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP

__all__ = ["create_mcp_server", "main"]


def create_mcp_server(name: str = "funcpie-physics") -> FastMCP:
    """Create and configure the FuncPie FastMCP server instance.

    Args:
        name: Name identifier for the server.

    Returns:
        FastMCP server instance.

    Raises:
        ImportError: If fastmcp or mcp is not installed.
    """
    from funcpie.mcp.server import create_mcp_server as _create_server

    return _create_server(name=name)


def main() -> None:
    """CLI entry point for the FuncPie FastMCP server."""
    from funcpie.mcp.server import main as _main

    _main()
