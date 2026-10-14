"""FastMCP server factory and CLI runner for FuncPie."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from funcpie.mcp.tools_physics import register_physics_tools

if TYPE_CHECKING:
    from fastmcp import FastMCP


def get_fastmcp_class() -> type[FastMCP]:
    """Retrieve the available FastMCP class or raise an informative ImportError."""
    try:
        from fastmcp import FastMCP as _FastMCP

        return _FastMCP
    except ImportError:
        pass

    raise ImportError(
        "FastMCP is required to run the FuncPie MCP server. "
        "Please install it with: pip install 'funcpie[mcp]' or pip install fastmcp"
    )


def create_mcp_server(name: str = "funcpie-physics") -> FastMCP:
    """Create and configure a FastMCP server instance equipped with physics tools.

    Args:
        name: Name identifier for the MCP server.

    Returns:
        Configured FastMCP instance ready for execution.
    """
    fastmcp_cls = get_fastmcp_class()
    mcp = fastmcp_cls(name)
    register_physics_tools(mcp)
    return mcp


def main() -> None:
    """Entry point for the `funcpie-mcp` command-line executable."""
    parser = argparse.ArgumentParser(
        prog="funcpie-mcp",
        description="FuncPie FastMCP server providing physics and mathematics tools for AI agents.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol to use (default: stdio)",
    )
    parser.add_argument(
        "--name",
        default="funcpie-physics",
        help="Server name identifier (default: funcpie-physics)",
    )
    args = parser.parse_args()

    mcp = create_mcp_server(name=args.name)
    try:
        mcp.run(transport=args.transport)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
