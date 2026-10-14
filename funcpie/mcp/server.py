"""FastMCP server factory and CLI runner for FuncPie with HTTP and Argon2/API Key support."""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

from funcpie.mcp.auth import APIKeyAuthenticator, APIKeyMiddleware
from funcpie.mcp.tools_physics import register_physics_tools
from funcpie.mcp.tools_runner import register_runner_tools

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


def create_mcp_server(name: str = "funcpie-physics", legacy_tools: bool = False) -> FastMCP:
    """Create and configure a FastMCP server instance equipped with discovery and runner tools.

    Args:
        name: Name identifier for the MCP server.
        legacy_tools: If True, also register individual legacy physics tools.

    Returns:
        Configured FastMCP instance ready for execution.
    """
    fastmcp_cls = get_fastmcp_class()
    mcp = fastmcp_cls(name)
    register_runner_tools(mcp)
    if legacy_tools:
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
        choices=["stdio", "sse", "http", "streamable-http"],
        default="stdio",
        help="Transport protocol to use (default: stdio)",
    )
    parser.add_argument(
        "--name",
        default="funcpie-physics",
        help="Server name identifier (default: funcpie-physics)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address to bind HTTP/SSE server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number to bind HTTP/SSE server (default: 8080)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Argon2 hash or plain API Key for authentication (supports ${ENV_VAR} format)",
    )
    parser.add_argument(
        "--api-key-env",
        default="FUNCPie_API_KEY",
        help="Environment variable name to read API Key / Argon2 hash from (default: FUNCPie_API_KEY)",
    )
    parser.add_argument(
        "--legacy-tools",
        action="store_true",
        help="Register legacy granular physics tools in addition to runner tools (default: False)",
    )
    args = parser.parse_args()

    mcp = create_mcp_server(name=args.name, legacy_tools=args.legacy_tools)

    try:
        if args.transport == "stdio":
            mcp.run(transport="stdio")
        else:
            # Determine API key / Argon2 hash from CLI or environment variable
            raw_key: str | None = args.api_key
            if not raw_key:
                raw_key = os.getenv(args.api_key_env) or os.getenv("FUNCPie_API_KEY") or os.getenv("MCP_API_KEY")

            if raw_key:
                from starlette.middleware import Middleware

                authenticator = APIKeyAuthenticator(raw_hash_or_key=raw_key)
                middleware = [Middleware(APIKeyMiddleware, authenticator=authenticator)]
                mcp.run(
                    transport=args.transport,
                    host=args.host,
                    port=args.port,
                    middleware=middleware,
                )
            else:
                mcp.run(
                    transport=args.transport,
                    host=args.host,
                    port=args.port,
                )
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
