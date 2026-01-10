"""MCP transport factory for creating client connections."""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

from l3m_backend.mcp.exceptions import MCPTransportError

if TYPE_CHECKING:
    from mcp import ClientSession
    from l3m_backend.mcp.config import MCPServerConfig


@asynccontextmanager
async def create_stdio_transport(
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
) -> AsyncIterator[tuple["ClientSession", any]]:
    """Create a stdio transport connection to an MCP server.

    Args:
        command: The command to run
        args: Command arguments
        env: Environment variables (merged with current env)

    Yields:
        Tuple of (read_stream, write_stream) for the MCP client
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        raise MCPTransportError(
            "MCP package not installed. Install with: pip install 'l3m-backend[mcp]'"
        )

    # Merge environment
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=full_env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def create_sse_transport(
    url: str,
    headers: dict[str, str] | None = None,
) -> AsyncIterator["ClientSession"]:
    """Create an SSE transport connection to an MCP server.

    Args:
        url: The SSE endpoint URL
        headers: Optional HTTP headers

    Yields:
        ClientSession connected via SSE
    """
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
    except ImportError:
        raise MCPTransportError(
            "MCP package not installed. Install with: pip install 'l3m-backend[mcp]'"
        )

    async with sse_client(url, headers=headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def create_transport(config: "MCPServerConfig") -> AsyncIterator["ClientSession"]:
    """Create a transport connection based on server configuration.

    Args:
        config: Server configuration

    Yields:
        ClientSession for the MCP server
    """
    if config.transport == "stdio":
        if not config.command:
            raise MCPTransportError("stdio transport requires 'command' in config")
        async with create_stdio_transport(
            command=config.command,
            args=config.args,
            env=config.resolve_env(),
        ) as session:
            yield session

    elif config.transport in ("http", "sse"):
        if not config.url:
            raise MCPTransportError(f"{config.transport} transport requires 'url' in config")
        async with create_sse_transport(
            url=config.url,
            headers=config.headers or None,
        ) as session:
            yield session

    else:
        raise MCPTransportError(f"Unknown transport type: {config.transport}")
