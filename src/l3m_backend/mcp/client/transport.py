"""MCP transport factory for creating client connections."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

import httpx

from l3m_backend.mcp.exceptions import MCPTransportError
from l3m_backend.mcp.logging import log_mcp_exception

logger = logging.getLogger(__name__)


def _make_httpx_factory(verify: bool):
    """Create an httpx client factory with custom SSL verification setting."""
    def factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout or httpx.Timeout(30.0, read=300.0),
            auth=auth,
            follow_redirects=True,
            verify=verify,
        )
    return factory

# Retry configuration
SSE_MAX_RETRIES = 3
SSE_INITIAL_DELAY = 1.0  # seconds
SSE_MAX_DELAY = 10.0  # seconds
SSE_BACKOFF_FACTOR = 2.0

if TYPE_CHECKING:
    from mcp import ClientSession
    from l3m_backend.mcp.config import MCPServerConfig


@asynccontextmanager
async def create_stdio_transport(
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
) -> AsyncIterator["ClientSession"]:
    """Create a stdio transport connection to an MCP server.

    Args:
        command: The command to run
        args: Command arguments
        env: Environment variables (merged with current env)

    Yields:
        ClientSession for the MCP server
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

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    except Exception as e:
        if isinstance(e, MCPTransportError):
            raise
        # Log full traceback, raise clean error
        user_msg = log_mcp_exception(e, context="stdio transport error")
        raise MCPTransportError(user_msg) from None


async def _connect_sse_with_retry(
    url: str,
    headers: dict[str, str] | None,
    max_retries: int,
    sse_client,
    ClientSession,
    httpx_errors: tuple,
):
    """Attempt SSE connection with retry logic.

    Returns the context managers and session, or raises MCPTransportError.
    """
    delay = SSE_INITIAL_DELAY

    for attempt in range(max_retries + 1):
        try:
            # These are async context managers - we enter them manually
            sse_ctx = sse_client(url, headers=headers)
            read, write = await sse_ctx.__aenter__()

            session_ctx = ClientSession(read, write)
            session = await session_ctx.__aenter__()

            await session.initialize()
            return sse_ctx, session_ctx, session

        except httpx_errors as e:
            if attempt < max_retries:
                logger.warning(
                    f"SSE connection failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * SSE_BACKOFF_FACTOR, SSE_MAX_DELAY)
            else:
                # Log full traceback, raise clean error
                user_msg = log_mcp_exception(
                    e, context=f"SSE connection failed after {max_retries + 1} attempts"
                )
                raise MCPTransportError(user_msg) from None
        except Exception as e:
            # Log full traceback, raise clean error
            user_msg = log_mcp_exception(e, context="SSE connection error")
            raise MCPTransportError(user_msg) from None

    raise MCPTransportError("SSE connection failed: max retries exceeded")


@asynccontextmanager
async def create_sse_transport(
    url: str,
    headers: dict[str, str] | None = None,
    max_retries: int = SSE_MAX_RETRIES,
) -> AsyncIterator["ClientSession"]:
    """Create an SSE transport connection to an MCP server.

    Args:
        url: The SSE endpoint URL
        headers: Optional HTTP headers
        max_retries: Maximum number of connection retries

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

    # Import httpx errors for wrapping
    try:
        import httpx
        httpx_errors = (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout)
    except ImportError:
        httpx_errors = ()

    # Connect with retry
    sse_ctx, session_ctx, session = await _connect_sse_with_retry(
        url, headers, max_retries, sse_client, ClientSession, httpx_errors
    )

    try:
        yield session
    except Exception as e:
        # Wrap errors during session usage
        if httpx_errors and isinstance(e, httpx_errors):
            # Log full traceback, raise clean error
            user_msg = log_mcp_exception(e, context="SSE connection lost")
            raise MCPTransportError(user_msg) from None
        raise
    finally:
        # Clean up context managers in reverse order
        try:
            await session_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            await sse_ctx.__aexit__(None, None, None)
        except Exception:
            pass


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
