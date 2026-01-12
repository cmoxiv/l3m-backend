"""MCP (Model Context Protocol) support for l3m-backend.

This module provides both client and server capabilities:

Client:
    Connect to external MCP servers and use their tools in l3m.

    from l3m_backend.mcp import MCPClient, MCPRegistryAdapter

    client = MCPClient()
    await client.connect("filesystem")

    adapter = MCPRegistryAdapter(registry, client)
    await adapter.register_all()

Server:
    Expose l3m tools as an MCP server for use by other clients.

    from l3m_backend.mcp import MCPServer, create_mcp_server

    server = create_mcp_server()
    server.run()  # Runs on stdio
"""
from __future__ import annotations

from l3m_backend.mcp.client import MCPClient, MCPRegistryAdapter
from l3m_backend.mcp.server import MCPServer, create_mcp_server
from l3m_backend.mcp.config import (
    MCPConfig,
    MCPConfigManager,
    MCPServerConfig,
    get_mcp_config_manager,
)
from l3m_backend.mcp.exceptions import (
    MCPConfigError,
    MCPConnectionError,
    MCPError,
    MCPServerNotFoundError,
    MCPToolError,
    MCPTransportError,
)
from l3m_backend.mcp.logging import (
    close_session_logging,
    configure_session_logging,
    get_current_log_path,
    get_log_path,
    log_mcp_debug,
    log_mcp_exception,
    log_mcp_info,
    log_mcp_warning,
)

__all__ = [
    # Client
    "MCPClient",
    "MCPRegistryAdapter",
    # Server
    "MCPServer",
    "create_mcp_server",
    # Config
    "MCPConfig",
    "MCPConfigManager",
    "MCPServerConfig",
    "get_mcp_config_manager",
    # Exceptions
    "MCPError",
    "MCPConfigError",
    "MCPConnectionError",
    "MCPServerNotFoundError",
    "MCPToolError",
    "MCPTransportError",
    # Logging
    "configure_session_logging",
    "close_session_logging",
    "get_log_path",
    "get_current_log_path",
    "log_mcp_exception",
    "log_mcp_warning",
    "log_mcp_info",
    "log_mcp_debug",
]
