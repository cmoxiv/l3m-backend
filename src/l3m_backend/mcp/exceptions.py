"""MCP-specific exceptions."""
from __future__ import annotations


class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class MCPConnectionError(MCPError):
    """Error connecting to an MCP server."""
    pass


class MCPServerNotFoundError(MCPError):
    """MCP server not found in configuration."""
    pass


class MCPToolError(MCPError):
    """Error executing an MCP tool."""
    pass


class MCPConfigError(MCPError):
    """Error in MCP configuration."""
    pass


class MCPTransportError(MCPError):
    """Error with MCP transport layer."""
    pass
