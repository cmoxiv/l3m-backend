"""MCP Server - exposes l3m tools as an MCP server."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from l3m_backend.mcp.exceptions import MCPError

if TYPE_CHECKING:
    from l3m_backend.core.registry import ToolRegistry

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP Server that exposes l3m tools.

    This allows l3m tools to be used by MCP clients like Claude Desktop.

    Usage:
        from l3m_backend.tools import registry
        server = MCPServer(registry)
        server.run()  # Runs on stdio by default
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        name: str = "l3m-tools",
        version: str = "1.0.0",
    ):
        self.registry = registry
        self.name = name
        self.version = version
        self._mcp = None

    def _create_server(self):
        """Create the FastMCP server instance."""
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError:
            raise MCPError(
                "MCP package not installed. Install with: pip install 'l3m-backend[mcp]'"
            )

        self._mcp = FastMCP(self.name)
        self._register_tools()
        return self._mcp

    def _register_tools(self) -> None:
        """Register all l3m tools with the MCP server."""
        if self._mcp is None:
            return

        for entry in self.registry:
            # Create a tool handler for this entry
            self._register_tool(entry)

    def _register_tool(self, entry) -> None:
        """Register a single tool with MCP."""
        from l3m_backend.core.datamodels import ToolEntry

        if not isinstance(entry, ToolEntry):
            return

        # Get the parameter schema
        params_model = entry.get_params_model()
        schema = params_model.model_json_schema()

        # Create the tool function dynamically
        def make_handler(tool_entry):
            def handler(**kwargs) -> str:
                """Execute the l3m tool."""
                try:
                    result = self.registry.execute(tool_entry.name, kwargs)
                    # Format result for MCP
                    if hasattr(result.output, "llm_format") and result.output.llm_format:
                        return result.output.llm_format
                    elif hasattr(result.output, "data"):
                        return str(result.output.data)
                    else:
                        return str(result.output)
                except Exception as e:
                    return f"Error: {e}"

            return handler

        handler = make_handler(entry)
        handler.__name__ = entry.name
        handler.__doc__ = entry.description or f"l3m tool: {entry.name}"

        # Register with FastMCP
        # FastMCP uses decorators, but we can also register directly
        self._mcp.tool()(handler)

        logger.debug(f"Registered tool with MCP: {entry.name}")

    def run(self, transport: str = "stdio") -> None:
        """Run the MCP server.

        Args:
            transport: Transport type - "stdio" or "sse"
        """
        mcp = self._create_server()

        logger.info(f"Starting MCP server '{self.name}' with {transport} transport")

        if transport == "stdio":
            mcp.run(transport="stdio")
        elif transport == "sse":
            mcp.run(transport="sse")
        else:
            raise MCPError(f"Unknown transport: {transport}")


def create_mcp_server(
    registry: Optional["ToolRegistry"] = None,
    name: str = "l3m-tools",
) -> MCPServer:
    """Create an MCP server with the given or default registry.

    Args:
        registry: Tool registry to expose (uses global if None)
        name: Server name

    Returns:
        MCPServer instance
    """
    if registry is None:
        from l3m_backend.tools import registry as default_registry
        registry = default_registry

    return MCPServer(registry, name=name)
