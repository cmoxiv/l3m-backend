"""Adapter to register MCP tools with l3m's ToolRegistry."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field, create_model

from l3m_backend.core.helpers import _normalize_name
from l3m_backend.mcp.exceptions import MCPConnectionError, MCPToolError

if TYPE_CHECKING:
    from mcp.types import Tool

    from l3m_backend.core.registry import ToolRegistry
    from l3m_backend.mcp.client.client import MCPClient

logger = logging.getLogger(__name__)


def _json_schema_to_python_type(prop_schema: dict) -> Any:
    """Convert JSON Schema type to Python type."""
    json_type = prop_schema.get("type", "string")

    if json_type == "string":
        return str
    elif json_type == "integer":
        return int
    elif json_type == "number":
        return float
    elif json_type == "boolean":
        return bool
    elif json_type == "array":
        return list
    elif json_type == "object":
        return dict
    else:
        return Any


def _create_params_model_from_schema(name: str, schema: dict) -> type[BaseModel]:
    """Create a Pydantic model from a JSON Schema."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    field_definitions = {}
    for prop_name, prop_schema in properties.items():
        python_type = _json_schema_to_python_type(prop_schema)
        description = prop_schema.get("description", "")

        if prop_name in required:
            field_definitions[prop_name] = (python_type, Field(description=description))
        else:
            default = prop_schema.get("default", None)
            field_definitions[prop_name] = (python_type, Field(default=default, description=description))

    # Create model name from tool name
    model_name = f"{name.replace('.', '_').replace('-', '_')}_Params"

    return create_model(model_name, **field_definitions)


class MCPRegistryAdapter:
    """Adapts MCP tools to l3m's ToolRegistry format.

    This class discovers tools from connected MCP servers and registers them
    with the l3m ToolRegistry, allowing them to be used like built-in tools.

    Tool names are prefixed with 'mcp.<server>.' to avoid conflicts:
    - mcp.filesystem.read_file
    - mcp.brave.search

    Usage:
        registry = ToolRegistry()
        client = MCPClient()
        await client.connect("filesystem")

        adapter = MCPRegistryAdapter(registry, client)
        await adapter.register_all()

        # Now registry has mcp.filesystem.* tools
    """

    def __init__(self, registry: "ToolRegistry", client: "MCPClient"):
        self.registry = registry
        self.client = client
        self._registered_tools: dict[str, str] = {}  # tool_name -> server_name

    def _make_tool_name(self, server_name: str, tool_name: str) -> str:
        """Create the full tool name with server prefix."""
        return f"mcp.{server_name}.{tool_name}"

    def _create_tool_callable(
        self,
        server_name: str,
        tool_name: str,
    ) -> Callable[..., Any]:
        """Create a sync callable that wraps the async MCP tool call."""
        from l3m_backend.mcp.client.client import run_async

        def _extract_result(result: Any) -> str:
            """Extract content from MCP result."""
            if hasattr(result, "content") and result.content:
                # MCP returns content as a list of content blocks
                contents = []
                for block in result.content:
                    if hasattr(block, "text"):
                        contents.append(block.text)
                    elif hasattr(block, "data"):
                        contents.append(str(block.data))
                return "\n".join(contents) if contents else str(result)
            return str(result)

        def call_tool(**kwargs: Any) -> Any:
            """Synchronous wrapper for MCP tool call."""
            try:
                # Run the async call on the MCP event loop
                result = run_async(
                    self.client.call_tool(server_name, tool_name, kwargs)
                )
                return _extract_result(result)

            except (MCPConnectionError, MCPToolError) as e:
                # Check if connection is unhealthy and try to reconnect
                if not self.client.is_healthy(server_name):
                    logger.info(f"Connection to {server_name} appears dead, attempting reconnect...")
                    conn = run_async(self.client.reconnect(server_name))
                    if conn:
                        # Retry the call once
                        try:
                            result = run_async(
                                self.client.call_tool(server_name, tool_name, kwargs)
                            )
                            return _extract_result(result)
                        except Exception as retry_e:
                            # Clean error message on retry failure
                            error_msg = str(retry_e).split('\n')[0]
                            raise MCPToolError(f"MCP error: {error_msg}")
                # Clean error message (no traceback)
                error_msg = str(e).split('\n')[0]
                raise MCPToolError(f"MCP error: {error_msg}")

            except Exception as e:
                # Clean error message for other exceptions
                error_msg = str(e).split('\n')[0]
                raise MCPToolError(f"MCP error: {error_msg}")

        return call_tool

    async def register_server_tools(self, server_name: str) -> int:
        """Register all tools from a specific MCP server.

        Args:
            server_name: Name of the connected server

        Returns:
            Number of tools registered
        """
        from l3m_backend.core.datamodels import ToolEntry

        tools = await self.client.list_tools(server_name)
        count = 0

        for mcp_tool in tools:
            full_name = self._make_tool_name(server_name, mcp_tool.name)

            # Create the callable
            callable_fn = self._create_tool_callable(server_name, mcp_tool.name)

            # Create params model from MCP schema
            params_model = None
            if mcp_tool.inputSchema:
                try:
                    params_model = _create_params_model_from_schema(
                        full_name, mcp_tool.inputSchema
                    )
                except Exception as e:
                    logger.warning(f"Failed to create params model for {full_name}: {e}")

            # Create ToolEntry with minimal description (first line only)
            description = mcp_tool.description or f"MCP tool from {server_name}"
            if description:
                description = description.split('\n')[0].strip()

            entry = ToolEntry(
                name=full_name,
                callable_fn=callable_fn,
                aliases=[mcp_tool.name] if not self._has_name_conflict(mcp_tool.name) else [],
                description=description,
                params_model=params_model,
            )

            # Register with the registry (use normalized key for consistency)
            normalized_name = _normalize_name(full_name)
            self.registry._tools[normalized_name] = entry
            self._registered_tools[normalized_name] = server_name

            # Also register short alias if no conflict
            if entry.aliases:
                for alias in entry.aliases:
                    self.registry._aliases[_normalize_name(alias)] = normalized_name

            count += 1
            logger.debug(f"Registered MCP tool: {full_name}")

        logger.info(f"Registered {count} tools from MCP server: {server_name}")
        return count

    async def register_all(self) -> int:
        """Register tools from all connected MCP servers.

        Returns:
            Total number of tools registered
        """
        total = 0
        for server_name in self.client.connected_servers:
            total += await self.register_server_tools(server_name)
        return total

    def unregister_server_tools(self, server_name: str) -> int:
        """Unregister all tools from a specific MCP server.

        Args:
            server_name: Name of the server

        Returns:
            Number of tools unregistered
        """
        to_remove = [
            name for name, srv in self._registered_tools.items() if srv == server_name
        ]

        for name in to_remove:
            # Remove from registry
            self.registry._tools.pop(name, None)
            # Remove aliases
            for alias, target in list(self.registry._aliases.items()):
                if target == name:
                    del self.registry._aliases[alias]
            # Remove from tracking
            del self._registered_tools[name]

        logger.info(f"Unregistered {len(to_remove)} tools from MCP server: {server_name}")
        return len(to_remove)

    def unregister_all(self) -> int:
        """Unregister all MCP tools.

        Returns:
            Number of tools unregistered
        """
        count = len(self._registered_tools)
        servers = set(self._registered_tools.values())
        for server in servers:
            self.unregister_server_tools(server)
        return count

    def _has_name_conflict(self, name: str) -> bool:
        """Check if a tool name conflicts with existing tools."""
        # Check if name exists as a tool or alias (use normalized name)
        normalized = _normalize_name(name)
        return (
            normalized in self.registry._tools
            or normalized in self.registry._aliases
            or normalized in self._registered_tools
        )

    def get_mcp_tools(self) -> dict[str, str]:
        """Get all registered MCP tools and their servers.

        Returns:
            Dict mapping tool name to server name
        """
        return dict(self._registered_tools)
