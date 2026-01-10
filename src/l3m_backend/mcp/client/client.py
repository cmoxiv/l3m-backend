"""MCP Client - manages connections to MCP servers."""
from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from l3m_backend.mcp.config import get_mcp_config_manager
from l3m_backend.mcp.exceptions import (
    MCPConnectionError,
    MCPServerNotFoundError,
    MCPToolError,
)

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Tool

logger = logging.getLogger(__name__)


# Global event loop for MCP operations - runs in a dedicated thread
_mcp_loop: asyncio.AbstractEventLoop | None = None
_mcp_thread: threading.Thread | None = None
_mcp_lock = threading.Lock()


def _get_mcp_loop() -> asyncio.AbstractEventLoop:
    """Get or create the dedicated MCP event loop."""
    global _mcp_loop, _mcp_thread

    with _mcp_lock:
        if _mcp_loop is None or not _mcp_loop.is_running():
            _mcp_loop = asyncio.new_event_loop()

            def run_loop():
                asyncio.set_event_loop(_mcp_loop)
                _mcp_loop.run_forever()

            _mcp_thread = threading.Thread(target=run_loop, daemon=True, name="mcp-event-loop")
            _mcp_thread.start()

            # Give the loop a moment to start
            import time
            time.sleep(0.01)

            # Register cleanup on exit
            atexit.register(_shutdown_mcp_loop)

        return _mcp_loop


def _shutdown_mcp_loop():
    """Shutdown the MCP event loop."""
    global _mcp_loop, _mcp_thread

    with _mcp_lock:
        if _mcp_loop is not None and _mcp_loop.is_running():
            _mcp_loop.call_soon_threadsafe(_mcp_loop.stop)
            if _mcp_thread is not None:
                _mcp_thread.join(timeout=2.0)
            _mcp_loop = None
            _mcp_thread = None


@dataclass
class MCPConnection:
    """Represents an active connection to an MCP server."""

    name: str
    session: "ClientSession"
    tools: list["Tool"] = field(default_factory=list)
    resources: list[Any] = field(default_factory=list)
    resource_templates: list[Any] = field(default_factory=list)
    prompts: list[Any] = field(default_factory=list)
    _stop_event: asyncio.Event | None = field(default=None, repr=False)
    _task: asyncio.Task | None = field(default=None, repr=False)

    async def refresh_capabilities(self) -> None:
        """Refresh the list of available tools, resources, and prompts."""
        try:
            tools_response = await self.session.list_tools()
            self.tools = tools_response.tools
        except Exception as e:
            logger.warning(f"Failed to list tools from {self.name}: {e}")
            self.tools = []

        try:
            resources_response = await self.session.list_resources()
            self.resources = resources_response.resources
        except Exception:
            self.resources = []

        try:
            templates_response = await self.session.list_resource_templates()
            self.resource_templates = templates_response.resourceTemplates
        except Exception:
            self.resource_templates = []

        try:
            prompts_response = await self.session.list_prompts()
            self.prompts = prompts_response.prompts
        except Exception:
            self.prompts = []


class _ConnectionRunner:
    """Manages an MCP connection lifecycle within a single async task."""

    def __init__(self, server_name: str, config: Any):
        self.server_name = server_name
        self.config = config
        self.connection: MCPConnection | None = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._error: Exception | None = None
        self._task: asyncio.Task | None = None

    async def run(self) -> None:
        """Run the connection - this stays alive until stop() is called."""
        try:
            from l3m_backend.mcp.client.transport import create_transport

            async with create_transport(self.config) as session:
                self.connection = MCPConnection(
                    name=self.server_name,
                    session=session,
                    _stop_event=self._stop_event,
                )
                await self.connection.refresh_capabilities()
                self._ready_event.set()

                # Wait until stop is requested
                await self._stop_event.wait()

        except Exception as e:
            self._error = e
            # Only raise if connection was never established (startup failure)
            # If connection was working and then dropped, don't raise - just exit
            # This prevents asyncio from printing "unhandled exception in task"
            if not self._ready_event.is_set():
                self._ready_event.set()  # Unblock waiters
                raise
            # Connection dropped after it was working - log quietly
            logger.debug(f"Connection to {self.server_name} dropped: {e}")

    async def wait_ready(self, timeout: float = 30.0) -> MCPConnection:
        """Wait for the connection to be ready."""
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise MCPConnectionError(f"Connection to {self.server_name} timed out")

        if self._error:
            # Extract actual error from TaskGroup exceptions
            error_msg = str(self._error)
            if hasattr(self._error, 'exceptions'):
                sub_errors = [str(ex) for ex in self._error.exceptions]
                error_msg = "; ".join(sub_errors)
            raise MCPConnectionError(f"Failed to connect to {self.server_name}: {error_msg}")

        if self.connection is None:
            raise MCPConnectionError(f"Connection to {self.server_name} failed")

        return self.connection

    def stop(self) -> None:
        """Signal the connection to stop."""
        self._stop_event.set()


class MCPClient:
    """Client for connecting to and managing MCP servers.

    This class manages connections to multiple MCP servers and provides
    a unified interface for listing and calling tools across all connected servers.

    Usage:
        client = MCPClient()

        # Connect to configured servers
        await client.connect("filesystem")
        await client.connect("brave-search")

        # List all available tools
        tools = await client.list_all_tools()

        # Call a tool
        result = await client.call_tool("filesystem", "read_file", {"path": "/etc/hosts"})

        # Disconnect
        await client.disconnect("filesystem")
        await client.disconnect_all()
    """

    def __init__(self):
        self._connections: dict[str, MCPConnection] = {}
        self._runners: dict[str, _ConnectionRunner] = {}
        self._configs: dict[str, Any] = {}  # Store configs for reconnect
        self._config_manager = get_mcp_config_manager()

    @property
    def connected_servers(self) -> list[str]:
        """List of currently connected server names."""
        return list(self._connections.keys())

    def is_connected(self, server_name: str) -> bool:
        """Check if a server is currently connected."""
        return server_name in self._connections

    def is_healthy(self, server_name: str) -> bool:
        """Check if a connection is still healthy.

        Returns False if the connection is missing, the runner task
        has completed (server died), or if the connection appears broken.
        """
        conn = self._connections.get(server_name)
        if not conn:
            return False
        runner = self._runners.get(server_name)
        if not runner:
            return False
        # If the task is done, the connection is dead
        if runner._task and runner._task.done():
            return False
        return True

    async def reconnect(self, server_name: str) -> MCPConnection | None:
        """Attempt to reconnect to a disconnected server.

        Works for both local (stdio) and remote (HTTP/SSE) servers.
        Uses the stored config from the original connection.

        Args:
            server_name: Name of the server to reconnect

        Returns:
            MCPConnection if successful, None if failed
        """
        # Get stored config
        config = self._configs.get(server_name)
        if not config:
            # Try to get from config manager
            try:
                config = self._config_manager.get_server(server_name)
            except MCPServerNotFoundError:
                logger.debug(f"No config found for {server_name}, cannot reconnect")
                return None

        # Disconnect cleanly if partially connected
        await self.disconnect(server_name)

        # Try to reconnect
        try:
            logger.info(f"Attempting to reconnect to {server_name}...")
            return await self.connect_with_config(server_name, config)
        except MCPConnectionError as e:
            logger.debug(f"Reconnection failed for {server_name}: {e}")
            return None

    def get_connection(self, server_name: str) -> MCPConnection | None:
        """Get the connection for a server, if connected."""
        return self._connections.get(server_name)

    async def connect(self, server_name: str) -> MCPConnection:
        """Connect to an MCP server by name.

        Args:
            server_name: Name of the server from configuration

        Returns:
            MCPConnection object for the connected server

        Raises:
            MCPServerNotFoundError: If server not in configuration
            MCPConnectionError: If connection fails
        """
        if server_name in self._connections:
            return self._connections[server_name]

        try:
            config = self._config_manager.get_server(server_name)
        except MCPServerNotFoundError:
            raise

        return await self.connect_with_config(server_name, config)

    async def connect_with_config(self, name: str, config: Any) -> MCPConnection:
        """Connect to an MCP server with explicit configuration.

        Args:
            name: Name to identify this connection
            config: MCPServerConfig object with connection parameters

        Returns:
            MCPConnection object for the connected server

        Raises:
            MCPConnectionError: If connection fails
        """
        if name in self._connections:
            return self._connections[name]

        try:
            # Store config for reconnect
            self._configs[name] = config

            # Create a runner that manages the connection lifecycle
            runner = _ConnectionRunner(name, config)
            self._runners[name] = runner

            # Start the connection task
            loop = asyncio.get_event_loop()
            runner._task = loop.create_task(runner.run())

            # Wait for connection to be ready
            connection = await runner.wait_ready()
            self._connections[name] = connection

            logger.info(f"Connected to MCP server: {name}")
            return connection

        except Exception as e:
            # Clean up on failure
            self._runners.pop(name, None)
            # Extract actual error from TaskGroup exceptions
            error_msg = str(e)
            if hasattr(e, 'exceptions'):
                # ExceptionGroup/TaskGroup - get first sub-exception
                sub_errors = [str(ex) for ex in e.exceptions]
                error_msg = "; ".join(sub_errors)
            raise MCPConnectionError(f"Failed to connect to {name}: {error_msg}")

    async def disconnect(self, server_name: str) -> bool:
        """Disconnect from an MCP server.

        Args:
            server_name: Name of the server to disconnect

        Returns:
            True if disconnected, False if wasn't connected
        """
        if server_name not in self._connections:
            return False

        try:
            # Signal the runner to stop
            runner = self._runners.get(server_name)
            if runner:
                runner.stop()
                # Wait for the task to complete
                if runner._task:
                    try:
                        await asyncio.wait_for(runner._task, timeout=5.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        runner._task.cancel()
                        try:
                            await runner._task
                        except asyncio.CancelledError:
                            pass

            del self._connections[server_name]
            self._runners.pop(server_name, None)
            logger.info(f"Disconnected from MCP server: {server_name}")
            return True

        except Exception as e:
            logger.warning(f"Error disconnecting from {server_name}: {e}")
            # Clean up anyway
            self._connections.pop(server_name, None)
            self._runners.pop(server_name, None)
            return True

    async def disconnect_all(self) -> None:
        """Disconnect from all connected servers."""
        for server_name in list(self._connections.keys()):
            await self.disconnect(server_name)

    async def list_tools(self, server_name: str) -> list["Tool"]:
        """List tools from a specific server.

        Args:
            server_name: Name of the server

        Returns:
            List of Tool objects
        """
        conn = self._connections.get(server_name)
        if not conn:
            raise MCPConnectionError(f"Not connected to server: {server_name}")
        return conn.tools

    async def list_all_tools(self) -> dict[str, list["Tool"]]:
        """List tools from all connected servers.

        Returns:
            Dict mapping server name to list of tools
        """
        return {name: conn.tools for name, conn in self._connections.items()}

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Call a tool on a specific server.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool result

        Raises:
            MCPConnectionError: If not connected to server
            MCPToolError: If tool call fails
        """
        conn = self._connections.get(server_name)
        if not conn:
            raise MCPConnectionError(f"Not connected to server: {server_name}")

        try:
            result = await conn.session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            raise MCPToolError(f"Error calling {tool_name} on {server_name}: {e}")

    async def connect_auto_servers(self) -> list[str]:
        """Connect to all servers marked for auto-connect.

        Returns:
            List of successfully connected server names
        """
        connected = []
        for server_name in self._config_manager.get_auto_connect_servers():
            try:
                await self.connect(server_name)
                connected.append(server_name)
            except Exception as e:
                logger.warning(f"Failed to auto-connect to {server_name}: {e}")
        return connected

    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - disconnect all servers."""
        await self.disconnect_all()


def run_async(coro):
    """Run an async coroutine in a sync context.

    This helper schedules the coroutine on the dedicated MCP event loop
    running in a background thread. This ensures all MCP operations
    happen in the same event loop, which is required for proper async
    context manager cleanup.
    """
    loop = _get_mcp_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()
