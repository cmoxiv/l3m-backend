"""MCP command - manage MCP server connections."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register(
    "mcp",
    "Manage MCP server connections",
    usage="/mcp <subcommand> [args]",
    has_args=True,
)
def cmd_mcp(
    engine: "ChatEngine",
    session_mgr: "SessionManager | None",
    args: str,
) -> bool | None:
    """Manage MCP server connections.

    Subcommands:
        /mcp status              - Show connected servers and tools
        /mcp list                - List configured servers
        /mcp connect <server>    - Connect to a server
        /mcp disconnect <server> - Disconnect from a server
        /mcp tools [server]      - List tools from MCP servers
        /mcp add <name> --command <cmd>  - Add a server configuration
        /mcp remove <name>       - Remove a server configuration
    """
    if not args:
        _show_help()
        return None

    parts = args.split(None, 1)
    subcommand = parts[0].lower()
    subargs = parts[1] if len(parts) > 1 else ""

    # Check if MCP is available
    try:
        from l3m_backend.mcp import get_mcp_config_manager
    except ImportError:
        print("MCP support not installed. Install with: pip install 'l3m-backend[mcp]'")
        return None

    if subcommand == "status":
        _cmd_status(engine)
    elif subcommand == "list":
        _cmd_list()
    elif subcommand == "connect":
        _cmd_connect(engine, subargs)
    elif subcommand == "disconnect":
        _cmd_disconnect(engine, subargs)
    elif subcommand == "tools":
        _cmd_tools(engine, subargs)
    elif subcommand == "prompts":
        _cmd_prompts(engine, subargs)
    elif subcommand == "prompt":
        _cmd_prompt(engine, subargs)
    elif subcommand == "resources":
        _cmd_resources(engine, subargs)
    elif subcommand == "resource":
        _cmd_resource(engine, subargs)
    elif subcommand == "add":
        _cmd_add(subargs)
    elif subcommand == "remove":
        _cmd_remove(subargs)
    elif subcommand == "auto":
        _cmd_auto(subargs)
    else:
        print(f"Unknown subcommand: {subcommand}")
        _show_help()

    return None


def _show_help() -> None:
    """Show MCP command help."""
    print("""
MCP Server Management

Usage: /mcp <subcommand> [args]

Subcommands:
  status              Show connected servers and their tools
  list                List configured servers
  connect             Connect to MCP server (see below)
  disconnect <name>   Disconnect from a server
  tools [server]      List tools from connected servers
  prompts [server]    List prompts from connected servers
  prompt <name> [...] Get a prompt (e.g., /mcp prompt summarize text="hello")
  resources [server]  List resources from connected servers
  resource <uri>      Read a resource (e.g., /mcp resource test://info)
  add <name> <cmd|url> Add server config
  remove <name>       Remove a server configuration
  auto <name>         Toggle auto-connect for a server

Connect Syntax:
  /mcp connect <name>                          Pre-configured server
  /mcp connect <name> stdio <command...>       Subprocess
  /mcp connect <name> http://localhost:8080    HTTP/SSE URL
  /mcp connect <name> sse <url>                Explicit SSE

Examples:
  /mcp connect test                            # Pre-configured
  /mcp connect app stdio python -m my_server   # Subprocess
  /mcp connect remote http://localhost:8080    # HTTP/SSE
  /mcp tools app
  /mcp disconnect app
""")


def _cmd_status(engine) -> None:
    """Show MCP connection status."""
    mcp_client = getattr(engine, "_mcp_client", None)
    if not mcp_client:
        print("No MCP client initialized. Use --mcp flag when starting l3m-chat.")
        return

    servers = mcp_client.connected_servers
    if not servers:
        print("No MCP servers connected.")
        return

    print("Connected MCP servers:")
    for name in servers:
        conn = mcp_client.get_connection(name)
        if conn:
            print(f"  {name}: {len(conn.tools)} tools")


def _cmd_list() -> None:
    """List configured MCP servers."""
    from l3m_backend.mcp import get_mcp_config_manager

    mgr = get_mcp_config_manager()
    servers = mgr.list_servers()

    if not servers:
        print("No MCP servers configured.")
        print("Add one with: /mcp add <name> <command>")
        return

    print("Configured MCP servers:")
    config = mgr.load()
    for name in servers:
        srv = config.servers[name]
        auto = " [auto]" if srv.auto_connect else ""
        desc = f" - {srv.description}" if srv.description else ""
        print(f"  {name}{auto}{desc}")
        if srv.transport == "stdio" and srv.command:
            cmd = f"{srv.command} {' '.join(srv.args)}"
            print(f"    Command: {cmd}")
        elif srv.url:
            print(f"    URL: {srv.url}")


def _cmd_connect(engine, args: str) -> None:
    """Connect to an MCP server.

    Supports two modes:
    1. Pre-configured: /mcp connect <name>
    2. Dynamic: /mcp connect <name> <transport> <params...>

    Transports:
        stdio  - Subprocess command (e.g., /mcp connect app stdio python -m server)
        http   - HTTP URL (e.g., /mcp connect app http://localhost:8080)
        sse    - SSE URL (e.g., /mcp connect app sse http://localhost:8080)
    """
    if not args:
        print("Usage: /mcp connect <name> [transport] [params...]")
        print()
        print("Pre-configured server:")
        print("  /mcp connect test")
        print()
        print("Dynamic connection:")
        print("  /mcp connect app stdio python -m my_mcp_server")
        print("  /mcp connect app http://localhost:8080")
        print("  /mcp connect app sse http://localhost:8080/sse")
        return

    mcp_client = getattr(engine, "_mcp_client", None)
    if not mcp_client:
        print("No MCP client initialized. Use --mcp flag when starting l3m-chat.")
        return

    parts = args.split(None, 2)
    name = parts[0]

    if len(parts) == 1:
        # Pre-configured server
        _connect_configured_server(engine, mcp_client, name)
    else:
        # Dynamic connection - check if second part is a transport or URL
        transport_or_url = parts[1]
        params = parts[2] if len(parts) > 2 else ""

        if transport_or_url.startswith("http://") or transport_or_url.startswith("https://"):
            # URL provided directly: /mcp connect app http://localhost:8080
            _connect_dynamic_server(engine, mcp_client, name, "sse", transport_or_url)
        elif transport_or_url.lower() in ("stdio", "http", "sse"):
            # Transport explicitly specified
            _connect_dynamic_server(engine, mcp_client, name, transport_or_url.lower(), params)
        else:
            # Assume it's a command (stdio transport implied)
            # /mcp connect app python -m server
            full_cmd = f"{transport_or_url} {params}".strip()
            _connect_dynamic_server(engine, mcp_client, name, "stdio", full_cmd)


def _connect_configured_server(engine, mcp_client, server_name: str) -> None:
    """Connect to a pre-configured MCP server."""
    from l3m_backend.mcp.client.client import run_async

    try:
        run_async(mcp_client.connect(server_name))
        conn = mcp_client.get_connection(server_name)
        if conn:
            print(f"Connected to {server_name}: {len(conn.tools)} tools available")

            # Register tools with engine's registry
            mcp_adapter = getattr(engine, "_mcp_adapter", None)
            if mcp_adapter:
                run_async(mcp_adapter.register_server_tools(server_name))

    except Exception as e:
        # Clean error message (no traceback)
        error_msg = str(e).split('\n')[0]
        print(f"Failed to connect: {error_msg}")


def _connect_dynamic_server(engine, mcp_client, name: str, transport: str, params: str) -> None:
    """Connect to an MCP server with dynamic configuration."""
    from l3m_backend.mcp.client.client import run_async
    from l3m_backend.mcp.config import MCPServerConfig

    # Build config based on transport
    if transport == "stdio":
        if not params:
            print("Error: stdio transport requires a command")
            print("Example: /mcp connect app stdio python -m my_server")
            return
        cmd_parts = params.split()
        command = cmd_parts[0]
        args = cmd_parts[1:] if len(cmd_parts) > 1 else []
        config = MCPServerConfig(transport="stdio", command=command, args=args)
    elif transport in ("http", "sse"):
        if not params:
            print(f"Error: {transport} transport requires a URL")
            print(f"Example: /mcp connect app {transport} http://localhost:8080")
            return
        config = MCPServerConfig(transport="sse", url=params)
    else:
        print(f"Unknown transport: {transport}")
        print("Supported: stdio, http, sse")
        return

    try:
        run_async(mcp_client.connect_with_config(name, config))
        conn = mcp_client.get_connection(name)
        if conn:
            print(f"Connected to {name}: {len(conn.tools)} tools available")

            # Register tools with engine's registry
            mcp_adapter = getattr(engine, "_mcp_adapter", None)
            if mcp_adapter:
                run_async(mcp_adapter.register_server_tools(name))

    except Exception as e:
        # Clean error message (no traceback)
        error_msg = str(e).split('\n')[0]
        print(f"Failed to connect: {error_msg}")


def _cmd_disconnect(engine, server_name: str) -> None:
    """Disconnect from an MCP server."""
    if not server_name:
        print("Usage: /mcp disconnect <server>")
        return

    mcp_client = getattr(engine, "_mcp_client", None)
    if not mcp_client:
        print("No MCP client initialized.")
        return

    from l3m_backend.mcp.client.client import run_async

    try:
        # Unregister tools first
        mcp_adapter = getattr(engine, "_mcp_adapter", None)
        if mcp_adapter:
            mcp_adapter.unregister_server_tools(server_name)

        run_async(mcp_client.disconnect(server_name))
        print(f"Disconnected from {server_name}")
    except Exception as e:
        # Clean error message (no traceback)
        error_msg = str(e).split('\n')[0]
        print(f"Failed to disconnect: {error_msg}")


def _cmd_tools(engine, server_name: str = "") -> None:
    """List tools from MCP servers."""
    mcp_client = getattr(engine, "_mcp_client", None)
    if not mcp_client:
        print("No MCP client initialized.")
        return

    servers = [server_name] if server_name else mcp_client.connected_servers
    if not servers:
        print("No MCP servers connected.")
        return

    for name in servers:
        conn = mcp_client.get_connection(name)
        if not conn:
            print(f"{name}: not connected")
            continue

        print(f"\n{name}:")
        for tool in conn.tools:
            desc = f" - {tool.description}" if tool.description else ""
            print(f"  {tool.name}{desc}")


def _cmd_prompts(engine, server_name: str = "") -> None:
    """List prompts from MCP servers."""
    mcp_client = getattr(engine, "_mcp_client", None)
    if not mcp_client:
        print("No MCP client initialized.")
        return

    servers = [server_name] if server_name else mcp_client.connected_servers
    if not servers:
        print("No MCP servers connected.")
        return

    for name in servers:
        conn = mcp_client.get_connection(name)
        if not conn:
            print(f"{name}: not connected")
            continue

        print(f"\n{name}:")
        if not conn.prompts:
            print("  (no prompts)")
            continue
        for prompt in conn.prompts:
            desc = f" - {prompt.description}" if prompt.description else ""
            print(f"  {prompt.name}{desc}")
            if hasattr(prompt, "arguments") and prompt.arguments:
                args_str = ", ".join(
                    f"{a.name}{'?' if not a.required else ''}"
                    for a in prompt.arguments
                )
                print(f"    Arguments: {args_str}")


def _cmd_resources(engine, server_name: str = "") -> None:
    """List resources from MCP servers."""
    mcp_client = getattr(engine, "_mcp_client", None)
    if not mcp_client:
        print("No MCP client initialized.")
        return

    servers = [server_name] if server_name else mcp_client.connected_servers
    if not servers:
        print("No MCP servers connected.")
        return

    for name in servers:
        conn = mcp_client.get_connection(name)
        if not conn:
            print(f"{name}: not connected")
            continue

        print(f"\n{name}:")
        has_any = False

        # Static resources
        if conn.resources:
            has_any = True
            for resource in conn.resources:
                desc = f" - {resource.description}" if hasattr(resource, "description") and resource.description else ""
                uri = str(resource.uri) if hasattr(resource, "uri") else str(resource)
                print(f"  {uri}{desc}")

        # Resource templates
        if hasattr(conn, "resource_templates") and conn.resource_templates:
            has_any = True
            for template in conn.resource_templates:
                desc = f" - {template.description.split(chr(10))[0]}" if hasattr(template, "description") and template.description else ""
                uri = str(template.uriTemplate) if hasattr(template, "uriTemplate") else str(template)
                print(f"  {uri}{desc}")

        if not has_any:
            print("  (no resources)")


def _cmd_resource(engine, uri: str) -> None:
    """Read a specific resource by URI."""
    if not uri:
        print("Usage: /mcp resource <uri>")
        print("Example: /mcp resource test://info")
        print("Example: /mcp resource test://greeting/World")
        return

    mcp_client = getattr(engine, "_mcp_client", None)
    if not mcp_client:
        print("No MCP client initialized.")
        return

    from l3m_backend.mcp.client.client import run_async

    # Try each connected server
    for server_name in mcp_client.connected_servers:
        conn = mcp_client.get_connection(server_name)
        if not conn:
            continue

        # Check static resources
        for resource in conn.resources:
            resource_uri = str(resource.uri) if hasattr(resource, "uri") else str(resource)
            if resource_uri == uri:
                try:
                    result = run_async(conn.session.read_resource(uri))
                    print(f"\n{uri}:")
                    for content in result.contents:
                        if hasattr(content, "text"):
                            print(content.text)
                        else:
                            print(str(content))
                    return
                except Exception as e:
                    # Clean error message (no traceback)
                    error_msg = str(e).split('\n')[0]
                    print(f"Error: {error_msg}")
                    return

        # Check if URI matches a resource template pattern
        if hasattr(conn, "resource_templates"):
            for template in conn.resource_templates:
                template_uri = str(template.uriTemplate) if hasattr(template, "uriTemplate") else ""
                # Simple check: if the URI scheme matches the template scheme
                if template_uri and uri.split("/")[0] == template_uri.split("/")[0]:
                    try:
                        result = run_async(conn.session.read_resource(uri))
                        print(f"\n{uri}:")
                        for content in result.contents:
                            if hasattr(content, "text"):
                                print(content.text)
                            else:
                                print(str(content))
                        return
                    except Exception as e:
                        # Clean error message (no traceback)
                        error_msg = str(e).split('\n')[0]
                        print(f"Error: {error_msg}")
                        return

    print(f"Resource not found: {uri}")
    print("Use '/mcp resources' to list available resources.")


def _cmd_prompt(engine, args: str) -> None:
    """Get a specific prompt with arguments."""
    if not args:
        print("Usage: /mcp prompt <name> [key=value ...]")
        print("Example: /mcp prompt summarize text=\"Hello world\"")
        return

    mcp_client = getattr(engine, "_mcp_client", None)
    if not mcp_client:
        print("No MCP client initialized.")
        return

    from l3m_backend.mcp.client.client import run_async
    import shlex

    # Parse args: first word is prompt name, rest are key=value pairs
    try:
        parts = shlex.split(args)
    except ValueError:
        parts = args.split()

    prompt_name = parts[0]
    prompt_args = {}

    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            prompt_args[key] = value

    # Try to find the prompt in connected servers
    for server_name in mcp_client.connected_servers:
        conn = mcp_client.get_connection(server_name)
        if not conn:
            continue

        # Check if this server has the prompt
        for prompt in conn.prompts:
            if prompt.name == prompt_name:
                try:
                    result = run_async(conn.session.get_prompt(prompt_name, arguments=prompt_args))
                    print(f"\n{prompt_name}:")
                    for msg in result.messages:
                        role = msg.role if hasattr(msg, "role") else "user"
                        content = msg.content
                        if hasattr(content, "text"):
                            content = content.text
                        print(f"[{role}] {content}")
                    return
                except Exception as e:
                    # Clean error message (no traceback)
                    error_msg = str(e).split('\n')[0]
                    print(f"Error: {error_msg}")
                    return

    print(f"Prompt not found: {prompt_name}")
    print("Use '/mcp prompts' to list available prompts.")


def _cmd_add(args: str) -> None:
    """Add an MCP server configuration."""
    if not args:
        print("Usage: /mcp add <name> <command|url>")
        print()
        print("Examples:")
        print("  /mcp add fs npx -y @modelcontextprotocol/server-filesystem /")
        print("  /mcp add remote http://localhost:8080/sse")
        return

    parts = args.split(None, 1)
    if len(parts) < 2:
        print("Usage: /mcp add <name> <command|url>")
        return

    name = parts[0]
    value = parts[1]

    from l3m_backend.mcp import get_mcp_config_manager
    mgr = get_mcp_config_manager()

    if value.startswith("http://") or value.startswith("https://"):
        # HTTP/SSE server
        mgr.add_server(name=name, transport="sse", url=value)
        print(f"Added MCP server: {name}")
        print(f"  URL: {value}")
    else:
        # stdio server
        cmd_parts = value.split()
        command = cmd_parts[0]
        cmd_args = cmd_parts[1:] if len(cmd_parts) > 1 else []
        mgr.add_server(name=name, transport="stdio", command=command, args=cmd_args)
        print(f"Added MCP server: {name}")
        print(f"  Command: {value}")

    print(f"Connect with: /mcp connect {name}")


def _cmd_remove(name: str) -> None:
    """Remove an MCP server configuration."""
    if not name:
        print("Usage: /mcp remove <name>")
        return

    from l3m_backend.mcp import get_mcp_config_manager

    mgr = get_mcp_config_manager()
    if mgr.remove_server(name):
        print(f"Removed MCP server: {name}")
    else:
        print(f"Server not found: {name}")


def _cmd_auto(name: str) -> None:
    """Toggle auto-connect for an MCP server."""
    if not name:
        print("Usage: /mcp auto <name>")
        return

    from l3m_backend.mcp import get_mcp_config_manager

    mgr = get_mcp_config_manager()
    config = mgr.load()

    if name not in config.servers:
        print(f"Server not found: {name}")
        return

    # Toggle auto_connect
    server = config.servers[name]
    server.auto_connect = not server.auto_connect
    mgr.save()

    status = "enabled" if server.auto_connect else "disabled"
    print(f"Auto-connect {status} for: {name}")
