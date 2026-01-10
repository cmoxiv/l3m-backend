"""Example MCP server running over HTTP/SSE for testing l3m's MCP client.

This server provides sample tools, resources, and prompts over HTTP.

Run with:
    python -m examples.mcp_server_http

Connect from l3m-chat:
    /mcp connect remote http://localhost:8080/sse
"""
from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "MCP package not installed. Install with: pip install 'mcp[cli]'"
    )

# Create the MCP server
mcp = FastMCP("remote")


# ============================================================================
# Tools
# ============================================================================


@mcp.tool()
def echo(message: str) -> str:
    """Echo back the input message."""
    return f"Echo: {message}"


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def get_time() -> str:
    """Get the current server time."""
    from datetime import datetime
    return datetime.now().isoformat()


# ============================================================================
# Resources
# ============================================================================


@mcp.resource("http://info")
def get_server_info() -> str:
    """Get server information."""
    return """
l3m HTTP MCP Server
===================
Running on http://localhost:8080/sse

Available tools:
- echo: Echo a message back
- add_numbers: Add two numbers
- get_time: Get server time

Available resources:
- <server-name>:///<filename>: Read local files
"""


# Dynamically register file resources for files in working directory
def _register_file_resources():
    """Register a resource for each file in the working directory (first level only)."""
    from pathlib import Path

    cwd = Path.cwd()

    # Glob common file patterns (first level only)
    patterns = ["*.py", "*.md", "*.txt", "*.json", "*.yaml", "*.yml", "*.toml", "*.cfg", "*.ini"]

    registered = set()
    for pattern in patterns:
        for file_path in cwd.glob(pattern):
            if file_path.is_file() and file_path.name not in registered:
                registered.add(file_path.name)
                _register_single_file(file_path.name)


def _register_single_file(filename: str):
    """Register a single file as an MCP resource."""
    from pathlib import Path
    from pydantic import AnyUrl
    from mcp.server.fastmcp.resources import FunctionResource

    # Use server name as URI scheme
    server_name = mcp.name
    uri_str = f"{server_name}:///{filename}"

    def make_reader(fname):
        def read_file() -> str:
            file_path = Path.cwd() / fname
            if not file_path.exists():
                return f"Error: File not found: {fname}"
            try:
                return file_path.read_text()
            except Exception as e:
                return f"Error reading file: {e}"
        return read_file

    # Create and register the resource directly
    resource = FunctionResource(
        uri=AnyUrl(uri_str),
        name=filename,
        description=f"Content of {filename}",
        fn=make_reader(filename),
    )
    mcp._resource_manager.add_resource(resource)


# Register file resources at module load
_register_file_resources()


# ============================================================================
# Main
# ============================================================================


def main():
    """Run the MCP server over HTTP/SSE."""
    import uvicorn
    from mcp.server.sse import SseServerTransport

    sse = SseServerTransport("/messages/")

    async def handle_sse(scope, receive, send):
        """Handle SSE connections."""
        async with sse.connect_sse(scope, receive, send) as streams:
            await mcp._mcp_server.run(
                streams[0], streams[1], mcp._mcp_server.create_initialization_options()
            )

    async def handle_messages(scope, receive, send):
        """Handle POST messages from client."""
        await sse.handle_post_message(scope, receive, send)

    async def homepage(scope, receive, send):
        """Serve homepage."""
        body = b"""
        <html>
        <head><title>MCP Server</title></head>
        <body>
            <h1>l3m HTTP MCP Server</h1>
            <p>Connect via SSE at: <code>/sse</code></p>
            <p>From l3m-chat:</p>
            <pre>/mcp connect remote http://localhost:8080/sse</pre>
        </body>
        </html>
        """
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [[b"content-type", b"text/html"]],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })

    async def app(scope, receive, send):
        """ASGI application."""
        if scope["type"] != "http":
            return

        path = scope["path"]
        method = scope["method"]

        if path == "/":
            await homepage(scope, receive, send)
        elif path == "/sse":
            await handle_sse(scope, receive, send)
        elif path == "/messages/" and method == "POST":
            await handle_messages(scope, receive, send)
        else:
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({
                "type": "http.response.body",
                "body": b"Not Found",
            })

    print("Starting MCP HTTP server on http://localhost:8080")
    print("Connect with: /mcp connect remote http://localhost:8080/sse")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")


if __name__ == "__main__":
    main()
