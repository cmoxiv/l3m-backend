"""Example MCP server for testing l3m's MCP client implementation.

This server provides sample tools, resources, and prompts for testing.

Run with:
    python -m examples.mcp_server

Or with MCP dev tools:
    mcp dev examples/mcp_server/__init__.py

Connect from l3m-chat:
    l3m-chat --mcp-add test --command "python -m examples.mcp_server"
    l3m-chat --mcp test
"""
from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp import Context
except ImportError:
    raise ImportError(
        "MCP package not installed. Install with: pip install 'mcp[cli]'"
    )

# Create the MCP server
mcp = FastMCP("test")


# ============================================================================
# Tools
# ============================================================================


@mcp.tool()
def echo(message: str) -> str:
    """Echo back the input message.

    Args:
        message: The message to echo back
    """
    return f"Echo: {message}"


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return a + b


@mcp.tool()
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return a * b


@mcp.tool()
def reverse_string(text: str) -> str:
    """Reverse a string.

    Args:
        text: The string to reverse
    """
    return text[::-1]


@mcp.tool()
def count_words(text: str) -> dict:
    """Count words in a text.

    Args:
        text: The text to analyze
    """
    words = text.split()
    return {
        "total_words": len(words),
        "unique_words": len(set(words)),
        "character_count": len(text),
    }


@mcp.tool()
def score(n: float, b: float) -> float:
    """Calculate the logarithm of n to base b.

    Args:
        n: The number to take the logarithm of
        b: The base of the logarithm
    """
    import math
    if n <= 0:
        raise ValueError("n must be positive")
    if b <= 0 or b == 1:
        raise ValueError("b must be positive and not equal to 1")
    return math.log(n) / math.log(b)


# ============================================================================
# Resources
# ============================================================================


@mcp.resource("test://greeting/{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting.

    Args:
        name: Name to greet
    """
    return f"Hello, {name}! Welcome to the l3m test MCP server."


@mcp.resource("test://info")
def get_server_info() -> str:
    """Get server information."""
    return """
l3m Test MCP Server
===================
This is an example MCP server for testing l3m's MCP client implementation.

Available tools:
- echo: Echo a message back
- add_numbers: Add two numbers
- multiply_numbers: Multiply two numbers
- reverse_string: Reverse a string
- count_words: Count words in text
- score: Calculate log base b of n

Available resources:
- test://greeting/{name}: Get a personalized greeting
- test://info: Get this server info
- file://{path}: Read a local file

Available prompts:
- summarize: Generate a summary prompt
- analyze: Generate an analysis prompt
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
# Prompts
# ============================================================================


@mcp.prompt()
def title(text: str, ctx: Context) -> str:
    """Generate a prompt for summarizing text.
    Args:
        text: The text to summarize
    """
    # print(ctx)
    return f"""Please generate a title that represents the {text}."""



@mcp.prompt()
def summarize(text: str) -> str:
    """Generate a prompt for summarizing text.

    Args:
        text: The text to summarize
    """
    return f"""Please summarize the following text in a concise manner:

{text}

Provide a summary that captures the main points."""


@mcp.prompt()
def analyze(topic: str) -> str:
    """Generate a prompt for analyzing a topic.

    Args:
        topic: The topic to analyze
    """
    return f"""Please provide a detailed analysis of the following topic:

Topic: {topic}

Include:
1. Key concepts and definitions
2. Main considerations
3. Potential implications
4. Recommendations if applicable"""


# ============================================================================
# Main
# ============================================================================


def main():
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
