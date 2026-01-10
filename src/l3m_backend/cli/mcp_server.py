#!/usr/bin/env python3
"""CLI entry point for running l3m as an MCP server (l3m-mcp-server command).

This exposes l3m's tools as an MCP server that can be used by MCP clients
like Claude Desktop.
"""
from __future__ import annotations

import argparse
import sys


def main():
    """Main entry point for the l3m-mcp-server CLI."""
    parser = argparse.ArgumentParser(
        description="Run l3m tools as an MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    l3m-mcp-server                    # Run with stdio transport (default)
    l3m-mcp-server --transport sse    # Run with SSE transport
    l3m-mcp-server --list-tools       # List available tools

To use with Claude Desktop, add to your MCP settings:
    {
      "mcpServers": {
        "l3m-tools": {
          "command": "l3m-mcp-server"
        }
      }
    }
        """,
    )
    parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport type (default: stdio)"
    )
    parser.add_argument(
        "--name", "-n",
        default="l3m-tools",
        help="Server name (default: l3m-tools)"
    )
    parser.add_argument(
        "--list-tools", "-l",
        action="store_true",
        help="List available tools and exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    try:
        # Import here to avoid loading everything for --help
        from l3m_backend.tools import registry
        from l3m_backend.tools.loader import load_user_tools

        # Load user tools
        load_user_tools()

        if args.list_tools:
            print("Available tools:")
            for entry in registry:
                aliases = f" (aliases: {', '.join(entry.aliases)})" if entry.aliases else ""
                print(f"  {entry.name}{aliases}")
                if entry.description:
                    print(f"    {entry.description}")
            return

        # Check if MCP is installed
        try:
            from l3m_backend.mcp import create_mcp_server
        except ImportError:
            print("Error: MCP support not installed.", file=sys.stderr)
            print("Install with: pip install 'l3m-backend[mcp]'", file=sys.stderr)
            sys.exit(1)

        # Create and run the server
        server = create_mcp_server(registry=registry, name=args.name)
        server.run(transport=args.transport)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
