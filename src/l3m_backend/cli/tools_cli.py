#!/usr/bin/env python3
"""
CLI entry point for listing, inspecting, and creating tools (l3m-tools command).
"""

import argparse
import json
import sys


def parse_arg_spec(spec: str) -> tuple[str, str, str | None, str | None]:
    """Parse argument specification: name:type=default:description.

    Examples:
        "n:int=10:Number of items"  -> ("n", "int", "10", "Number of items")
        "query:str:Search query"    -> ("query", "str", None, "Search query")
        "verbose:bool=false"        -> ("verbose", "bool", "false", None)
        "path:str"                  -> ("path", "str", None, None)

    Returns:
        Tuple of (name, type, default, description).
    """
    parts = spec.split(":", 2)  # Split into max 3 parts

    if len(parts) < 2:
        raise ValueError(f"Invalid arg spec '{spec}': must be name:type[=default][:description]")

    name = parts[0].strip()
    type_and_default = parts[1].strip()
    description = parts[2].strip() if len(parts) > 2 else None

    # Parse type and optional default
    if "=" in type_and_default:
        arg_type, default = type_and_default.split("=", 1)
        arg_type = arg_type.strip()
        default = default.strip()
    else:
        arg_type = type_and_default
        default = None

    # Validate type
    valid_types = ("str", "int", "float", "bool")
    if arg_type not in valid_types:
        raise ValueError(f"Invalid type '{arg_type}': must be one of {valid_types}")

    return (name, arg_type, default, description)


def cmd_list(args):
    """List all registered tools."""
    from l3m_backend.tools import registry

    if args.as_json:
        tools = [
            {
                "name": e.name,
                "aliases": e.aliases,
                "description": e.get_description(),
            }
            for e in registry
        ]
        print(json.dumps(tools, indent=2))
    else:
        print("\nAvailable tools:")
        for entry in registry:
            aliases = f" (aliases: {', '.join(entry.aliases)})" if entry.aliases else ""
            print(f"  - {entry.name}{aliases}")
            print(f"    {entry.get_description()}")
        print()


def cmd_schema(args):
    """Output OpenAI-style tool schema."""
    from l3m_backend.tools import registry

    print(json.dumps(registry.to_openai_tools(), indent=2))


def cmd_info(args):
    """Show detailed info for a specific tool."""
    from l3m_backend.tools import registry

    try:
        entry = registry.get(args.tool)
        spec = entry.to_openai_spec()
        if args.as_json:
            print(json.dumps(spec, indent=2))
        else:
            print(f"\nTool: {entry.name}")
            print(f"Aliases: {', '.join(entry.aliases) or 'none'}")
            print(f"Description: {entry.get_description()}")
            print("\nParameters:")
            params = spec["function"]["parameters"]
            for name, prop in params.get("properties", {}).items():
                required = name in params.get("required", [])
                req_marker = " (required)" if required else ""
                print(f"  - {name}: {prop.get('type', 'any')}{req_marker}")
                if "description" in prop:
                    print(f"    {prop['description']}")
            print()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_create(args):
    """Create a new user tool scaffold."""
    from l3m_backend.tools.generator import generate_tool

    # Parse --arg options into list of (name, type, default, description)
    tool_args = []
    if args.arg:
        for arg_spec in args.arg:
            tool_args.append(parse_arg_spec(arg_spec))

    success, message, path = generate_tool(
        name=args.name,
        wrap_command=args.wrap,
        tool_args=tool_args,
        force=args.force,
    )

    if success:
        print(message)
        print(f"\nEdit the tool at: {path / '__init__.py'}")
        print("\nRestart l3m-chat to use your new tool.")
    else:
        print(f"Error: {message}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the l3m-tools CLI."""
    parser = argparse.ArgumentParser(
        description="List, inspect, and create tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    l3m-tools list              List all available tools
    l3m-tools schema            Output OpenAI-style tool schema
    l3m-tools info get_weather  Show details for a specific tool
    l3m-tools create my_tool    Create a new user tool
    l3m-tools create git-log --wrap "git log --oneline -10"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List all registered tools")
    list_parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Output as JSON"
    )
    list_parser.set_defaults(func=cmd_list)

    # schema command
    schema_parser = subparsers.add_parser(
        "schema", help="Output OpenAI-style tool schema"
    )
    schema_parser.set_defaults(func=cmd_schema, as_json=False)

    # info command
    info_parser = subparsers.add_parser("info", help="Show details for a specific tool")
    info_parser.add_argument("tool", help="Tool name or alias")
    info_parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Output as JSON"
    )
    info_parser.set_defaults(func=cmd_info)

    # create command
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new user tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    l3m-tools create my_tool
        Create a basic tool scaffold at ~/.l3m/tools/my_tool/

    l3m-tools create docker-ps --wrap "docker ps"
        Create a tool that wraps the 'docker ps' command

    l3m-tools create git-log --wrap "git log" --arg "n:int=10:Number of commits"
        Create a wrapper with a custom argument

    l3m-tools create search --wrap "grep -r" --arg "pattern:str:Search pattern" --arg "path:str=.:Directory"
        Create a wrapper with multiple arguments
        """,
    )
    create_parser.add_argument("name", help="Tool name")
    create_parser.add_argument(
        "--wrap", metavar="COMMAND", help="Create wrapper for shell command"
    )
    create_parser.add_argument(
        "--arg", metavar="SPEC", action="append",
        help="Add argument: name:type[=default][:description] (can repeat)"
    )
    create_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing tool"
    )
    create_parser.set_defaults(func=cmd_create)

    args = parser.parse_args()

    # Default to 'list' if no command given
    if args.command is None:
        args.command = "list"
        args.as_json = False
        args.func = cmd_list

    args.func(args)


if __name__ == "__main__":
    main()
