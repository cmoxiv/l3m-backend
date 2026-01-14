"""
Tools loader - discovers and loads tools from ~/.l3m/tools/

Tools are loaded ONLY from ~/.l3m/tools/. Use `l3m-init` to create
symlinks to package builtins. Users can:
- Remove symlinks to disable built-in tools
- Replace symlinks with custom implementations

Each tool must be a module (directory with __init__.py), matching the
pattern used by commands and magic.

The tool registers itself using the same decorators as built-in tools:

    # ~/.l3m/tools/my_tool/__init__.py
    from l3m_backend.tools._registry import registry
    from l3m_backend.core import tool_output

    @registry.register(aliases=["mt"])
    @tool_output(llm_format="{result}")
    def my_tool(input: str) -> dict:
        return {"result": f"Processed: {input}"}
"""

from __future__ import annotations

import logging
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

logger = logging.getLogger(__name__)

# Default user tools directory
USER_TOOLS_DIR = Path.home() / ".l3m" / "tools"

# Package builtins directory
PACKAGE_TOOLS_DIR = Path(__file__).parent


def discover_tools(tools_dir: Path | None = None) -> list[Path]:
    """
    Discover tool modules in the given directory.

    Each tool must be a directory with __init__.py (module), matching the
    pattern used by commands and magic.

    Args:
        tools_dir: Directory to search (default: ~/.l3m/tools)

    Returns:
        List of __init__.py paths for valid tools.
    """
    tools_dir = tools_dir or USER_TOOLS_DIR

    if not tools_dir.exists():
        return []

    if not tools_dir.is_dir():
        logger.warning(f"Tools path is not a directory: {tools_dir}")
        return []

    tool_paths = []
    for item in sorted(tools_dir.iterdir()):
        # Skip non-directories, hidden, and private
        if not item.is_dir() or item.name.startswith((".", "_")):
            continue

        init_file = item / "__init__.py"
        if init_file.exists():
            tool_paths.append(init_file)
        else:
            logger.debug(f"Skipping {item.name}: no __init__.py")

    return tool_paths


# Backwards compatibility alias
discover_user_tools = discover_tools


def load_tool(tool_path: Path, prefix: str = "l3m_tool") -> tuple[str, bool, str]:
    """
    Load a single tool module.

    Args:
        tool_path: Path to the tool's __init__.py file.
        prefix: Module name prefix for sys.modules

    Returns:
        Tuple of (tool_name, success, error_message)
    """
    tool_name = tool_path.parent.name
    module_name = f"{prefix}.{tool_name}"

    try:
        spec = spec_from_file_location(module_name, tool_path)
        if spec is None or spec.loader is None:
            return (tool_name, False, "Could not create module spec")

        module = module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return (tool_name, True, "")

    except SyntaxError as e:
        return (tool_name, False, f"Syntax error: {e}")
    except ImportError as e:
        return (tool_name, False, f"Import error: {e}")
    except Exception as e:
        return (tool_name, False, f"Error: {e}")


# Backwards compatibility alias
load_user_tool = load_tool


def load_all_tools(tools_dir: Path | None = None, verbose: bool = False) -> int:
    """
    Load all tools from the tools directory.

    Tools are loaded ONLY from ~/.l3m/tools/. Use `l3m-init` to create
    symlinks to package builtins. Users can:
    - Remove symlinks to disable built-in tools
    - Replace symlinks with custom implementations

    Args:
        tools_dir: Directory to load from (default: ~/.l3m/tools)
        verbose: Log info messages for successful loads

    Returns:
        Number of successfully loaded tools.
    """
    tools_dir = tools_dir or USER_TOOLS_DIR
    tool_paths = discover_tools(tools_dir)

    if not tool_paths:
        return 0

    loaded = 0
    for tool_path in tool_paths:
        tool_name, success, error = load_tool(tool_path)

        if success:
            loaded += 1
            if verbose:
                logger.info(f"Loaded tool: {tool_name}")
        else:
            logger.warning(f"Failed to load tool '{tool_name}': {error}")

    return loaded


# Backwards compatibility alias
load_user_tools = load_all_tools
