"""
Command loader - discovers and loads commands from user dir.

Commands are loaded ONLY from ~/.l3m/commands/. Use `l3m-init` to create
symlinks to package builtins. Users can:
- Remove symlinks to disable built-in commands
- Replace symlinks with custom implementations

Each command must be in its own subdirectory with an __init__.py file.
The command registers itself using the command_registry decorator:

    # ~/.l3m/commands/my_cmd/__init__.py
    from l3m_backend.cli.commands import command_registry

    @command_registry.register("mycmd", "My custom command")
    def cmd_mycmd(engine, session_mgr, args):
        print("Hello from my command!")
"""

from __future__ import annotations

import logging
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

logger = logging.getLogger(__name__)

# Default user commands directory
USER_COMMANDS_DIR = Path.home() / ".l3m" / "commands"

# Package builtins directory
PACKAGE_BUILTINS_DIR = Path(__file__).parent / "builtins"


def discover_commands(commands_dir: Path) -> list[Path]:
    """
    Discover command directories in the given path.

    Each command must be in its own subdirectory with an __init__.py file.

    Args:
        commands_dir: Directory to search

    Returns:
        List of __init__.py paths for valid commands.
    """
    if not commands_dir.exists():
        return []

    if not commands_dir.is_dir():
        logger.warning(f"Commands path is not a directory: {commands_dir}")
        return []

    cmd_paths = []
    for subdir in sorted(commands_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Skip hidden and private directories
        if subdir.name.startswith((".", "_")):
            continue
        init_file = subdir / "__init__.py"
        if init_file.exists():
            cmd_paths.append(init_file)
        else:
            logger.debug(f"Skipping {subdir.name}: no __init__.py")

    return cmd_paths


def load_command(cmd_path: Path, prefix: str = "l3m_cmd") -> tuple[str, bool, str]:
    """
    Load a single command module.

    Args:
        cmd_path: Path to the command's __init__.py file.
        prefix: Module name prefix for sys.modules

    Returns:
        Tuple of (cmd_name, success, error_message)
    """
    cmd_name = cmd_path.parent.name
    module_name = f"{prefix}.{cmd_name}"

    try:
        spec = spec_from_file_location(module_name, cmd_path)
        if spec is None or spec.loader is None:
            return (cmd_name, False, "Could not create module spec")

        module = module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return (cmd_name, True, "")

    except SyntaxError as e:
        return (cmd_name, False, f"Syntax error: {e}")
    except ImportError as e:
        return (cmd_name, False, f"Import error: {e}")
    except Exception as e:
        return (cmd_name, False, f"Error: {e}")


def load_all_commands(user_dir: Path | None = None, verbose: bool = False) -> int:
    """
    Load all commands from user directory.

    Commands are loaded ONLY from ~/.l3m/commands/. Use `l3m-init` to create
    symlinks to package builtins. Users can:
    - Remove symlinks to disable built-in commands
    - Replace symlinks with custom implementations

    Args:
        user_dir: User commands directory (default: ~/.l3m/commands)
        verbose: Log info messages for successful loads

    Returns:
        Number of successfully loaded commands.
    """
    user_dir = user_dir or USER_COMMANDS_DIR
    total_loaded = 0

    # Load commands from user directory only (symlinks or custom)
    for cmd_path in discover_commands(user_dir):
        cmd_name, success, error = load_command(cmd_path, prefix="l3m_cmd")

        if success:
            total_loaded += 1
            if verbose:
                logger.info(f"Loaded command: {cmd_name}")
        else:
            logger.warning(f"Failed to load command '{cmd_name}': {error}")

    return total_loaded


# Backwards compatibility aliases
discover_user_commands = discover_commands
load_user_command = load_command
load_user_commands = load_all_commands
