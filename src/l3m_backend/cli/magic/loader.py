"""
Magic command loader - discovers and loads magic from user dir.

Magic commands are loaded ONLY from ~/.l3m/magic/. Use `l3m-init` to create
symlinks to package builtins. Users can:
- Remove symlinks to disable built-in magic commands
- Replace symlinks with custom implementations

Each magic command must be in its own subdirectory with an __init__.py file.
The command registers itself using the magic_registry decorator:

    # ~/.l3m/magic/greet/__init__.py
    from l3m_backend.cli.magic import magic_registry

    @magic_registry.register("greet", "Greet someone")
    def magic_greet(engine, session_mgr, args, add_to_history):
        output = f"Hello, {args or 'world'}!"
        print(output)
        add_to_history(f"%greet {args}", output)
        return True
"""

from __future__ import annotations

import logging
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

logger = logging.getLogger(__name__)

# Default user magic commands directory
USER_MAGIC_DIR = Path.home() / ".l3m" / "magic"

# Package builtins directory
PACKAGE_BUILTINS_DIR = Path(__file__).parent / "builtins"


def discover_magic(magic_dir: Path) -> list[Path]:
    """
    Discover magic command directories in the given path.

    Each command must be in its own subdirectory with an __init__.py file.

    Args:
        magic_dir: Directory to search

    Returns:
        List of __init__.py paths for valid magic commands.
    """
    if not magic_dir.exists():
        return []

    if not magic_dir.is_dir():
        logger.warning(f"Magic path is not a directory: {magic_dir}")
        return []

    magic_paths = []
    for subdir in sorted(magic_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Skip hidden and private directories
        if subdir.name.startswith((".", "_")):
            continue
        init_file = subdir / "__init__.py"
        if init_file.exists():
            magic_paths.append(init_file)
        else:
            logger.debug(f"Skipping {subdir.name}: no __init__.py")

    return magic_paths


def load_magic_command(magic_path: Path, prefix: str = "l3m_magic") -> tuple[str, bool, str]:
    """
    Load a single magic command module.

    Args:
        magic_path: Path to the magic command's __init__.py file.
        prefix: Module name prefix for sys.modules

    Returns:
        Tuple of (cmd_name, success, error_message)
    """
    cmd_name = magic_path.parent.name
    module_name = f"{prefix}.{cmd_name}"

    try:
        spec = spec_from_file_location(module_name, magic_path)
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


def load_all_magic(user_dir: Path | None = None, verbose: bool = False) -> int:
    """
    Load all magic commands from user directory.

    Magic commands are loaded ONLY from ~/.l3m/magic/. Use `l3m-init` to create
    symlinks to package builtins. Users can:
    - Remove symlinks to disable built-in magic commands
    - Replace symlinks with custom implementations

    Args:
        user_dir: User magic directory (default: ~/.l3m/magic)
        verbose: Log info messages for successful loads

    Returns:
        Number of successfully loaded magic commands.
    """
    user_dir = user_dir or USER_MAGIC_DIR
    total_loaded = 0

    # Load magic from user directory only (symlinks or custom)
    for magic_path in discover_magic(user_dir):
        cmd_name, success, error = load_magic_command(magic_path, prefix="l3m_magic")

        if success:
            total_loaded += 1
            if verbose:
                logger.info(f"Loaded magic: {cmd_name}")
        else:
            logger.warning(f"Failed to load magic '{cmd_name}': {error}")

    return total_loaded


# Backwards compatibility aliases
discover_user_magic = discover_magic
load_user_magic_command = load_magic_command
load_user_magic = load_all_magic
