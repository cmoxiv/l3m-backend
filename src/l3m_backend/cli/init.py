#!/usr/bin/env python3
"""
CLI entry point for initializing l3m-backend (l3m-init command).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _install_builtins(base_dir: Path, force: bool = False, quiet: bool = False) -> list[str]:
    """Install package builtins as symlinks to user directory.

    Creates symlinks from ~/.l3m/{commands,magic,tools}/ to package builtins.
    Users can:
    - Remove symlinks to disable built-ins
    - Replace symlinks with custom implementations

    Args:
        base_dir: Base ~/.l3m directory
        force: Overwrite existing files/symlinks
        quiet: Suppress output

    Returns:
        List of installed paths
    """
    installed = []

    # Get package builtin directories
    from l3m_backend.cli.commands.loader import PACKAGE_BUILTINS_DIR as CMD_BUILTINS
    from l3m_backend.cli.magic.loader import PACKAGE_BUILTINS_DIR as MAGIC_BUILTINS
    from l3m_backend.tools.loader import PACKAGE_TOOLS_DIR

    # Install command builtins as symlinks
    user_commands = base_dir / "commands"
    if CMD_BUILTINS.exists():
        for cmd_dir in CMD_BUILTINS.iterdir():
            if not cmd_dir.is_dir() or cmd_dir.name.startswith((".", "_")):
                continue
            user_cmd = user_commands / cmd_dir.name
            if user_cmd.exists():
                if not force:
                    continue  # Don't overwrite existing user customizations
                # Remove existing (file, symlink, or directory)
                if user_cmd.is_symlink() or user_cmd.is_file():
                    user_cmd.unlink()
                else:
                    shutil.rmtree(user_cmd)
            # Create symlink to package builtin
            user_cmd.symlink_to(cmd_dir)
            installed.append(f"{user_cmd} -> {cmd_dir}")

    # Install magic builtins as symlinks
    user_magic = base_dir / "magic"
    if MAGIC_BUILTINS.exists():
        for magic_dir in MAGIC_BUILTINS.iterdir():
            if not magic_dir.is_dir() or magic_dir.name.startswith((".", "_")):
                continue
            user_mgc = user_magic / magic_dir.name
            if user_mgc.exists():
                if not force:
                    continue  # Don't overwrite existing user customizations
                # Remove existing (file, symlink, or directory)
                if user_mgc.is_symlink() or user_mgc.is_file():
                    user_mgc.unlink()
                else:
                    shutil.rmtree(user_mgc)
            # Create symlink to package builtin
            user_mgc.symlink_to(magic_dir)
            installed.append(f"{user_mgc} -> {magic_dir}")

    # Install tool builtins as symlinks (directories only, like commands/magic)
    user_tools = base_dir / "tools"
    if PACKAGE_TOOLS_DIR.exists():
        for tool_dir in PACKAGE_TOOLS_DIR.iterdir():
            # Only process directories with __init__.py (modules)
            if not tool_dir.is_dir() or tool_dir.name.startswith((".", "_")):
                continue
            if not (tool_dir / "__init__.py").exists():
                continue
            user_tool = user_tools / tool_dir.name
            if user_tool.exists():
                if not force:
                    continue  # Don't overwrite existing user customizations
                # Remove existing (file, symlink, or directory)
                if user_tool.is_symlink() or user_tool.is_file():
                    user_tool.unlink()
                else:
                    shutil.rmtree(user_tool)
            # Create symlink to package builtin
            user_tool.symlink_to(tool_dir)
            installed.append(f"{user_tool} -> {tool_dir}")

    return installed


def main():
    """Main entry point for the l3m-init CLI."""
    parser = argparse.ArgumentParser(
        description="Initialize l3m-backend configuration and directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This command creates:
  ~/.l3m/              - Main configuration directory
  ~/.l3m/config.json   - Configuration file with default settings
  ~/.l3m/models/       - Directory for GGUF model files
  ~/.l3m/sessions/     - Directory for chat session files
  ~/.l3m/tools/        - Directory for user-defined tools
  ~/.l3m/commands/     - Directory for / commands (with builtins installed)
  ~/.l3m/magic/        - Directory for % magic commands (with builtins installed)

Examples:
    l3m-init              # Initialize with defaults
    l3m-init --force      # Overwrite existing config and builtins
        """,
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing configuration and builtins"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()

    # Import here to avoid circular imports
    from l3m_backend.config import get_config_manager

    base_dir = Path.home() / ".l3m"
    models_dir = base_dir / "models"
    sessions_dir = base_dir / "sessions"
    tools_dir = base_dir / "tools"
    commands_dir = base_dir / "commands"
    magic_dir = base_dir / "magic"

    cfg_mgr = get_config_manager()
    config_file = cfg_mgr.CONFIG_FILE

    # Track what was created
    created = []
    skipped = []

    # Create directories
    for dir_path in [base_dir, models_dir, sessions_dir, tools_dir, commands_dir, magic_dir]:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created.append(str(dir_path))
        else:
            skipped.append(str(dir_path))

    # Create/replace config file (always overwrite)
    existed = config_file.exists()
    cfg_mgr._create_default_config()
    if existed:
        created.append(f"{config_file} (replaced)")
    else:
        created.append(str(config_file))

    # Install builtins to user directories
    installed = _install_builtins(base_dir, force=args.force, quiet=args.quiet)
    if installed:
        created.extend(installed)

    # Output results
    if not args.quiet:
        if created:
            print("Created:")
            for path in created:
                print(f"  {path}")

        if skipped:
            print("\nAlready exists (skipped):")
            for path in skipped:
                print(f"  {path}")

        print("\nInitialization complete!")
        print("\nNext steps:")
        print("  1. Download a model:  l3m-download --preset llama3.2-1b")
        print("  2. Configure defaults: l3m-chat --set-config default_model=<model>")
        print("  3. Start chatting:    l3m-chat")
        print("\nCustomize commands: Edit files in ~/.l3m/commands/")
        print("Customize magic:    Edit files in ~/.l3m/magic/")


if __name__ == "__main__":
    main()
