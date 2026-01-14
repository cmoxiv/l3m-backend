"""Command command - list/edit/create commands."""
from __future__ import annotations

import os
import shlex
import subprocess
from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry
from l3m_backend.cli.commands.loader import USER_COMMANDS_DIR

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


# Basic command scaffold template
BASIC_SCAFFOLD = '''"""{cmd_name} command - description."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("{cmd_name}", "Description")
def cmd_{func_name}(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Description."""
    # Your code goes here
    pass
'''

# Shell wrapper command scaffold template
WRAPPER_SCAFFOLD = '''"""{cmd_name} command - runs: {shell_cmd}"""
from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("{cmd_name}", "{description}")
def cmd_{func_name}(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Run: {shell_cmd}"""
    try:
        result = subprocess.run(
            """{shell_cmd}""",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("Command timed out after 30 seconds")
    except Exception as e:
        print(f"Error: {{e}}")
'''


@command_registry.register("command", "Commands (ls/edit/new)", usage="/command [ls | edit <name> | new <name> [\"shell cmd\"]]", has_args=True)
def cmd_command(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """List, edit, or create commands."""
    parts = args.split(maxsplit=1) if args else []
    subcommand = parts[0] if parts else "ls"

    if subcommand == "ls":
        # /command or /command ls - list commands
        print(f"\nCommands directory: {USER_COMMANDS_DIR}")
        if USER_COMMANDS_DIR.exists():
            for cmd_dir in sorted(USER_COMMANDS_DIR.iterdir()):
                if cmd_dir.is_dir() and not cmd_dir.name.startswith((".", "_")):
                    is_symlink = cmd_dir.is_symlink()
                    marker = " -> builtin" if is_symlink else " (custom)"
                    print(f"  {cmd_dir.name}{marker}")
        print()
    elif subcommand == "edit" and len(parts) >= 2:
        # /command edit <name>
        cmd_name = parts[1].strip()
        cmd_path = USER_COMMANDS_DIR / cmd_name / "__init__.py"
        if cmd_path.exists():
            editor = os.environ.get("EDITOR", "vim")
            subprocess.run([editor, str(cmd_path)])
            print(f"Restart l3m-chat to reload {cmd_name}.\n")
        else:
            print(f"Command not found: {cmd_name}\n")
    elif subcommand == "new" and len(parts) >= 2:
        # /command new <name> ["shell command"]
        # Parse: name and optional quoted shell command
        rest = parts[1]
        try:
            tokens = shlex.split(rest)
        except ValueError:
            tokens = rest.split()

        cmd_name = tokens[0] if tokens else ""
        shell_cmd = tokens[1] if len(tokens) > 1 else None

        if not cmd_name:
            print("Usage: /command new <name> [\"shell command\"]\n")
            return

        cmd_dir = USER_COMMANDS_DIR / cmd_name
        cmd_path = cmd_dir / "__init__.py"
        if cmd_path.exists():
            print(f"Command already exists: {cmd_name}\n")
        else:
            cmd_dir.mkdir(parents=True, exist_ok=True)
            func_name = cmd_name.replace("-", "_")

            if shell_cmd:
                # Generate shell wrapper scaffold
                # Extract short description from command
                desc_cmd = shell_cmd.split()[0] if shell_cmd else cmd_name
                description = f"Run {desc_cmd}"
                scaffold = WRAPPER_SCAFFOLD.format(
                    cmd_name=cmd_name,
                    func_name=func_name,
                    shell_cmd=shell_cmd,
                    description=description,
                )
            else:
                # Generate basic scaffold
                scaffold = BASIC_SCAFFOLD.format(
                    cmd_name=cmd_name,
                    func_name=func_name,
                )

            cmd_path.write_text(scaffold)
            print(f"Created: {cmd_path}")
            editor = os.environ.get("EDITOR", "vim")
            subprocess.run([editor, str(cmd_path)])
            print(f"Restart l3m-chat to load {cmd_name}.\n")
    else:
        print("Usage: /command [ls | edit <name> | new <name> [\"shell cmd\"]]\n")
