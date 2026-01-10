"""Help command - show available commands."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("help", "Show available commands")
def cmd_help(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show help message."""
    print("\nCommands:")
    for entry in command_registry.all_commands():
        print(f"  {entry.usage:<16} - {entry.description}")
    print(f"  {'!<cmd>':<16} - Run shell command")
    print(f"  {'%<cmd>':<16} - Magic commands (type %help)")
    print()
