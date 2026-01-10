"""Tools command - list available tools."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("tools", "List available tools")
def cmd_tools(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """List available tools."""
    print("\nAvailable tools:")
    for entry in engine.registry:
        aliases = f" (aliases: {', '.join(entry.aliases)})" if entry.aliases else ""
        print(f"  - {entry.name}{aliases}")
        print(f"    {entry.get_description()}")
    print()
