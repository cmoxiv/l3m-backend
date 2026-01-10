"""Clear command - clear conversation history."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("clear", "Clear conversation history")
def cmd_clear(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Clear conversation history."""
    engine.clear()
    print("Conversation cleared.\n")
