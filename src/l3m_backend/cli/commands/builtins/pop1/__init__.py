"""Pop1 command - remove last single message."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("pop1", "Remove last single message")
def cmd_pop1(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Remove last single message."""
    if engine.history:
        msg = engine.history.pop()
        role = msg.get("role", "?")
        if session_mgr:
            session_mgr.sync_from_engine(engine.history)
        print(f"Removed 1 {role} message.\n")
    else:
        print("History is empty.\n")
