"""Undo command - remove last user+assistant exchange."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("undo", "Remove last user+assistant exchange")
def cmd_undo(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Remove last user+assistant exchange."""
    if len(engine.history) >= 2:
        engine.history.pop()  # assistant
        engine.history.pop()  # user
        if session_mgr:
            session_mgr.sync_from_engine(engine.history)
        print("Removed last exchange.\n")
    elif engine.history:
        engine.history.pop()
        if session_mgr:
            session_mgr.sync_from_engine(engine.history)
        print("Removed last message.\n")
    else:
        print("History is empty.\n")
