"""Clear command - clear conversation history."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("clear", "Clear conversation history")
def cmd_clear(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Clear conversation history, transcript, and summaries."""
    engine.clear()

    # Also clear session history, transcript, and summaries
    if session_mgr and session_mgr.session:
        session_mgr.session.history = []
        session_mgr.session.transcript = []
        session_mgr.session.metadata.summaries = []
        # Save the cleared session
        cwd = Path(session_mgr.session.metadata.working_directory)
        session_mgr.save(cwd)
        print("Conversation, transcript, and summaries cleared.\n")
    else:
        print("Conversation cleared.\n")
