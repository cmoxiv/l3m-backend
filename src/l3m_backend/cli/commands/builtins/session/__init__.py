"""Session command - show current session info."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("session", "Show current session info")
def cmd_session(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show current session info."""
    if session_mgr and session_mgr.session:
        s = session_mgr.session
        print(f"\nSession ID: {s.metadata.id}")
        print(f"Title: {s.metadata.title or '(untitled)'}")
        print(f"Tag: {s.metadata.tag or '(none)'}")
        print(f"Created: {s.metadata.created_at[:19]}")
        print(f"Updated: {s.metadata.updated_at[:19]}")
        print(f"Initial: {s.metadata.initial_datetime}")
        print(f"Last save: {s.metadata.last_save_datetime or '(not saved)'}")
        print(f"Working dir: {s.metadata.working_directory}")
        print(f"Incognito: {s.metadata.is_incognito}")
        print(f"History: {len(s.history)} messages")
        print(f"Transcript: {len(s.transcript)} messages")
        print(f"Summaries: {len(s.metadata.summaries)}\n")
    else:
        print("No active session.\n")
