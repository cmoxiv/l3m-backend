"""Session-load command - load session as context."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("session-load", "Load session as context", usage="/session-load <id>", has_args=True)
def cmd_session_load(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Load session as context."""
    if not session_mgr:
        print("Session management not available.\n")
        return

    if not args:
        print("Usage: /session-load <session_id>")
        print("Use /sessions to list available sessions.\n")
        return

    session_id = args.strip()
    print(f"Loading session {session_id[:8]}...")
    loaded = session_mgr.load_as_context([session_id], engine)
    if loaded:
        print(f"Loaded: {', '.join(loaded)}\n")
    else:
        print(f"Could not load session: {session_id}\n")
