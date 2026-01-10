"""Sessions command - list/search sessions."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("sessions", "List/search sessions", usage="/sessions [query]", has_args=True)
def cmd_sessions(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """List or search sessions."""
    if not session_mgr:
        print("Session management not available.\n")
        return

    query = args.strip() if args else None
    if query:
        sessions = session_mgr.search(query)
        if not sessions:
            print(f"No sessions matching '{query}'\n")
            return
        print(f"\nSessions matching '{query}':")
    else:
        sessions = session_mgr.list_sessions()
        if not sessions:
            print("No sessions found.\n")
            return
        print("\nAvailable sessions:")

    for s in sessions[:10]:
        title = s.title or "(untitled)"
        print(f"  {s.id[:8]}  {title[:40]}")
    if len(sessions) > 10:
        print(f"  ... and {len(sessions) - 10} more")
    print()
