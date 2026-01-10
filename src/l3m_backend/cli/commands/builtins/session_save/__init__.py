"""Session-save command - save session."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


def _feedback(msg: str) -> None:
    """Print feedback message in grey."""
    GREY = "\033[90m"
    RESET = "\033[0m"
    print(f"{GREY}{msg}{RESET}", file=sys.stderr)


@command_registry.register("session-save", "Save session", usage="/session-save [title]", has_args=True)
def cmd_session_save(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Save session with optional title."""
    if not session_mgr or not session_mgr.session:
        print("No active session.\n")
        return

    s = session_mgr.session
    if args:
        s.metadata.title = args
        _feedback(f"Set session title: {s.metadata.title}")
    elif not s.metadata.title and s.transcript:
        _feedback("Generating title...")
        title = session_mgr.generate_title(engine)
        if title:
            _feedback(f"Generated title: {title}")
        else:
            _feedback("Could not generate title.")

    path = session_mgr.save()
    _feedback(f"Session saved: {path}\n")
