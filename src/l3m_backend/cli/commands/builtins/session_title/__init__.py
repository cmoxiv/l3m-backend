"""Session-title command - generate/set session title."""
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


@command_registry.register("session-title", "Generate/set session title", usage="/session-title [title]", has_args=True)
def cmd_session_title(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Generate or set session title."""
    if not session_mgr or not session_mgr.session:
        print("No active session.\n")
        return

    s = session_mgr.session
    if args:
        s.metadata.title = args
        _feedback(f"Title set: {s.metadata.title}")
        session_mgr.save()
    elif s.transcript:
        _feedback("Generating title...")
        title = session_mgr.generate_title(engine)
        if title:
            _feedback(f"Generated title: {title}")
            session_mgr.save()
        else:
            _feedback("Could not generate title.")
    else:
        print("No conversation yet to generate title from.")
    print()
