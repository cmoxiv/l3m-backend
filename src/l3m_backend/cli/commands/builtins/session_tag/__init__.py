"""Session-tag command - add tag to session."""
from __future__ import annotations

import sys
from pathlib import Path
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


@command_registry.register("session-tag", "Add tag to session", usage="/session-tag <tag>", has_args=True)
def cmd_session_tag(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Add or change session tag."""
    if not session_mgr or not session_mgr.session:
        print("No active session.\n")
        return

    s = session_mgr.session
    if not args:
        print(f"Usage: /session-tag <tag>")
        print(f"Current tag: {s.metadata.tag or '(none)'}\n")
        return

    tag = args.strip().lower()
    old_tag = s.metadata.tag
    s.metadata.tag = tag
    cwd = Path(s.metadata.working_directory)
    session_mgr.save(cwd)

    if old_tag:
        _feedback(f"Changed tag: {old_tag} -> {tag}\n")
    else:
        _feedback(f"Set tag: {tag}\n")
