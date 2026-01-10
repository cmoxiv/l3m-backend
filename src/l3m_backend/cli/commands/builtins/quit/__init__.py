"""Quit command - exit the REPL."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


def _feedback(msg: str) -> None:
    """Print feedback message in grey."""
    import sys
    GREY = "\033[90m"
    RESET = "\033[0m"
    print(f"{GREY}{msg}{RESET}", file=sys.stderr)


@command_registry.register("quit", "Exit the REPL", aliases=["exit", "q"])
def cmd_quit(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str) -> bool:
    """Exit the REPL, generating summary and saving session."""
    # Generate summary before exit
    if session_mgr and session_mgr.session and session_mgr.session.transcript:
        last_end = session_mgr.get_last_summary_end_idx()
        if len(session_mgr.session.transcript) > last_end:
            _feedback("Generating session summary...")
            session_mgr.generate_summary(engine, last_end)

    # Save and update symlink
    if session_mgr and session_mgr.session:
        cwd = Path(session_mgr.session.metadata.working_directory)
        session_mgr.save(cwd)
        session_mgr.create_symlink(cwd)

    print("Goodbye!")
    return True  # Signal to exit REPL
