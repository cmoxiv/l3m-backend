"""Load magic command - load file content and add to history."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from l3m_backend.cli.magic.registry import magic_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@magic_registry.register("load", "Load file content (adds to history)")
def magic_load(
    engine: "ChatEngine",
    session_mgr: "SessionManager | None",
    args: str,
    add_to_history: Callable[[str, str], None],
) -> bool:
    """Load file content and add to history as user/assistant exchange."""
    if not args.strip():
        print("Usage: %load <file>")
        return True

    filepath = args.strip()
    path = Path(filepath).expanduser()
    try:
        content = path.read_text()
        print(f"Loaded {len(content)} chars from {path.name}")

        # Add to history as user message + assistant response
        user_msg = f"%load {filepath}"
        assistant_msg = f"[File: {path.name}]\n{content}"
        add_to_history(user_msg, assistant_msg)
    except Exception as e:
        print(f"Error loading file: {e}")
    return True
