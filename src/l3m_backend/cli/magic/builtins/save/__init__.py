"""Save magic command - save conversation to file."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from l3m_backend.cli.magic.registry import magic_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@magic_registry.register("save", "Save conversation (adds to history)")
def magic_save(
    engine: "ChatEngine",
    session_mgr: "SessionManager | None",
    args: str,
    add_to_history: Callable[[str, str], None],
) -> bool:
    """Save conversation to file and add to history as user/assistant exchange."""
    if not args.strip():
        print("Usage: %save <file>")
        return True

    filepath = args.strip()
    path = Path(filepath).expanduser()
    try:
        content = []
        for msg in engine.history:
            role = msg.get("role", "?")
            text = msg.get("content", "")
            content.append(f"## {role.upper()}\n{text}\n")
        path.write_text("\n".join(content))
        abs_path = path.absolute()
        print(f"Saved {len(engine.history)} messages to {abs_path}")

        # Add to history as user message + assistant response
        user_msg = f"%save {filepath}"
        assistant_msg = f"Saved {len(engine.history)} messages to {abs_path}"
        add_to_history(user_msg, assistant_msg)
    except Exception as e:
        print(f"Error saving: {e}")
    return True
