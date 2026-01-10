"""Edit-response magic command - edit last assistant response in editor."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from l3m_backend.cli.magic.registry import magic_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@magic_registry.register("edit-response", "Edit last assistant response in $VISUAL")
def magic_edit_response(
    engine: "ChatEngine",
    session_mgr: "SessionManager | None",
    args: str,
    add_to_history: Callable[[str, str], None],
) -> bool:
    """Edit last assistant response in $VISUAL editor."""
    # Check for conversation history
    if not engine.history:
        print("No conversation history yet.")
        return True

    # Find the last assistant message
    assistant_idx = None
    for i in range(len(engine.history) - 1, -1, -1):
        if engine.history[i].get("role") == "assistant":
            assistant_idx = i
            break

    if assistant_idx is None:
        print("No assistant response to edit.")
        return True

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR", "vi")
    original = engine.history[assistant_idx]["content"]

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(original)
        tmpfile = f.name

    try:
        # Open editor
        subprocess.run([editor, tmpfile], check=True)

        # Read edited content
        edited = Path(tmpfile).read_text()
        if edited != original:
            # Replace the existing assistant message (no new message created)
            engine.history[assistant_idx]["content"] = edited
            print(f"Updated assistant response ({len(edited)} chars)")
        else:
            print("No changes made.")
    except subprocess.CalledProcessError as e:
        print(f"Editor exited with error: {e.returncode}")
    except Exception as e:
        print(f"Editor error: {e}")
    finally:
        Path(tmpfile).unlink(missing_ok=True)

    return True
