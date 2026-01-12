"""System magic command - edit the system message."""
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


@magic_registry.register("system", "Edit system message in $VISUAL")
def magic_system(
    engine: "ChatEngine",
    session_mgr: "SessionManager | None",
    args: str,
    add_to_history: Callable[[str, str], None],
) -> bool:
    """Edit the system message in $VISUAL editor.

    Usage:
        %system           - Open editor to modify system prompt
        %system show      - Display current system prompt
        %system reset     - Reset to default system prompt
        %system <text>    - Set system prompt directly
    """
    arg = args.strip()

    # Show current system prompt
    if arg == "show":
        print(f"\nSystem prompt:\n{'-' * 40}")
        print(engine.system_prompt)
        print(f"{'-' * 40}\n")
        return True

    # Reset to default
    if arg == "reset":
        engine.system_prompt = "You are a helpful assistant with access to tools."
        print("System prompt reset to default.\n")
        return True

    # Set directly if text provided
    if arg and arg not in ("show", "reset"):
        engine.system_prompt = arg
        print(f"System prompt updated ({len(arg)} chars).\n")
        return True

    # Open editor
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR", "vi")
    original = engine.system_prompt

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(original)
        tmpfile = f.name

    try:
        subprocess.run([editor, tmpfile], check=True)

        edited = Path(tmpfile).read_text()
        if edited != original:
            engine.system_prompt = edited.strip()
            print(f"System prompt updated ({len(edited.strip())} chars).\n")
        else:
            print("No changes made.\n")
    except subprocess.CalledProcessError as e:
        print(f"Editor exited with error: {e.returncode}")
    except Exception as e:
        print(f"Editor error: {e}")
    finally:
        Path(tmpfile).unlink(missing_ok=True)

    return True
