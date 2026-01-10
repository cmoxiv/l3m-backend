"""Shell magic command - run shell command and add to history."""
from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Callable

from l3m_backend.cli.magic.registry import magic_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@magic_registry.register("!", "Run shell command (adds to history)")
def magic_shell(
    engine: "ChatEngine",
    session_mgr: "SessionManager | None",
    args: str,
    add_to_history: Callable[[str, str], None],
) -> bool:
    """Run shell command and add to history as user/assistant exchange."""
    if not args.strip():
        print("Usage: %!<command>")
        return True

    try:
        result = subprocess.run(
            args,
            shell=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        if output:
            print(output, end="" if output.endswith("\n") else "\n")

        # Add to history as user message + assistant response
        user_msg = f"%!{args}"
        assistant_msg = output.strip() if output.strip() else "(no output)"
        add_to_history(user_msg, assistant_msg)
    except Exception as e:
        print(f"Error: {e}")
    return True
