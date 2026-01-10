"""Time magic command - report current time."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Callable

from l3m_backend.cli.magic.registry import magic_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@magic_registry.register("time", "Report current time (adds to history)")
def magic_time(
    engine: "ChatEngine",
    session_mgr: "SessionManager | None",
    args: str,
    add_to_history: Callable[[str, str], None],
) -> bool:
    """Report current time and add to history as user/assistant exchange."""
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current time: {time_str}")

    # Add to history as user message + assistant response
    user_msg = "%time"
    assistant_msg = f"Current time: {time_str}"
    add_to_history(user_msg, assistant_msg)
    return True
