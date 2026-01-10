"""Debug command - toggle debug mode."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("debug", "Toggle debug mode on/off")
def cmd_debug(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Toggle debug mode."""
    engine.debug = not engine.debug
    status = "enabled" if engine.debug else "disabled"
    print(f"Debug mode {status}.\n")
