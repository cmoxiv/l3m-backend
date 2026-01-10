"""System command - show system prompt."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("system", "Show system prompt")
def cmd_system(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show system prompt."""
    print(f"\n{engine.system_prompt}\n")
