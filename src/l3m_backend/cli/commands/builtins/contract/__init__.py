"""Contract command - show full system message with tools contract."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("contract", "Show full system message with tools contract")
def cmd_contract(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show full system message with tools contract."""
    print(f"\n{engine._build_system_message()['content']}\n")
