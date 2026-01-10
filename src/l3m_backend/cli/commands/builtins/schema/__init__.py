"""Schema command - show tool schema."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("schema", "Show tool schema (OpenAI format)")
def cmd_schema(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show tool schema in OpenAI format."""
    print(f"\n{json.dumps(engine.tools, indent=2)}\n")
