"""History command - show conversation history."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("history", "Show conversation history")
def cmd_history(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show conversation history."""
    print()
    for i, msg in enumerate(engine.messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if role == "system":
            print(f"[{i}] system: (system prompt)")
        elif role == "tool":
            print(f"[{i}] tool [{msg.get('tool_call_id', '?')}]: {content}")
        elif msg.get("tool_calls"):
            calls = ", ".join(tc["function"]["name"] for tc in msg["tool_calls"])
            print(f"[{i}] {role}: [calls: {calls}]")
        else:
            preview = (content[:60] + "...") if content and len(content) > 60 else content
            print(f"[{i}] {role}: {preview}")
    print()
