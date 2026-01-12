"""History command - show conversation history."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


def _parse_tool_result(content: str) -> tuple[str | None, str]:
    """Parse tool result content, returning (tool_name, result_content)."""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("type") == "tool_result":
            return data.get("name"), data.get("content", "")
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback for old format
    if content.startswith("[Tool Result]:"):
        return None, content[14:].strip()
    return None, content


@command_registry.register("history", "Show conversation history")
def cmd_history(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show conversation history.

    Usage:
        /history      - Show history with truncated messages
        /history full - Show history with full message content
    """
    full = args.strip().lower() == "full"
    print()
    for i, msg in enumerate(engine.messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if role == "system":
            print(f"[{i}] system: (system prompt)")
        elif role == "tool":
            # Tool results - either native (with tool_call_id) or contract-based
            if msg.get("tool_call_id"):
                # Native tool result
                if full:
                    print(f"[{i}] tool [{msg.get('tool_call_id')}]: {content}")
                else:
                    preview = (content[:60] + "...") if content and len(content) > 60 else content
                    print(f"[{i}] tool [{msg.get('tool_call_id')}]: {preview}")
            else:
                # Contract-based tool result (JSON format)
                tool_name, result = _parse_tool_result(content)
                label = f"tool [{tool_name}]" if tool_name else "tool"
                if full:
                    print(f"[{i}] {label}: {result}")
                else:
                    preview = (result[:60] + "...") if result and len(result) > 60 else result
                    print(f"[{i}] {label}: {preview}")
        elif msg.get("tool_calls"):
            calls = ", ".join(tc["function"]["name"] for tc in msg["tool_calls"])
            print(f"[{i}] {role}: [calls: {calls}]")
        else:
            if full:
                print(f"[{i}] {role}: {content}")
            else:
                preview = (content[:60] + "...") if content and len(content) > 60 else content
                print(f"[{i}] {role}: {preview}")
    print()
