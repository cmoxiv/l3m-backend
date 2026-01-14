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


def _parse_json_content(msg: dict) -> dict:
    """Parse JSON strings in message content into nested dicts."""
    result = dict(msg)
    content = result.get("content", "")

    if isinstance(content, str) and content.strip().startswith("{"):
        try:
            result["content"] = json.loads(content)
        except json.JSONDecodeError:
            pass

    return result


@command_registry.register("history", "Show conversation history", usage="/history [full|raw]", has_args=True)
def cmd_history(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show conversation history.

    Usage:
        /history      - Show history with truncated messages
        /history full - Show as JSON with parsed nested content
        /history raw  - Show raw message format (no parsing)
    """
    arg = args.strip().lower()
    full = arg == "full"
    raw = arg == "raw"

    print()
    for i, msg in enumerate(engine.messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")

        if raw:
            # Raw mode - show message as-is (JSON format, single line)
            print(f"[{i}] {json.dumps(msg)}")
            continue

        if full:
            # Full mode - show as JSON with parsed nested content
            parsed_msg = _parse_json_content(msg)
            formatted = json.dumps(parsed_msg, indent=2)
            # Indent all lines and add index prefix
            lines = formatted.split("\n")
            print(f"[{i}] {lines[0]}")
            for line in lines[1:]:
                print(f"    {line}")
            continue

        # Default mode - truncated readable format
        if role == "system":
            print(f"[{i}] system: (system prompt)")
        elif role == "tool":
            # Tool results - either native (with tool_call_id) or contract-based
            if msg.get("tool_call_id"):
                # Native tool result
                preview = (content[:60] + "...") if content and len(content) > 60 else content
                print(f"[{i}] tool [{msg.get('tool_call_id')}]: {preview}")
            else:
                # Contract-based tool result (JSON format)
                tool_name, result = _parse_tool_result(content)
                label = f"tool [{tool_name}]" if tool_name else "tool"
                preview = (result[:60] + "...") if result and len(result) > 60 else result
                print(f"[{i}] {label}: {preview}")
        elif msg.get("tool_calls"):
            calls = ", ".join(tc["function"]["name"] for tc in msg["tool_calls"])
            print(f"[{i}] {role}: [calls: {calls}]")
        else:
            preview = (content[:60] + "...") if content and len(content) > 60 else content
            print(f"[{i}] {role}: {preview}")
    print()
