"""Transcript command - show full session transcript."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("transcript", "Show full session transcript", aliases=["tr"])
def cmd_transcript(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show the full session transcript (may differ from engine history)."""
    if not session_mgr or not session_mgr.session:
        print("\nNo session loaded.\n")
        return

    transcript = session_mgr.session.transcript
    history = session_mgr.session.history

    print(f"\nSession transcript ({len(transcript)} messages):")
    print(f"(Engine history has {len(history)} messages)")
    print("-" * 50)

    for i, msg in enumerate(transcript):
        role = msg.role
        content = msg.content
        in_history = "+" if msg in history else "-"

        if role == "system":
            print(f"[{i}] {in_history} system: (system prompt)")
        elif role == "tool":
            print(f"[{i}] {in_history} tool [{msg.tool_call_id or '?'}]: {content[:50]}...")
        elif msg.tool_calls:
            calls = ", ".join(tc["function"]["name"] for tc in msg.tool_calls)
            print(f"[{i}] {in_history} {role}: [calls: {calls}]")
        else:
            preview = (content[:60] + "...") if content and len(content) > 60 else content
            print(f"[{i}] {in_history} {role}: {preview}")

    print()
    if len(transcript) != len(history):
        print("Legend: + = in history, - = transcript only")
        print()
