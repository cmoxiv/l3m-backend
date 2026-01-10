"""Push1 command - push a single message."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("push1", "Push a single message", usage="/push1 <content>", has_args=True)
def cmd_push1(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Push a single message (role inferred from last message)."""
    if not args:
        print("Usage: /push1 <content>\n")
        return
    # Infer role from last message
    if engine.history:
        last_role = engine.history[-1].get("role", "user")
        role = "assistant" if last_role == "user" else "user"
    else:
        role = "user"
    engine.history.append({"role": role, "content": args})
    if session_mgr:
        session_mgr.sync_from_engine(engine.history)
    print(f"Pushed 1 {role} message.\n")
