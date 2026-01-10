"""Pop command - remove last n message pairs."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("pop", "Remove last n message pairs", usage="/pop [n]", has_args=True)
def cmd_pop(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Remove last n message pairs."""
    n = int(args) if args.isdigit() else 1
    removed_pairs = 0
    for _ in range(n):
        if len(engine.history) >= 2:
            engine.history.pop()  # assistant
            engine.history.pop()  # user
            removed_pairs += 1
        else:
            break
    if removed_pairs > 0:
        if session_mgr:
            session_mgr.sync_from_engine(engine.history)
        print(f"Removed {removed_pairs} message pair(s).\n")
    else:
        print("History is empty or incomplete pair.\n")
