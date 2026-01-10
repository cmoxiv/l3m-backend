"""Context command - show context usage estimate."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("context", "Show context usage estimate")
def cmd_context(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show context usage estimate."""
    messages = engine._build_messages()
    try:
        total_tokens = engine._count_tokens(messages)
        n_ctx = engine.llm.n_ctx()
        if not isinstance(n_ctx, int):
            n_ctx = 32768
    except (AttributeError, TypeError):
        # Fallback: estimate 4 chars per token
        total_chars = sum(len(m.get("content", "")) for m in messages)
        total_tokens = total_chars // 4
        n_ctx = 32768
    pct = (total_tokens / n_ctx) * 100
    print(f"\nContext: {total_tokens:,} / {n_ctx:,} tokens ({pct:.1f}%)")
    print(f"History: {len(engine.history)} messages\n")
