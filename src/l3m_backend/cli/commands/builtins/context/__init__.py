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

        # Count tokens for each component separately
        system_tokens = engine._count_tokens([{"role": "system", "content": engine.system_prompt}])
        contract_tokens = engine._count_tokens([{"role": "system", "content": engine.tools_contract}])
        priming_tokens = engine._count_tokens(engine.priming_messages) if engine.priming_messages else 0
        history_tokens = engine._count_tokens(engine.history) if engine.history else 0

        # Count partition context if present
        partition_tokens = 0
        if engine.partition.has_partitions():
            partition_msgs = []
            if engine.partition.loaded_summaries:
                summary_content = "\n\n".join(engine.partition.loaded_summaries)
                partition_msgs.append({"role": "system", "content": summary_content})
            if engine.partition.loaded_transcript:
                partition_msgs.extend(engine.partition.loaded_transcript)
            if partition_msgs:
                partition_tokens = engine._count_tokens(partition_msgs)

        priming_count = len(engine.priming_messages) if engine.priming_messages else 0
        history_count = len(engine.history)

    except (AttributeError, TypeError):
        # Fallback: estimate 4 chars per token
        total_chars = sum(len(m.get("content", "")) for m in messages)
        total_tokens = total_chars // 4
        n_ctx = 32768
        system_tokens = 0
        contract_tokens = 0
        priming_tokens = 0
        history_tokens = 0
        partition_tokens = 0
        priming_count = 0
        history_count = 0

    pct = (total_tokens / n_ctx) * 100
    print(f"\nContext: {total_tokens:,} / {n_ctx:,} tokens ({pct:.1f}%)")
    print(f"  System:   {system_tokens:,} tokens")
    print(f"  Contract: {contract_tokens:,} tokens")
    print(f"  Priming:  {priming_tokens:,} tokens ({priming_count} messages)")
    if partition_tokens > 0:
        print(f"  Session:  {partition_tokens:,} tokens")
    print(f"  History:  {history_tokens:,} tokens ({history_count} messages)\n")
