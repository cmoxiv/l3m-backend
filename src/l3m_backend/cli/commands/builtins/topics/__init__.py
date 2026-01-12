"""Topics command - show topic summary from knowledge graph."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register(
    "topics",
    "Show topic summary from knowledge graph",
)
def cmd_topics(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Display topic-based summary of the session.

    Usage:
        /topics
    """
    if not hasattr(engine, "knowledge_graph") or engine.knowledge_graph is None:
        print("\nNo knowledge graph available.\n")
        return

    graph = engine.knowledge_graph
    topics = graph.get_topic_summary()

    if not topics:
        print("\nNo topics identified yet.")
        print("Topics are extracted from topic/theme/concept entities.\n")
        return

    num_messages = len(graph.message_entities)
    print(f"\nSession Topics ({len(topics)} found):\n")

    for topic_name, entities in list(topics.items())[:10]:
        print(f"• {topic_name}")
        # Show related entities (excluding the topic itself)
        related = [e.name for e in entities if e.name != topic_name][:5]
        if related:
            print(f"  └─ {', '.join(related)}")
        print()

    if len(topics) > 10:
        print(f"... and {len(topics) - 10} more topics\n")

    print(f"From {num_messages} indexed messages.\n")
