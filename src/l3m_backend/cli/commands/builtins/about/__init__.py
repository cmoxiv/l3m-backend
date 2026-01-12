"""About command - show message context for an entity."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register(
    "about",
    "Show messages about an entity",
    usage="/about <entity_name>",
    has_args=True,
)
def cmd_about(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Display messages where an entity was discussed.

    Usage:
        /about ChatEngine
        /about numpy
    """
    if not hasattr(engine, "knowledge_graph") or engine.knowledge_graph is None:
        print("\nNo knowledge graph available.\n")
        return

    entity_name = args.strip().strip('"').strip("'")
    if not entity_name:
        print("\nUsage: /about <entity_name>\n")
        return

    graph = engine.knowledge_graph
    entity = graph.get_entity(entity_name)

    if not entity:
        print(f"\nEntity not found: {entity_name}")
        similar = [e.name for e in graph.entities.values()
                   if entity_name.lower() in e.name.lower()][:5]
        if similar:
            print(f"Did you mean: {', '.join(similar)}")
        print()
        return

    # Get messages about this entity
    msg_ids = graph.get_messages_about(entity.id)

    print(f"\n\"{entity.name}\" [{entity.entity_type.value}]")
    print(f"Mentions: {entity.mention_count} | First seen: {entity.first_seen_msg or 'N/A'}")

    if entity.context:
        print(f"Context: {entity.context}")

    # Show related entities
    related = graph.get_related(entity.id)
    if related:
        related_names = [e.name for e, _ in related[:5]]
        print(f"Related: {', '.join(related_names)}")

    if msg_ids:
        print(f"\nMentioned in {len(msg_ids)} message(s):")

        # Try to get actual message content from history
        for msg_id in msg_ids[:10]:
            # Parse msg_id to get index (format: msg_N)
            try:
                idx = int(msg_id.replace("msg_", ""))
                if idx < len(engine.history):
                    msg = engine.history[idx]
                    role = msg.get("role", "?")
                    content = msg.get("content", "")[:60]
                    if len(msg.get("content", "")) > 60:
                        content += "..."
                    print(f"  [{msg_id}] {role}: {content}")
                else:
                    print(f"  [{msg_id}] (message not in current history)")
            except (ValueError, IndexError):
                print(f"  [{msg_id}]")

        if len(msg_ids) > 10:
            print(f"  ... and {len(msg_ids) - 10} more")
    else:
        print("\nNo message references found.")

    print()
