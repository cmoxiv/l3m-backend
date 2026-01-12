"""Related command - find entities related to a given entity."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register(
    "related",
    "Find related entities",
    usage="/related <entity_name>",
    has_args=True,
    aliases=["rel"],
)
def cmd_related(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show entities related to a given entity.

    Usage:
        /related ChatEngine
        /related "error handling"
    """
    if not hasattr(engine, "knowledge_graph") or engine.knowledge_graph is None:
        print("\nNo knowledge graph available.\n")
        return

    entity_name = args.strip().strip('"').strip("'")
    if not entity_name:
        print("\nUsage: /related <entity_name>\n")
        return

    graph = engine.knowledge_graph
    entity = graph.get_entity(entity_name)

    if not entity:
        print(f"\nEntity not found: {entity_name}")
        # Suggest similar entities
        similar = [e.name for e in graph.entities.values()
                   if entity_name.lower() in e.name.lower()][:5]
        if similar:
            print(f"Did you mean: {', '.join(similar)}")
        print()
        return

    # Get related entities
    related = graph.get_related(entity.id)

    if not related:
        print(f"\nNo entities related to \"{entity.name}\".\n")
        return

    print(f"\nRelated to \"{entity.name}\" [{entity.entity_type.value}]:")

    for rel_entity, edge in related[:15]:
        direction = "→" if edge.source_id == entity.id else "←"
        relation = edge.relation_type.value.upper()
        weight_str = f", weight: {edge.weight:.1f}" if edge.weight < 1.0 else ""
        msg_count = len(edge.message_ids)
        msg_str = f" in {msg_count} msg{'s' if msg_count != 1 else ''}" if msg_count else ""

        print(f"  {direction} {rel_entity.name:20} [{relation}{weight_str}{msg_str}]")

    if len(related) > 15:
        print(f"  ... and {len(related) - 15} more")

    print()
