"""Entities command - list extracted entities from knowledge graph."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register(
    "entities",
    "List entities from knowledge graph",
    usage="/entities [type]",
    has_args=True,
    aliases=["ent"],
)
def cmd_entities(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """List extracted entities, optionally filtered by type.

    Usage:
        /entities          - List all entities
        /entities class    - List only class entities
        /entities function - List only function entities
    """
    if not hasattr(engine, "knowledge_graph") or engine.knowledge_graph is None:
        print("\nNo knowledge graph available.")
        print("Resume a session to build one (enabled by default).\n")
        return

    graph = engine.knowledge_graph
    type_filter = args.strip().lower() if args else None

    # Get entities
    if type_filter:
        from l3m_backend.engine.knowledge import EntityType
        try:
            entity_type = EntityType(type_filter)
            entities = graph.get_entities_by_type(entity_type)
        except ValueError:
            print(f"\nUnknown entity type: {type_filter}")
            print(f"Valid types: {', '.join(t.value for t in EntityType)}\n")
            return
    else:
        entities = list(graph.entities.values())

    if not entities:
        if type_filter:
            print(f"\nNo {type_filter} entities found.\n")
        else:
            print("\nNo entities extracted yet.\n")
        return

    # Sort by mention count
    entities.sort(key=lambda e: e.mention_count, reverse=True)

    # Display
    total = len(entities)
    type_info = f" ({type_filter})" if type_filter else ""
    print(f"\nEntities{type_info} ({total} total):")

    for entity in entities[:20]:  # Limit display
        type_tag = f"[{entity.entity_type.value}]"
        mentions = f"({entity.mention_count} mentions)" if entity.mention_count > 1 else ""
        print(f"  {type_tag:12} {entity.name} {mentions}")

    if total > 20:
        print(f"  ... and {total - 20} more")

    print()
