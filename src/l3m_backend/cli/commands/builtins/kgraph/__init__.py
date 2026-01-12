"""KGraph command - knowledge graph overview and export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register(
    "kgraph",
    "Knowledge graph overview/export",
    usage="/kgraph [--stats|--export]",
    has_args=True,
    aliases=["kg"],
)
def cmd_kgraph(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show knowledge graph statistics or export to file.

    Usage:
        /kgraph          - Show overview
        /kgraph --stats  - Show detailed statistics
        /kgraph --export - Export graph to JSON file
    """
    if not hasattr(engine, "knowledge_graph") or engine.knowledge_graph is None:
        print("\nNo knowledge graph available.")
        print("Resume a session to build one (enabled by default).\n")
        return

    graph = engine.knowledge_graph
    arg = args.strip().lower()

    if arg == "--export":
        _export_graph(graph, session_mgr)
    elif arg == "--stats":
        _show_detailed_stats(graph)
    else:
        _show_overview(graph)


def _show_overview(graph) -> None:
    """Show basic graph overview."""
    stats = graph.stats

    # Count by type
    entity_types = stats.get("entity_types", {})
    type_summary = ", ".join(f"{v} {k}" for k, v in sorted(entity_types.items(), key=lambda x: -x[1])[:4])

    relation_types = stats.get("relation_types", {})
    rel_summary = ", ".join(f"{v} {k}" for k, v in sorted(relation_types.items(), key=lambda x: -x[1])[:3])

    print("\nKnowledge Graph:")
    print(f"  Entities: {stats['num_entities']} ({type_summary})")
    print(f"  Edges: {stats['num_edges']} ({rel_summary})")
    print(f"  Messages indexed: {stats['num_messages']}")

    # Most mentioned
    most_mentioned = graph.get_most_mentioned(limit=3)
    if most_mentioned:
        mentions = ", ".join(f"{e.name} ({e.mention_count})" for e in most_mentioned)
        print(f"\n  Most mentioned: {mentions}")

    # Most connected (count edges)
    if graph.edges:
        edge_counts: dict[str, int] = {}
        for edge in graph.edges:
            edge_counts[edge.source_id] = edge_counts.get(edge.source_id, 0) + 1
            edge_counts[edge.target_id] = edge_counts.get(edge.target_id, 0) + 1

        if edge_counts:
            top_id = max(edge_counts, key=edge_counts.get)  # type: ignore
            top_entity = graph.entities.get(top_id)
            if top_entity:
                print(f"  Most connected: {top_entity.name} ({edge_counts[top_id]} edges)")

    print()


def _show_detailed_stats(graph) -> None:
    """Show detailed statistics."""
    stats = graph.stats

    print("\nKnowledge Graph Statistics:")
    print("=" * 40)

    print(f"\nTotal Entities: {stats['num_entities']}")
    for etype, count in sorted(stats.get("entity_types", {}).items()):
        print(f"  {etype}: {count}")

    print(f"\nTotal Edges: {stats['num_edges']}")
    for rtype, count in sorted(stats.get("relation_types", {}).items()):
        print(f"  {rtype}: {count}")

    print(f"\nMessages Indexed: {stats['num_messages']}")

    # Top entities by mentions
    print("\nTop Entities by Mentions:")
    for entity in graph.get_most_mentioned(limit=10):
        print(f"  {entity.name}: {entity.mention_count} mentions")

    print()


def _export_graph(graph, session_mgr) -> None:
    """Export graph to JSON file."""
    # Determine export path
    export_dir = Path.home() / ".l3m" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    if session_mgr and session_mgr.session:
        session_id = session_mgr.session.metadata.id[:8]
        filename = f"kg-{session_id}.json"
    else:
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"kg-{timestamp}.json"

    export_path = export_dir / filename

    # Export
    data = graph.to_dict()
    data["_export_info"] = {
        "entities": len(graph.entities),
        "edges": len(graph.edges),
        "messages": len(graph.message_entities),
    }

    export_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"\nExported knowledge graph to: {export_path}\n")
