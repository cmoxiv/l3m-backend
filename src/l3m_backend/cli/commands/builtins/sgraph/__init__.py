"""SGraph command - similarity graph overview and export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register(
    "sgraph",
    "Similarity graph overview/export",
    usage="/sgraph [--stats|--export]",
    has_args=True,
    aliases=["sg"],
)
def cmd_sgraph(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show similarity graph statistics or export to file.

    Usage:
        /sgraph          - Show overview
        /sgraph --stats  - Show detailed statistics
        /sgraph --export - Export graph data to JSON file
    """
    if not hasattr(engine, "similarity_graph") or engine.similarity_graph is None:
        print("\nNo similarity graph available.")
        print("Resume a session with similarity graph enabled.\n")
        return

    graph = engine.similarity_graph
    if not graph.is_built:
        print("\nSimilarity graph not built yet.\n")
        return

    arg = args.strip().lower()

    if arg == "--export":
        _export_graph(graph, session_mgr)
    elif arg == "--stats":
        _show_detailed_stats(graph, engine)
    else:
        _show_overview(graph, engine)


def _show_overview(graph, engine) -> None:
    """Show basic graph overview."""
    stats = graph.stats

    # Determine embedding provider type
    provider_type = "unknown"
    if hasattr(engine, "_embedding_provider") and engine._embedding_provider is not None:
        provider_type = type(engine._embedding_provider).__name__

    # Get engine history count for comparison
    engine_history_count = len(engine.history) if engine.history else 0

    print("\nSimilarity Graph:")
    print(f"  Transcript messages: {stats['num_messages']}")
    if engine_history_count != stats['num_messages']:
        print(f"  Engine history: {engine_history_count}")
    print(f"  Summaries: {stats['num_summaries']}")
    print(f"  Edges (above threshold): {stats['num_edges']}")
    print(f"  Avg similarity: {stats['avg_msg_similarity']:.3f}")
    print(f"  Embedding dim: {stats['embedding_dim']}")
    print(f"  Provider: {provider_type}")
    print(f"  Threshold: {stats['threshold']}")
    print()


def _show_detailed_stats(graph, engine) -> None:
    """Show detailed statistics."""
    stats = graph.stats

    # Provider info
    provider_type = "unknown"
    if hasattr(engine, "_embedding_provider") and engine._embedding_provider is not None:
        provider_type = type(engine._embedding_provider).__name__

    print("\nSimilarity Graph Statistics:")
    print("=" * 50)

    print(f"\nEmbedding Provider: {provider_type}")
    print(f"Embedding Dimension: {stats['embedding_dim']}")
    print(f"Similarity Threshold: {stats['threshold']}")

    print(f"\nNodes:")
    print(f"  Messages: {stats['num_messages']}")
    print(f"  Summaries: {stats['num_summaries']}")

    print(f"\nGraph:")
    print(f"  Edges (above threshold): {stats['num_edges']}")
    print(f"  Average message similarity: {stats['avg_msg_similarity']:.4f}")

    # Find duplicates
    duplicates = graph.find_duplicates(threshold=0.9)
    print(f"\nDuplicates (>0.9 similarity): {len(duplicates)}")

    # Show some high-similarity pairs
    if duplicates:
        print("\nHighest similarity pairs:")
        for idx1, idx2, score in duplicates[:5]:
            print(f"  [{idx1}] <-> [{idx2}]: {score:.3f}")

    print()


def _export_graph(graph, session_mgr) -> None:
    """Export graph data to JSON file."""
    export_dir = Path.home() / ".l3m" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    if session_mgr and session_mgr.session:
        session_id = session_mgr.session.metadata.id[:8]
        filename = f"sg-{session_id}.json"
    else:
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"sg-{timestamp}.json"

    export_path = export_dir / filename

    # Build export data
    data = {
        "stats": graph.stats,
        "messages": [
            {
                "id": node.id,
                "index": node.index,
                "content": node.content[:500],  # Truncate for export
            }
            for node in graph.message_nodes
        ],
        "summaries": [
            {
                "id": node.id,
                "index": node.index,
                "content": node.content[:500],
            }
            for node in graph.summary_nodes
        ],
    }

    # Include similarity matrix stats (not full matrix - too large)
    if graph.msg_to_msg is not None:
        import numpy as np
        data["msg_similarity_stats"] = {
            "min": float(np.min(graph.msg_to_msg)),
            "max": float(np.max(graph.msg_to_msg)),
            "mean": float(np.mean(graph.msg_to_msg)),
            "std": float(np.std(graph.msg_to_msg)),
        }

    export_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"\nExported similarity graph to: {export_path}\n")
