"""Duplicates command - find near-duplicate messages."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register(
    "duplicates",
    "Find near-duplicate messages",
    usage="/duplicates [--threshold N] [--show]",
    has_args=True,
    aliases=["dups"],
)
def cmd_duplicates(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Find near-duplicate messages in conversation history.

    Useful for identifying redundant content that could be trimmed.

    Usage:
        /duplicates              - Show duplicate count
        /duplicates --show       - Display duplicate pairs
        /duplicates --threshold 0.85 - Use custom similarity threshold
    """
    if not hasattr(engine, "similarity_graph") or engine.similarity_graph is None:
        print("\nNo similarity graph available.")
        print("Resume a session with similarity graph enabled.\n")
        return

    graph = engine.similarity_graph
    if not graph.is_built:
        print("\nSimilarity graph not built yet.\n")
        return

    # Parse arguments
    parts = args.strip().split()
    threshold = 0.9
    show_pairs = False

    i = 0
    while i < len(parts):
        if parts[i] == "--threshold" and i + 1 < len(parts):
            try:
                threshold = float(parts[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        elif parts[i] == "--show":
            show_pairs = True
            i += 1
            continue
        i += 1

    # Find duplicates
    duplicates = graph.find_duplicates(threshold=threshold)
    unique_indices = graph.get_unique_messages(threshold=threshold)

    total_messages = len(graph.message_nodes)
    duplicate_count = total_messages - len(unique_indices)

    # Get engine history count for comparison
    engine_history_count = len(engine.history) if engine.history else 0

    print(f"\nDuplicate Analysis (threshold={threshold}):")
    print("-" * 50)
    print(f"  Transcript messages: {total_messages}")
    if engine_history_count != total_messages:
        print(f"  Engine history: {engine_history_count}")
    print(f"  Unique messages: {len(unique_indices)}")
    print(f"  Duplicates: {duplicate_count}")
    print(f"  Duplicate pairs found: {len(duplicates)}")

    if duplicates and show_pairs:
        print(f"\nDuplicate Pairs:")
        print("-" * 50)
        for i, (idx1, idx2, score) in enumerate(duplicates[:20]):  # Limit to 20
            node1 = graph.message_nodes[idx1] if idx1 < len(graph.message_nodes) else None
            node2 = graph.message_nodes[idx2] if idx2 < len(graph.message_nodes) else None

            if node1 and node2:
                content1 = node1.content[:50] + "..." if len(node1.content) > 50 else node1.content
                content2 = node2.content[:50] + "..." if len(node2.content) > 50 else node2.content
                content1 = content1.replace("\n", " ")
                content2 = content2.replace("\n", " ")
                print(f"\n  Pair {i+1} (similarity: {score:.2f}):")
                print(f"    [{idx1}] {content1}")
                print(f"    [{idx2}] {content2}")

        if len(duplicates) > 20:
            print(f"\n  ... and {len(duplicates) - 20} more pairs")

    if duplicate_count > 0:
        savings = (duplicate_count / total_messages) * 100
        print(f"\n  Potential context savings: {savings:.1f}%")

    print()
