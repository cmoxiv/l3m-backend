"""Neighbors command - find messages similar to a specific message."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register(
    "neighbors",
    "Find messages similar to one",
    usage="/neighbors <msg_id> [--k N]",
    has_args=True,
    aliases=["nbrs"],
)
def cmd_neighbors(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Find messages similar to a specific message.

    Usage:
        /neighbors 5       - Find messages similar to message #5
        /neighbors 5 --k 10 - Show top 10 similar messages
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
    msg_idx = None
    k = 5

    i = 0
    while i < len(parts):
        if parts[i] == "--k" and i + 1 < len(parts):
            try:
                k = int(parts[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        elif msg_idx is None:
            try:
                msg_idx = int(parts[i])
            except ValueError:
                pass
        i += 1

    if msg_idx is None:
        print("\nUsage: /neighbors <msg_id> [--k N]")
        print("Example: /neighbors 5")
        print(f"\nValid message IDs: 0 to {len(graph.message_nodes) - 1}\n")
        return

    if msg_idx < 0 or msg_idx >= len(graph.message_nodes):
        print(f"\nInvalid message ID: {msg_idx}")
        print(f"Valid range: 0 to {len(graph.message_nodes) - 1}\n")
        return

    # Get the source message
    source_node = graph.message_nodes[msg_idx]
    source_content = source_node.content[:100] + "..." if len(source_node.content) > 100 else source_node.content
    source_content = source_content.replace("\n", " ")

    print(f"\nMessage [{msg_idx}]: {source_content}")
    print("-" * 50)

    # Get neighbors
    neighbors = graph.get_message_neighbors(msg_idx, k=k)

    if neighbors:
        print(f"\nTop {len(neighbors)} similar messages:")
        for neighbor_idx, score in neighbors:
            if neighbor_idx < len(graph.message_nodes):
                node = graph.message_nodes[neighbor_idx]
                content = node.content[:80] + "..." if len(node.content) > 80 else node.content
                content = content.replace("\n", " ")
                print(f"  [{neighbor_idx}] ({score:.2f}) {content}")
    else:
        print("\nNo similar messages found above threshold.")

    print()
