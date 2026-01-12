"""Cluster command - group messages by semantic similarity."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register(
    "cluster",
    "Cluster messages by topic",
    usage="/cluster [--k N]",
    has_args=True,
)
def cmd_cluster(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Group messages into topic clusters using semantic similarity.

    Uses simple agglomerative clustering based on similarity matrix.

    Usage:
        /cluster         - Auto-detect clusters
        /cluster --k 5   - Create 5 clusters
    """
    if not hasattr(engine, "similarity_graph") or engine.similarity_graph is None:
        print("\nNo similarity graph available.")
        print("Resume a session with similarity graph enabled.\n")
        return

    graph = engine.similarity_graph
    if not graph.is_built or graph.msg_to_msg is None:
        print("\nSimilarity graph not built yet.\n")
        return

    # Parse arguments
    parts = args.strip().split()
    k = None  # Auto-detect

    i = 0
    while i < len(parts):
        if parts[i] == "--k" and i + 1 < len(parts):
            try:
                k = int(parts[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        i += 1

    # Simple clustering using similarity threshold
    n_messages = len(graph.message_nodes)
    if n_messages < 2:
        print("\nNot enough messages to cluster.\n")
        return

    # Use greedy clustering based on similarity
    clusters = _greedy_cluster(graph, k)

    print(f"\nMessage Clusters ({len(clusters)} clusters):")
    print("=" * 50)

    for cluster_id, members in enumerate(clusters):
        if not members:
            continue

        # Get representative (first member)
        rep_idx = members[0]
        rep_node = graph.message_nodes[rep_idx]
        rep_content = rep_node.content[:60] + "..." if len(rep_node.content) > 60 else rep_node.content
        rep_content = rep_content.replace("\n", " ")

        print(f"\nCluster {cluster_id + 1} ({len(members)} messages):")
        print(f"  Representative: {rep_content}")
        print(f"  Members: {members[:10]}{'...' if len(members) > 10 else ''}")

    print()


def _greedy_cluster(graph, k=None):
    """Simple greedy clustering based on similarity.

    Args:
        graph: SimilarityGraph
        k: Number of clusters (None for auto-detect)

    Returns:
        List of clusters, where each cluster is a list of message indices
    """
    import numpy as np

    n = len(graph.message_nodes)
    if n == 0:
        return []

    # Default: aim for clusters of ~5 messages
    if k is None:
        k = max(2, n // 5)

    # Initialize: each message is its own cluster
    sim_matrix = graph.msg_to_msg
    assigned = [-1] * n  # cluster assignment for each message
    clusters = []

    # Greedy: find seeds (messages with lowest average similarity - most distinct)
    avg_sims = np.mean(sim_matrix, axis=1)
    seeds = np.argsort(avg_sims)[:k]

    for cluster_id, seed_idx in enumerate(seeds):
        assigned[seed_idx] = cluster_id
        clusters.append([seed_idx])

    # Assign remaining messages to nearest cluster
    for msg_idx in range(n):
        if assigned[msg_idx] >= 0:
            continue

        # Find most similar seed
        best_cluster = 0
        best_sim = -1
        for cluster_id, seed_idx in enumerate(seeds):
            sim = sim_matrix[msg_idx, seed_idx]
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster_id

        assigned[msg_idx] = best_cluster
        clusters[best_cluster].append(msg_idx)

    # Filter out empty clusters
    clusters = [c for c in clusters if c]

    return clusters
