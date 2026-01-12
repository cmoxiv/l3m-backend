"""Relevant command - find context relevant to current topic."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register(
    "relevant",
    "Find context for current topic",
    usage="/relevant [query]",
    has_args=True,
    aliases=["rel", "context"],
)
def cmd_relevant(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Find messages and summaries relevant to current conversation topic.

    If no query provided, uses the last few messages to determine topic.

    Usage:
        /relevant              - Find context for current topic
        /relevant error logs   - Find context about error logs
    """
    if not hasattr(engine, "similarity_graph") or engine.similarity_graph is None:
        print("\nNo similarity graph available.")
        print("Resume a session with similarity graph enabled.\n")
        return

    graph = engine.similarity_graph
    if not graph.is_built:
        print("\nSimilarity graph not built yet.\n")
        return

    # Get embedding provider
    provider = engine._get_embedding_provider()
    if provider is None:
        print("\nNo embedding provider available.\n")
        return

    # Determine query
    query = args.strip()
    if not query:
        # Use last few messages as context
        if engine.history:
            recent = engine.history[-3:]
            query = " ".join(msg.get("content", "")[:200] for msg in recent)
        else:
            print("\nNo query provided and no history available.")
            print("Usage: /relevant [query]\n")
            return

    # Embed query
    if hasattr(provider, "embed_single"):
        if hasattr(provider, "model"):  # NomicEmbeddingProvider
            query_emb = provider.embed_single(query, is_query=True)
        else:
            query_emb = provider.embed_single(query)
    else:
        query_emb = provider.embed([query])[0]

    # Get relevant context
    msg_indices, sum_indices = graph.get_relevant_context(
        query_emb,
        max_messages=10,
        max_summaries=3,
    )

    if args.strip():
        print(f"\nRelevant context for: \"{args.strip()[:50]}...\"" if len(args.strip()) > 50 else f"\nRelevant context for: \"{args.strip()}\"")
    else:
        print("\nRelevant context for current topic:")
    print("=" * 50)

    # Show relevant messages
    if msg_indices:
        print(f"\nRelevant Messages ({len(msg_indices)}):")
        print("-" * 40)
        for idx in msg_indices[:10]:
            if idx < len(graph.message_nodes):
                node = graph.message_nodes[idx]
                content = node.content[:100] + "..." if len(node.content) > 100 else node.content
                content = content.replace("\n", " ")
                print(f"  [{idx}] {content}")
    else:
        print("\nNo relevant messages found.")

    # Show relevant summaries
    if sum_indices:
        print(f"\nRelevant Summaries ({len(sum_indices)}):")
        print("-" * 40)
        for idx in sum_indices:
            if idx < len(graph.summary_nodes):
                node = graph.summary_nodes[idx]
                content = node.content[:150] + "..." if len(node.content) > 150 else node.content
                content = content.replace("\n", " ")
                print(f"  [S{idx}] {content}")

    # Show what would be included in context
    total_chars = 0
    for idx in msg_indices[:10]:
        if idx < len(graph.message_nodes):
            total_chars += len(graph.message_nodes[idx].content)
    for idx in sum_indices:
        if idx < len(graph.summary_nodes):
            total_chars += len(graph.summary_nodes[idx].content)

    print(f"\nTotal context: ~{total_chars} chars, ~{total_chars // 4} tokens")
    print()
