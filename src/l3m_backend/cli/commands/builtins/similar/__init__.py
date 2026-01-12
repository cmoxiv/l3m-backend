"""Similar command - semantic search across conversation history."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register(
    "similar",
    "Semantic search in conversation",
    usage="/similar <query> [--top N] [--summaries]",
    has_args=True,
    aliases=["sim", "search"],
)
def cmd_similar(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Find messages semantically similar to a query.

    Usage:
        /similar error handling     - Find messages about error handling
        /similar --top 10 auth      - Show top 10 results
        /similar --summaries login  - Also search summaries
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
    top_k = 5
    search_summaries = False
    query_parts = []

    i = 0
    while i < len(parts):
        if parts[i] == "--top" and i + 1 < len(parts):
            try:
                top_k = int(parts[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        elif parts[i] == "--summaries":
            search_summaries = True
            i += 1
            continue
        query_parts.append(parts[i])
        i += 1

    query = " ".join(query_parts)
    if not query:
        print("\nUsage: /similar <query> [--top N] [--summaries]")
        print("Example: /similar error handling\n")
        return

    # Get embedding provider
    provider = engine._get_embedding_provider()
    if provider is None:
        print("\nNo embedding provider available.\n")
        return

    # Embed query
    if hasattr(provider, "embed_single"):
        # Nomic provider has is_query parameter
        if hasattr(provider, "model"):  # NomicEmbeddingProvider
            query_emb = provider.embed_single(query, is_query=True)
        else:
            query_emb = provider.embed_single(query)
    else:
        query_emb = provider.embed([query])[0]

    # Search messages
    print(f"\nSearching for: \"{query}\"\n")

    msg_results = graph.get_similar_messages(query_emb, k=top_k)
    if msg_results:
        print(f"Top {len(msg_results)} similar messages:")
        print("-" * 50)
        for idx, score in msg_results:
            if idx < len(graph.message_nodes):
                node = graph.message_nodes[idx]
                content = node.content[:100] + "..." if len(node.content) > 100 else node.content
                content = content.replace("\n", " ")
                print(f"  [{idx}] ({score:.2f}) {content}")
        print()

    # Search summaries if requested
    if search_summaries:
        sum_results = graph.get_similar_summaries(query_emb, k=min(top_k, 3))
        if sum_results:
            print(f"Similar summaries:")
            print("-" * 50)
            for idx, score in sum_results:
                if idx < len(graph.summary_nodes):
                    node = graph.summary_nodes[idx]
                    content = node.content[:100] + "..." if len(node.content) > 100 else node.content
                    content = content.replace("\n", " ")
                    print(f"  [S{idx}] ({score:.2f}) {content}")
            print()

    if not msg_results and not (search_summaries and sum_results):
        print("No similar messages found.\n")
