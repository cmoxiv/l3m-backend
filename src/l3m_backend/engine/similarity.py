"""Similarity graph for messages and summaries.

Provides semantic similarity computation using LLM embeddings for:
- Smart context retrieval
- Deduplication
- Relevance ranking
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        ...


@dataclass
class SimilarityNode:
    """A node in the similarity graph."""

    id: str
    node_type: Literal["message", "summary"]
    content: str
    index: int
    embedding: np.ndarray | None = None


@dataclass
class SimilarityGraph:
    """Graph structure for message and summary similarities.

    Stores pairwise cosine similarities between:
    - Messages and messages
    - Messages and summaries
    - Summaries and summaries

    Usage:
        graph = SimilarityGraph()
        graph.build(messages, summaries, embedding_provider)

        # Find similar messages
        similar = graph.get_similar_messages(query_embedding, k=5)

        # Find duplicates
        duplicates = graph.find_duplicates(threshold=0.9)
    """

    similarity_threshold: float = 0.3
    message_nodes: list[SimilarityNode] = field(default_factory=list)
    summary_nodes: list[SimilarityNode] = field(default_factory=list)

    # Similarity matrices
    msg_to_msg: np.ndarray | None = None
    msg_to_summary: np.ndarray | None = None
    summary_to_summary: np.ndarray | None = None

    # Cached embeddings
    _message_embeddings: np.ndarray | None = None
    _summary_embeddings: np.ndarray | None = None
    _embedding_dim: int | None = None

    def build(
        self,
        messages: list[dict[str, Any]],
        summaries: list[str],
        provider: EmbeddingProvider,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Build the similarity graph from messages and summaries.

        Args:
            messages: List of message dicts with 'role' and 'content'
            summaries: List of summary strings
            provider: Embedding provider for generating vectors
            progress_callback: Optional callback(current, total) for progress updates
        """
        logger.info(f"Building similarity graph: {len(messages)} messages, {len(summaries)} summaries")

        total_items = len(messages) + len(summaries)
        current = 0

        # Extract message content
        message_texts = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            role = msg.get("role", "user")
            # Prefix with role for better embeddings
            text = f"{role}: {content}" if content else f"{role}: [empty]"
            message_texts.append(text)

            self.message_nodes.append(SimilarityNode(
                id=f"msg_{i}",
                node_type="message",
                content=content,
                index=i,
            ))
            current += 1
            if progress_callback:
                progress_callback(current, total_items)

        # Create summary nodes
        for i, summary in enumerate(summaries):
            self.summary_nodes.append(SimilarityNode(
                id=f"sum_{i}",
                node_type="summary",
                content=summary,
                index=i,
            ))
            current += 1
            if progress_callback:
                progress_callback(current, total_items)

        # Compute embeddings
        all_texts = message_texts + summaries
        if not all_texts:
            logger.warning("No texts to embed, skipping similarity computation")
            return

        logger.debug(f"Computing embeddings for {len(all_texts)} texts")
        try:
            all_embeddings = provider.embed(all_texts)
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            raise
        self._embedding_dim = all_embeddings.shape[1] if len(all_embeddings.shape) > 1 else None

        # Split embeddings
        n_messages = len(message_texts)
        if n_messages > 0:
            self._message_embeddings = all_embeddings[:n_messages]
            # Store in nodes
            for i, node in enumerate(self.message_nodes):
                node.embedding = self._message_embeddings[i]

        if summaries:
            self._summary_embeddings = all_embeddings[n_messages:]
            for i, node in enumerate(self.summary_nodes):
                node.embedding = self._summary_embeddings[i]

        # Compute similarity matrices
        self._compute_similarities()

        logger.info(f"Similarity graph built: dim={self._embedding_dim}")

    def _compute_similarities(self) -> None:
        """Compute all pairwise similarity matrices."""
        # Message to message
        if self._message_embeddings is not None and len(self._message_embeddings) > 0:
            self.msg_to_msg = self._cosine_similarity_matrix(
                self._message_embeddings,
                self._message_embeddings,
            )

        # Message to summary
        if (self._message_embeddings is not None and len(self._message_embeddings) > 0 and
            self._summary_embeddings is not None and len(self._summary_embeddings) > 0):
            self.msg_to_summary = self._cosine_similarity_matrix(
                self._message_embeddings,
                self._summary_embeddings,
            )

        # Summary to summary
        if self._summary_embeddings is not None and len(self._summary_embeddings) > 0:
            self.summary_to_summary = self._cosine_similarity_matrix(
                self._summary_embeddings,
                self._summary_embeddings,
            )

    @staticmethod
    def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between all pairs of vectors.

        Args:
            a: Matrix of shape (n, d)
            b: Matrix of shape (m, d)

        Returns:
            Similarity matrix of shape (n, m)
        """
        # Normalize vectors
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        # Compute dot product
        return np.dot(a_norm, b_norm.T)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # === Retrieval API ===

    def get_similar_messages(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """Find k most similar messages to a query embedding.

        Args:
            query_embedding: Query vector
            k: Number of results to return

        Returns:
            List of (message_index, similarity_score) tuples, sorted by similarity
        """
        if self._message_embeddings is None or len(self._message_embeddings) == 0:
            return []

        # Compute similarities to query
        similarities = []
        for i, emb in enumerate(self._message_embeddings):
            sim = self._cosine_similarity(query_embedding, emb)
            similarities.append((i, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def get_similar_summaries(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
    ) -> list[tuple[int, float]]:
        """Find k most similar summaries to a query embedding.

        Args:
            query_embedding: Query vector
            k: Number of results to return

        Returns:
            List of (summary_index, similarity_score) tuples
        """
        if self._summary_embeddings is None or len(self._summary_embeddings) == 0:
            return []

        similarities = []
        for i, emb in enumerate(self._summary_embeddings):
            sim = self._cosine_similarity(query_embedding, emb)
            similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def get_relevant_context(
        self,
        query_embedding: np.ndarray,
        max_messages: int = 10,
        max_summaries: int = 3,
    ) -> tuple[list[int], list[int]]:
        """Get relevant messages and summaries for a query.

        Args:
            query_embedding: Query vector
            max_messages: Maximum message indices to return
            max_summaries: Maximum summary indices to return

        Returns:
            Tuple of (message_indices, summary_indices) for relevant context
        """
        msg_results = self.get_similar_messages(query_embedding, k=max_messages)
        sum_results = self.get_similar_summaries(query_embedding, k=max_summaries)

        # Filter by threshold
        msg_indices = [idx for idx, sim in msg_results if sim >= self.similarity_threshold]
        sum_indices = [idx for idx, sim in sum_results if sim >= self.similarity_threshold]

        return msg_indices, sum_indices

    # === Deduplication API ===

    def find_duplicates(
        self,
        threshold: float = 0.9,
    ) -> list[tuple[int, int, float]]:
        """Find near-duplicate message pairs.

        Args:
            threshold: Minimum similarity to consider as duplicate

        Returns:
            List of (msg_idx_1, msg_idx_2, similarity) for duplicates
        """
        if self.msg_to_msg is None:
            return []

        duplicates = []
        n = self.msg_to_msg.shape[0]

        for i in range(n):
            for j in range(i + 1, n):  # Only upper triangle
                sim = self.msg_to_msg[i, j]
                if sim >= threshold:
                    duplicates.append((i, j, float(sim)))

        # Sort by similarity descending
        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates

    def get_unique_messages(
        self,
        threshold: float = 0.9,
    ) -> list[int]:
        """Get indices of unique (non-duplicate) messages.

        Keeps the first occurrence of each duplicate group.

        Args:
            threshold: Similarity threshold for duplicate detection

        Returns:
            List of message indices representing unique content
        """
        if self.msg_to_msg is None:
            return list(range(len(self.message_nodes)))

        duplicates = self.find_duplicates(threshold)

        # Track which indices to exclude
        excluded = set()
        for i, j, _ in duplicates:
            # Keep earlier message, exclude later
            excluded.add(j)

        # Return indices not in excluded set
        return [i for i in range(len(self.message_nodes)) if i not in excluded]

    # === Ranking API ===

    def rank_messages_by_relevance(
        self,
        query_embedding: np.ndarray,
        message_indices: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Rank messages by relevance to a query.

        Args:
            query_embedding: Query vector
            message_indices: Optional subset of messages to rank (default: all)

        Returns:
            List of (message_index, relevance_score) sorted by score descending
        """
        if self._message_embeddings is None:
            return []

        if message_indices is None:
            message_indices = list(range(len(self.message_nodes)))

        results = []
        for idx in message_indices:
            if 0 <= idx < len(self._message_embeddings):
                sim = self._cosine_similarity(query_embedding, self._message_embeddings[idx])
                results.append((idx, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # === Graph Inspection ===

    def get_message_neighbors(
        self,
        message_idx: int,
        min_similarity: float | None = None,
        k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Get neighboring messages above similarity threshold.

        Args:
            message_idx: Index of the message
            min_similarity: Minimum similarity to include (default: self.similarity_threshold)
            k: Maximum number of neighbors (default: all above threshold)

        Returns:
            List of (neighbor_idx, similarity) tuples
        """
        if self.msg_to_msg is None or message_idx >= len(self.message_nodes):
            return []

        if min_similarity is None:
            min_similarity = self.similarity_threshold

        neighbors = []
        for i, sim in enumerate(self.msg_to_msg[message_idx]):
            if i != message_idx and sim >= min_similarity:
                neighbors.append((i, float(sim)))

        neighbors.sort(key=lambda x: x[1], reverse=True)

        if k is not None:
            return neighbors[:k]
        return neighbors

    @property
    def stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        num_edges = 0
        avg_similarity = 0.0

        if self.msg_to_msg is not None:
            # Count edges above threshold (excluding diagonal)
            mask = self.msg_to_msg >= self.similarity_threshold
            np.fill_diagonal(mask, False)
            num_edges += int(np.sum(mask)) // 2  # Divide by 2 for undirected

            # Average similarity (excluding diagonal)
            n = self.msg_to_msg.shape[0]
            if n > 1:
                total = np.sum(self.msg_to_msg) - np.trace(self.msg_to_msg)
                avg_similarity = float(total / (n * (n - 1)))

        return {
            "num_messages": len(self.message_nodes),
            "num_summaries": len(self.summary_nodes),
            "num_edges": num_edges,
            "avg_msg_similarity": round(avg_similarity, 4),
            "embedding_dim": self._embedding_dim,
            "threshold": self.similarity_threshold,
        }

    @property
    def is_built(self) -> bool:
        """Check if the graph has been built."""
        return self._message_embeddings is not None or self._summary_embeddings is not None


class LlamaEmbeddingProvider:
    """Embedding provider using llama-cpp-python.

    Requires the Llama model to be loaded with embedding=True.
    """

    def __init__(self, llm: "Llama"):
        """Initialize with a Llama instance.

        Args:
            llm: Llama instance (must be loaded with embedding=True)
        """
        self.llm = llm

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using the Llama model.

        Embeds one text at a time to avoid context overflow.

        Args:
            texts: List of strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        embeddings = []
        target_dim = None

        for text in texts:
            try:
                # Reset KV cache before each embedding
                if hasattr(self.llm, 'reset'):
                    self.llm.reset()

                result = self.llm.create_embedding([text])
                emb = np.array(result["data"][0]["embedding"], dtype=np.float32)

                # Set target dimension from first successful embedding
                if target_dim is None:
                    target_dim = len(emb)

                # Ensure consistent dimension
                if len(emb) != target_dim:
                    logger.warning(f"Embedding dimension mismatch: {len(emb)} vs {target_dim}")
                    # Truncate or pad to target dimension
                    if len(emb) > target_dim:
                        emb = emb[:target_dim]
                    else:
                        emb = np.pad(emb, (0, target_dim - len(emb)))

                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Embedding failed for text: {e}")
                # Use zero vector as fallback
                if target_dim is not None:
                    embeddings.append(np.zeros(target_dim, dtype=np.float32))
                else:
                    raise

        return np.vstack(embeddings) if embeddings else np.array([])

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text.

        Args:
            text: String to embed

        Returns:
            Embedding vector
        """
        result = self.embed([text])
        return result[0] if result.size > 0 else np.array([], dtype=np.float32)


class NomicEmbeddingProvider:
    """Embedding provider using Nomic's embedding model.

    Uses sentence-transformers with nomic-embed-text-v1.5.
    Much faster and better quality for semantic similarity than LLM embeddings.

    Install: pip install sentence-transformers
    """

    # Default model - can be overridden
    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"

    def __init__(self, model_name: str | None = None, device: str | None = None):
        """Initialize with Nomic embedding model.

        Args:
            model_name: Model name (default: nomic-embed-text-v1.5)
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required for Nomic embeddings. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device

        logger.info(f"Loading Nomic embedding model: {self.model_name}")
        self.model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,
            device=device,
        )
        self._dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Nomic embeddings ready: dim={self._dim}, device={self.model.device}")

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using Nomic model.

        Args:
            texts: List of strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        # Nomic recommends prefixing with task type
        # "search_document:" for documents, "search_query:" for queries
        prefixed = [f"search_document: {t}" for t in texts]

        embeddings = self.model.encode(
            prefixed,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings

    def embed_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """Embed a single text.

        Args:
            text: String to embed
            is_query: If True, use query prefix (for similarity search)

        Returns:
            Embedding vector
        """
        prefix = "search_query:" if is_query else "search_document:"
        embedding = self.model.encode(
            f"{prefix} {text}",
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding


def get_embedding_provider(
    llm: "Llama | None" = None,
    use_nomic: bool = True,
    nomic_model: str | None = None,
    device: str | None = None,
) -> EmbeddingProvider | None:
    """Get the best available embedding provider.

    Prefers Nomic if available and use_nomic=True, falls back to LLM.

    Args:
        llm: Llama instance for LLM-based embeddings
        use_nomic: Whether to prefer Nomic embeddings
        nomic_model: Optional Nomic model name override
        device: Device for Nomic ('cpu', 'cuda', 'mps')

    Returns:
        EmbeddingProvider instance, or None if no provider available
    """
    if use_nomic:
        try:
            return NomicEmbeddingProvider(model_name=nomic_model, device=device)
        except ImportError:
            logger.debug("Nomic not available, falling back to LLM embeddings")
        except Exception as e:
            logger.warning(f"Failed to load Nomic: {e}, falling back to LLM")

    if llm is not None:
        return LlamaEmbeddingProvider(llm)

    return None
