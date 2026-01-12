"""Tests for the similarity graph module."""

import numpy as np
import pytest

from l3m_backend.engine.similarity import (
    LlamaEmbeddingProvider,
    SimilarityGraph,
    SimilarityNode,
)


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(self, dim: int = 128, deterministic: bool = False):
        self.dim = dim
        self.deterministic = deterministic
        self._call_count = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings - deterministic based on text if requested."""
        if self.deterministic:
            # Generate deterministic embeddings based on text hash
            embeddings = []
            for text in texts:
                np.random.seed(hash(text) % (2**32))
                embeddings.append(np.random.randn(self.dim))
            return np.array(embeddings)
        else:
            self._call_count += 1
            return np.random.randn(len(texts), self.dim)


class TestSimilarityNode:
    """Tests for SimilarityNode dataclass."""

    def test_create_message_node(self):
        """Test creating a message node."""
        node = SimilarityNode(
            id="msg_0",
            node_type="message",
            content="Hello world",
            index=0,
        )
        assert node.id == "msg_0"
        assert node.node_type == "message"
        assert node.content == "Hello world"
        assert node.index == 0
        assert node.embedding is None

    def test_create_summary_node(self):
        """Test creating a summary node."""
        node = SimilarityNode(
            id="sum_0",
            node_type="summary",
            content="A conversation about greetings",
            index=0,
        )
        assert node.node_type == "summary"

    def test_node_with_embedding(self):
        """Test node with embedding attached."""
        emb = np.random.randn(128)
        node = SimilarityNode(
            id="msg_0",
            node_type="message",
            content="Test",
            index=0,
            embedding=emb,
        )
        assert node.embedding is not None
        assert len(node.embedding) == 128


class TestSimilarityGraph:
    """Tests for SimilarityGraph class."""

    def test_empty_graph(self):
        """Test creating an empty graph."""
        graph = SimilarityGraph()
        assert len(graph.message_nodes) == 0
        assert len(graph.summary_nodes) == 0
        assert not graph.is_built
        assert graph.stats["num_messages"] == 0

    def test_build_with_messages_only(self):
        """Test building graph with only messages."""
        graph = SimilarityGraph()
        provider = MockEmbeddingProvider(dim=64)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        graph.build(messages, [], provider)

        assert len(graph.message_nodes) == 3
        assert len(graph.summary_nodes) == 0
        assert graph.is_built
        assert graph.msg_to_msg is not None
        assert graph.msg_to_msg.shape == (3, 3)
        assert graph.msg_to_summary is None

    def test_build_with_summaries_only(self):
        """Test building graph with only summaries."""
        graph = SimilarityGraph()
        provider = MockEmbeddingProvider(dim=64)

        summaries = ["First summary", "Second summary"]

        graph.build([], summaries, provider)

        assert len(graph.message_nodes) == 0
        assert len(graph.summary_nodes) == 2
        assert graph.is_built
        assert graph.summary_to_summary is not None
        assert graph.summary_to_summary.shape == (2, 2)

    def test_build_with_messages_and_summaries(self):
        """Test building graph with both messages and summaries."""
        graph = SimilarityGraph()
        provider = MockEmbeddingProvider(dim=64)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        summaries = ["A greeting"]

        graph.build(messages, summaries, provider)

        assert len(graph.message_nodes) == 2
        assert len(graph.summary_nodes) == 1
        assert graph.msg_to_msg.shape == (2, 2)
        assert graph.msg_to_summary.shape == (2, 1)
        assert graph.summary_to_summary.shape == (1, 1)

    def test_build_empty_data(self):
        """Test building graph with no data."""
        graph = SimilarityGraph()
        provider = MockEmbeddingProvider()

        graph.build([], [], provider)

        assert not graph.is_built
        assert graph.stats["num_messages"] == 0

    def test_similarity_threshold(self):
        """Test custom similarity threshold."""
        graph = SimilarityGraph(similarity_threshold=0.8)
        assert graph.similarity_threshold == 0.8

    def test_stats(self):
        """Test graph statistics."""
        graph = SimilarityGraph()
        provider = MockEmbeddingProvider(dim=64)

        messages = [{"role": "user", "content": f"Message {i}"} for i in range(5)]
        summaries = ["Summary 1", "Summary 2"]

        graph.build(messages, summaries, provider)

        stats = graph.stats
        assert stats["num_messages"] == 5
        assert stats["num_summaries"] == 2
        assert stats["embedding_dim"] == 64
        assert "avg_msg_similarity" in stats
        assert "num_edges" in stats


class TestSimilarityGraphRetrieval:
    """Tests for retrieval APIs."""

    @pytest.fixture
    def built_graph(self):
        """Create a built graph for testing."""
        graph = SimilarityGraph(similarity_threshold=0.0)  # Low threshold for testing
        provider = MockEmbeddingProvider(dim=64, deterministic=True)

        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there, how can I help?"},
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language"},
            {"role": "user", "content": "Hello again"},  # Similar to first message
        ]
        summaries = ["Greeting and Python discussion"]

        graph.build(messages, summaries, provider)
        return graph, provider

    def test_get_similar_messages(self, built_graph):
        """Test finding similar messages."""
        graph, provider = built_graph

        # Use embedding of first message as query
        query_emb = graph.message_nodes[0].embedding

        results = graph.get_similar_messages(query_emb, k=3)

        assert len(results) == 3
        # First result should be the message itself (highest similarity)
        assert results[0][0] == 0
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_get_similar_summaries(self, built_graph):
        """Test finding similar summaries."""
        graph, provider = built_graph

        query_emb = np.random.randn(64)
        results = graph.get_similar_summaries(query_emb, k=5)

        assert len(results) == 1  # Only 1 summary

    def test_get_relevant_context(self, built_graph):
        """Test getting relevant context."""
        graph, provider = built_graph

        query_emb = graph.message_nodes[0].embedding

        msg_indices, sum_indices = graph.get_relevant_context(
            query_emb, max_messages=3, max_summaries=2
        )

        assert len(msg_indices) <= 3
        assert len(sum_indices) <= 2

    def test_empty_graph_retrieval(self):
        """Test retrieval on empty graph."""
        graph = SimilarityGraph()
        query_emb = np.random.randn(64)

        assert graph.get_similar_messages(query_emb) == []
        assert graph.get_similar_summaries(query_emb) == []


class TestSimilarityGraphDeduplication:
    """Tests for deduplication APIs."""

    def test_find_duplicates_none(self):
        """Test finding duplicates when there are none."""
        graph = SimilarityGraph()
        provider = MockEmbeddingProvider(dim=64)

        # Distinct messages
        messages = [
            {"role": "user", "content": "First unique message"},
            {"role": "assistant", "content": "Second unique response"},
            {"role": "user", "content": "Third unique question"},
        ]

        graph.build(messages, [], provider)

        # With high threshold, random embeddings unlikely to be duplicates
        duplicates = graph.find_duplicates(threshold=0.99)
        assert len(duplicates) == 0

    def test_find_duplicates_with_similar(self):
        """Test finding duplicates with similar content."""
        graph = SimilarityGraph()
        provider = MockEmbeddingProvider(dim=64, deterministic=True)

        # Messages where two are identical
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Hello world"},  # Duplicate of first
        ]

        graph.build(messages, [], provider)

        # With deterministic embeddings, identical text = identical embedding = similarity 1.0
        duplicates = graph.find_duplicates(threshold=0.99)
        assert len(duplicates) >= 1
        # Check that indices 0 and 2 are marked as duplicates
        dup_pairs = [(d[0], d[1]) for d in duplicates]
        assert (0, 2) in dup_pairs or (2, 0) in dup_pairs

    def test_get_unique_messages(self):
        """Test getting unique message indices."""
        graph = SimilarityGraph()
        provider = MockEmbeddingProvider(dim=64, deterministic=True)

        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Hello world"},  # Duplicate
        ]

        graph.build(messages, [], provider)

        unique = graph.get_unique_messages(threshold=0.99)
        # Should keep first occurrence, exclude duplicate
        assert 0 in unique
        assert 1 in unique
        assert 2 not in unique

    def test_empty_graph_deduplication(self):
        """Test deduplication on empty graph."""
        graph = SimilarityGraph()

        assert graph.find_duplicates() == []
        assert graph.get_unique_messages() == []


class TestSimilarityGraphRanking:
    """Tests for ranking APIs."""

    def test_rank_messages_by_relevance(self):
        """Test ranking messages by relevance."""
        graph = SimilarityGraph()
        provider = MockEmbeddingProvider(dim=64, deterministic=True)

        messages = [
            {"role": "user", "content": "Python programming"},
            {"role": "assistant", "content": "JavaScript coding"},
            {"role": "user", "content": "Python programming"},  # Same as first
        ]

        graph.build(messages, [], provider)

        # Query with first message's embedding
        query_emb = graph.message_nodes[0].embedding
        ranked = graph.rank_messages_by_relevance(query_emb)

        assert len(ranked) == 3
        # First message should rank highest with itself
        assert ranked[0][0] == 0 or ranked[0][0] == 2  # Either identical message
        assert ranked[0][1] == pytest.approx(1.0, abs=0.01)

    def test_rank_subset_of_messages(self):
        """Test ranking a subset of messages."""
        graph = SimilarityGraph()
        provider = MockEmbeddingProvider(dim=64)

        messages = [{"role": "user", "content": f"Msg {i}"} for i in range(5)]
        graph.build(messages, [], provider)

        query_emb = np.random.randn(64)
        ranked = graph.rank_messages_by_relevance(query_emb, message_indices=[0, 2, 4])

        assert len(ranked) == 3
        assert all(idx in [0, 2, 4] for idx, _ in ranked)

    def test_empty_graph_ranking(self):
        """Test ranking on empty graph."""
        graph = SimilarityGraph()
        query_emb = np.random.randn(64)

        assert graph.rank_messages_by_relevance(query_emb) == []


class TestSimilarityGraphNeighbors:
    """Tests for neighbor lookup APIs."""

    def test_get_message_neighbors(self):
        """Test getting neighboring messages."""
        graph = SimilarityGraph(similarity_threshold=0.0)
        provider = MockEmbeddingProvider(dim=64)

        messages = [{"role": "user", "content": f"Msg {i}"} for i in range(5)]
        graph.build(messages, [], provider)

        neighbors = graph.get_message_neighbors(0, min_similarity=-1.0)  # Get all
        assert len(neighbors) == 4  # All except self

    def test_get_message_neighbors_with_k(self):
        """Test getting top-k neighbors."""
        graph = SimilarityGraph()
        provider = MockEmbeddingProvider(dim=64)

        messages = [{"role": "user", "content": f"Msg {i}"} for i in range(10)]
        graph.build(messages, [], provider)

        neighbors = graph.get_message_neighbors(0, min_similarity=-1.0, k=3)
        assert len(neighbors) == 3

    def test_get_message_neighbors_invalid_index(self):
        """Test neighbors with invalid index."""
        graph = SimilarityGraph()
        provider = MockEmbeddingProvider(dim=64)

        messages = [{"role": "user", "content": "Hello"}]
        graph.build(messages, [], provider)

        neighbors = graph.get_message_neighbors(999)
        assert neighbors == []


class TestCosineSimliarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        vec = np.array([1.0, 2.0, 3.0])
        sim = SimilarityGraph._cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        sim = SimilarityGraph._cosine_similarity(vec1, vec2)
        assert sim == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        vec1 = np.array([1.0, 1.0])
        vec2 = np.array([-1.0, -1.0])
        sim = SimilarityGraph._cosine_similarity(vec1, vec2)
        assert sim == pytest.approx(-1.0)

    def test_zero_vector(self):
        """Test similarity with zero vector."""
        vec1 = np.array([1.0, 2.0])
        vec2 = np.array([0.0, 0.0])
        sim = SimilarityGraph._cosine_similarity(vec1, vec2)
        assert sim == 0.0

    def test_similarity_matrix(self):
        """Test similarity matrix computation."""
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        sim = SimilarityGraph._cosine_similarity_matrix(a, b)

        assert sim.shape == (2, 3)
        assert sim[0, 0] == pytest.approx(1.0)  # [1,0] vs [1,0]
        assert sim[0, 1] == pytest.approx(0.0)  # [1,0] vs [0,1]
        assert sim[1, 1] == pytest.approx(1.0)  # [0,1] vs [0,1]
