"""Tests for the knowledge graph module."""

from datetime import datetime

import pytest

from l3m_backend.engine.knowledge import (
    Entity,
    EntityType,
    KnowledgeEdge,
    KnowledgeGraph,
    RelationType,
)
from l3m_backend.engine.extraction import EntityExtractor


class TestEntity:
    """Tests for Entity dataclass."""

    def test_create_entity(self):
        """Test creating an entity."""
        entity = Entity(
            id="person_john",
            name="John",
            entity_type=EntityType.PERSON,
            context="User mentioned in chat",
        )
        assert entity.id == "person_john"
        assert entity.name == "John"
        assert entity.entity_type == EntityType.PERSON
        assert entity.mention_count == 1

    def test_entity_serialization(self):
        """Test entity to_dict and from_dict."""
        entity = Entity(
            id="function_get_weather",
            name="get_weather",
            entity_type=EntityType.FUNCTION,
            context="Weather fetching function",
            first_seen_msg="msg_0",
            mention_count=3,
        )

        data = entity.to_dict()
        restored = Entity.from_dict(data)

        assert restored.id == entity.id
        assert restored.name == entity.name
        assert restored.entity_type == entity.entity_type
        assert restored.context == entity.context
        assert restored.mention_count == 3


class TestKnowledgeEdge:
    """Tests for KnowledgeEdge dataclass."""

    def test_create_edge(self):
        """Test creating an edge."""
        edge = KnowledgeEdge(
            source_id="person_john",
            target_id="topic_python",
            relation_type=RelationType.MENTIONS,
            weight=0.8,
            message_ids=["msg_1", "msg_2"],
        )
        assert edge.source_id == "person_john"
        assert edge.target_id == "topic_python"
        assert edge.relation_type == RelationType.MENTIONS
        assert edge.weight == 0.8
        assert len(edge.message_ids) == 2

    def test_edge_serialization(self):
        """Test edge to_dict and from_dict."""
        edge = KnowledgeEdge(
            source_id="class_chatengine",
            target_id="class_session",
            relation_type=RelationType.DEPENDS_ON,
            weight=0.9,
        )

        data = edge.to_dict()
        restored = KnowledgeEdge.from_dict(data)

        assert restored.source_id == edge.source_id
        assert restored.target_id == edge.target_id
        assert restored.relation_type == edge.relation_type
        assert restored.weight == edge.weight


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph class."""

    def test_empty_graph(self):
        """Test creating an empty graph."""
        graph = KnowledgeGraph()
        assert len(graph.entities) == 0
        assert len(graph.edges) == 0
        assert graph.stats["num_entities"] == 0

    def test_add_entity(self):
        """Test adding entities."""
        graph = KnowledgeGraph()

        entity = graph.add_entity(
            name="ChatEngine",
            entity_type=EntityType.CLASS,
            context="Main chat engine class",
            message_id="msg_0",
        )

        assert entity.name == "ChatEngine"
        assert "class_chatengine" in graph.entities
        assert len(graph.entities) == 1

    def test_add_duplicate_entity(self):
        """Test adding duplicate entity increments mention count."""
        graph = KnowledgeGraph()

        entity1 = graph.add_entity(
            name="ChatEngine",
            entity_type=EntityType.CLASS,
            message_id="msg_0",
        )

        entity2 = graph.add_entity(
            name="ChatEngine",
            entity_type=EntityType.CLASS,
            message_id="msg_1",
        )

        # Should be same entity
        assert entity1.id == entity2.id
        assert len(graph.entities) == 1
        assert entity1.mention_count == 2

    def test_add_edge(self):
        """Test adding edges."""
        graph = KnowledgeGraph()

        # Add entities first
        e1 = graph.add_entity("ChatEngine", EntityType.CLASS)
        e2 = graph.add_entity("Session", EntityType.CLASS)

        edge = graph.add_edge(
            source_id=e1.id,
            target_id=e2.id,
            relation_type=RelationType.DEPENDS_ON,
            message_id="msg_0",
        )

        assert edge is not None
        assert len(graph.edges) == 1
        assert edge.source_id == e1.id

    def test_add_edge_nonexistent_entity(self):
        """Test adding edge with nonexistent entity returns None."""
        graph = KnowledgeGraph()
        graph.add_entity("ChatEngine", EntityType.CLASS)

        edge = graph.add_edge(
            source_id="class_chatengine",
            target_id="nonexistent",
            relation_type=RelationType.DEPENDS_ON,
        )

        assert edge is None
        assert len(graph.edges) == 0

    def test_add_entities_from_extraction(self):
        """Test adding entities from extraction result."""
        graph = KnowledgeGraph()

        extraction = {
            "entities": [
                {"name": "ChatEngine", "type": "class", "context": "Main class"},
                {"name": "numpy", "type": "library", "context": "Math library"},
            ],
            "relationships": [
                {"from": "ChatEngine", "to": "numpy", "type": "depends_on"},
            ],
        }

        entities = graph.add_entities_from_extraction(extraction, "msg_0")

        assert len(entities) == 2
        assert len(graph.entities) == 2
        assert len(graph.edges) == 1

    def test_build_co_occurrence_edges(self):
        """Test building co-occurrence edges."""
        graph = KnowledgeGraph()

        # Add multiple entities to same message
        graph.add_entity("Python", EntityType.TOPIC, message_id="msg_0")
        graph.add_entity("numpy", EntityType.LIBRARY, message_id="msg_0")
        graph.add_entity("pandas", EntityType.LIBRARY, message_id="msg_0")

        edges_created = graph.build_co_occurrence_edges()

        # 3 entities = 3 pairs (3 choose 2)
        assert edges_created == 3
        assert len(graph.edges) == 3

    def test_get_entity_by_name(self):
        """Test getting entity by name."""
        graph = KnowledgeGraph()
        graph.add_entity("ChatEngine", EntityType.CLASS)

        entity = graph.get_entity("ChatEngine")
        assert entity is not None
        assert entity.name == "ChatEngine"

        # Case insensitive
        entity = graph.get_entity("chatengine")
        assert entity is not None

        # Not found
        entity = graph.get_entity("nonexistent")
        assert entity is None

    def test_get_entities_by_type(self):
        """Test getting entities by type."""
        graph = KnowledgeGraph()
        graph.add_entity("ChatEngine", EntityType.CLASS)
        graph.add_entity("Session", EntityType.CLASS)
        graph.add_entity("numpy", EntityType.LIBRARY)

        classes = graph.get_entities_by_type(EntityType.CLASS)
        assert len(classes) == 2

        libraries = graph.get_entities_by_type(EntityType.LIBRARY)
        assert len(libraries) == 1

    def test_get_related(self):
        """Test getting related entities."""
        graph = KnowledgeGraph()

        e1 = graph.add_entity("ChatEngine", EntityType.CLASS)
        e2 = graph.add_entity("Session", EntityType.CLASS)
        e3 = graph.add_entity("numpy", EntityType.LIBRARY)

        graph.add_edge(e1.id, e2.id, RelationType.DEPENDS_ON)
        graph.add_edge(e1.id, e3.id, RelationType.DEPENDS_ON)

        related = graph.get_related(e1.id)
        assert len(related) == 2

        # Filter by relation type
        related = graph.get_related(e1.id, RelationType.DEPENDS_ON)
        assert len(related) == 2

        related = graph.get_related(e1.id, RelationType.CO_OCCURS)
        assert len(related) == 0

    def test_get_messages_about(self):
        """Test getting messages about an entity."""
        graph = KnowledgeGraph()

        e1 = graph.add_entity("ChatEngine", EntityType.CLASS, message_id="msg_0")
        # Add to more messages via message_entities
        graph.message_entities["msg_1"] = [e1.id]
        graph.message_entities["msg_2"] = [e1.id]

        messages = graph.get_messages_about(e1.id)
        assert len(messages) == 3
        assert "msg_0" in messages
        assert "msg_1" in messages

    def test_get_entities_in_message(self):
        """Test getting entities in a specific message."""
        graph = KnowledgeGraph()

        graph.add_entity("ChatEngine", EntityType.CLASS, message_id="msg_0")
        graph.add_entity("Session", EntityType.CLASS, message_id="msg_0")
        graph.add_entity("numpy", EntityType.LIBRARY, message_id="msg_1")

        entities = graph.get_entities_in_message("msg_0")
        assert len(entities) == 2

        entities = graph.get_entities_in_message("msg_1")
        assert len(entities) == 1


class TestKnowledgeGraphAnalysis:
    """Tests for knowledge graph analysis methods."""

    def test_get_topic_summary(self):
        """Test getting topic summary."""
        graph = KnowledgeGraph()

        topic = graph.add_entity("error handling", EntityType.TOPIC)
        e1 = graph.add_entity("try/except", EntityType.CONCEPT)
        e2 = graph.add_entity("logging", EntityType.LIBRARY)

        graph.add_edge(topic.id, e1.id, RelationType.MENTIONS)
        graph.add_edge(topic.id, e2.id, RelationType.MENTIONS)

        summary = graph.get_topic_summary()

        assert "error handling" in summary
        assert len(summary["error handling"]) == 3  # topic + 2 related

    def test_find_paths(self):
        """Test finding paths between entities."""
        graph = KnowledgeGraph()

        e1 = graph.add_entity("A", EntityType.CONCEPT)
        e2 = graph.add_entity("B", EntityType.CONCEPT)
        e3 = graph.add_entity("C", EntityType.CONCEPT)
        e4 = graph.add_entity("D", EntityType.CONCEPT)

        # A -> B -> C -> D
        graph.add_edge(e1.id, e2.id, RelationType.FOLLOWS)
        graph.add_edge(e2.id, e3.id, RelationType.FOLLOWS)
        graph.add_edge(e3.id, e4.id, RelationType.FOLLOWS)

        paths = graph.find_paths(e1.id, e3.id, max_depth=3)
        assert len(paths) >= 1
        assert paths[0] == [e1.id, e2.id, e3.id]

    def test_find_paths_no_path(self):
        """Test finding paths when none exist."""
        graph = KnowledgeGraph()

        e1 = graph.add_entity("A", EntityType.CONCEPT)
        e2 = graph.add_entity("B", EntityType.CONCEPT)

        # No edge between them
        paths = graph.find_paths(e1.id, e2.id)
        assert len(paths) == 0

    def test_get_most_mentioned(self):
        """Test getting most mentioned entities."""
        graph = KnowledgeGraph()

        e1 = graph.add_entity("Popular", EntityType.TOPIC)
        e1.mention_count = 10

        e2 = graph.add_entity("Medium", EntityType.TOPIC)
        e2.mention_count = 5

        e3 = graph.add_entity("Rare", EntityType.TOPIC)
        e3.mention_count = 1

        most = graph.get_most_mentioned(limit=2)
        assert len(most) == 2
        assert most[0].name == "Popular"
        assert most[1].name == "Medium"


class TestKnowledgeGraphPersistence:
    """Tests for knowledge graph serialization."""

    def test_to_dict_from_dict(self):
        """Test full graph serialization round-trip."""
        graph = KnowledgeGraph()

        e1 = graph.add_entity("ChatEngine", EntityType.CLASS, message_id="msg_0")
        e2 = graph.add_entity("Session", EntityType.CLASS, message_id="msg_0")
        graph.add_edge(e1.id, e2.id, RelationType.DEPENDS_ON, message_id="msg_0")

        data = graph.to_dict()
        restored = KnowledgeGraph.from_dict(data)

        assert len(restored.entities) == 2
        assert len(restored.edges) == 1
        assert "msg_0" in restored.message_entities

    def test_merge_graphs(self):
        """Test merging two knowledge graphs."""
        graph1 = KnowledgeGraph()
        graph1.add_entity("ChatEngine", EntityType.CLASS)
        graph1.add_entity("Session", EntityType.CLASS)

        graph2 = KnowledgeGraph()
        graph2.add_entity("ChatEngine", EntityType.CLASS)  # Duplicate
        graph2.add_entity("numpy", EntityType.LIBRARY)  # New

        graph1.merge(graph2)

        assert len(graph1.entities) == 3
        # ChatEngine mention count should be 2
        engine = graph1.get_entity("ChatEngine")
        assert engine.mention_count == 2


class TestEntityExtractor:
    """Tests for EntityExtractor class."""

    def test_pattern_extraction_files(self):
        """Test extracting file paths."""
        extractor = EntityExtractor()

        content = "Check the chat.py file and settings.yaml for config."
        result = extractor.extract(content)

        entity_names = [e["name"] for e in result["entities"]]
        assert "chat.py" in entity_names
        assert "settings.yaml" in entity_names

    def test_pattern_extraction_functions(self):
        """Test extracting function names."""
        extractor = EntityExtractor()

        content = "The get_weather() function calls process_data() internally."
        result = extractor.extract(content)

        entity_names = [e["name"] for e in result["entities"]]
        assert "get_weather" in entity_names
        assert "process_data" in entity_names

    def test_pattern_extraction_classes(self):
        """Test extracting class names."""
        extractor = EntityExtractor()

        content = "The ChatEngine class uses SessionManager for persistence."
        result = extractor.extract(content)

        entity_names = [e["name"] for e in result["entities"]]
        assert "ChatEngine" in entity_names
        assert "SessionManager" in entity_names

    def test_pattern_extraction_imports(self):
        """Test extracting import statements."""
        extractor = EntityExtractor()

        content = "import numpy\nfrom pandas import DataFrame"
        result = extractor.extract(content)

        entity_names = [e["name"] for e in result["entities"]]
        assert "numpy" in entity_names
        assert "pandas" in entity_names

    def test_co_occurrence_relationships(self):
        """Test that co-occurrence relationships are built."""
        extractor = EntityExtractor()

        content = "Check chat.py and config.json files."
        result = extractor.extract(content)

        # Should have co-occurrence between the two files
        assert len(result["relationships"]) > 0
        rel_types = [r["type"] for r in result["relationships"]]
        assert "co_occurs" in rel_types

    def test_empty_content(self):
        """Test extraction from empty content."""
        extractor = EntityExtractor()

        result = extractor.extract("")

        assert result["entities"] == []
        assert result["relationships"] == []

    def test_batch_extraction(self):
        """Test batch extraction."""
        extractor = EntityExtractor()

        messages = [
            {"role": "user", "content": "Check chat.py"},
            {"role": "assistant", "content": "Looking at ChatEngine class"},
            {"role": "user", "content": "What about numpy?"},
        ]

        results = extractor.extract_batch(messages)

        assert len(results) == 3
        # First message should have chat.py
        assert any(e["name"] == "chat.py" for e in results[0]["entities"])


class TestBuildKnowledgeGraphFromTranscript:
    """Tests for build_knowledge_graph_from_transcript function."""

    def test_build_from_transcript(self):
        """Test building graph from transcript."""
        from l3m_backend.engine.extraction import build_knowledge_graph_from_transcript

        transcript = [
            {"role": "user", "content": "Tell me about the ChatEngine class"},
            {"role": "assistant", "content": "ChatEngine manages the LLM interaction"},
            {"role": "user", "content": "Does it use numpy?"},
        ]

        graph = build_knowledge_graph_from_transcript(transcript)

        assert len(graph.entities) > 0
        assert graph.stats["num_messages"] > 0

    def test_build_with_progress(self):
        """Test building graph with progress callback."""
        from l3m_backend.engine.extraction import build_knowledge_graph_from_transcript

        progress_calls = []

        def on_progress(current, total):
            progress_calls.append((current, total))

        transcript = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ]

        build_knowledge_graph_from_transcript(
            transcript, progress_callback=on_progress
        )

        assert len(progress_calls) == 2
        assert progress_calls[-1] == (2, 2)


class TestEntityEmbeddings:
    """Tests for entity embedding functionality."""

    def test_get_embedding_text_with_context(self):
        """Test embedding text format with context."""
        entity = Entity(
            id="class_chatengine",
            name="ChatEngine",
            entity_type=EntityType.CLASS,
            context="Main chat engine class",
        )
        text = entity.get_embedding_text()
        assert text == "class ChatEngine: Main chat engine class"

    def test_get_embedding_text_without_context(self):
        """Test embedding text format without context."""
        entity = Entity(
            id="function_foo",
            name="foo",
            entity_type=EntityType.FUNCTION,
        )
        text = entity.get_embedding_text()
        assert text == "function foo"

    def test_entity_embedding_serialization(self):
        """Test embedding is serialized correctly."""
        import numpy as np

        entity = Entity(
            id="class_test",
            name="Test",
            entity_type=EntityType.CLASS,
            embedding=np.array([0.1, 0.2, 0.3]),
        )
        data = entity.to_dict()
        assert "embedding" in data
        assert data["embedding"] == [0.1, 0.2, 0.3]

        restored = Entity.from_dict(data)
        assert restored.embedding is not None
        assert np.allclose(restored.embedding, entity.embedding)

    def test_has_embeddings_empty_graph(self):
        """Test has_embeddings on empty graph."""
        graph = KnowledgeGraph()
        assert not graph.has_embeddings

    def test_has_embeddings_no_embeddings(self):
        """Test has_embeddings when entities have no embeddings."""
        graph = KnowledgeGraph()
        graph.add_entity("Test", EntityType.CLASS)
        assert not graph.has_embeddings

    def test_has_embeddings_with_embeddings(self):
        """Test has_embeddings when entities have embeddings."""
        import numpy as np

        graph = KnowledgeGraph()
        entity = graph.add_entity("Test", EntityType.CLASS)
        entity.embedding = np.array([0.1, 0.2, 0.3])
        assert graph.has_embeddings

    def test_compute_embeddings(self):
        """Test compute_embeddings with mock provider."""
        import numpy as np

        # Mock embedding provider
        class MockProvider:
            def embed(self, texts: list[str]) -> np.ndarray:
                # Return 4-dim embeddings (one per text)
                return np.array([[0.1, 0.2, 0.3, 0.4]] * len(texts))

        graph = KnowledgeGraph()
        graph.add_entity("ChatEngine", EntityType.CLASS, context="Main engine")
        graph.add_entity("get_weather", EntityType.FUNCTION, context="Weather func")

        count = graph.compute_embeddings(MockProvider())
        assert count == 2
        assert graph.has_embeddings

        # Check embeddings were assigned
        entity = graph.get_entity("ChatEngine")
        assert entity.embedding is not None
        assert len(entity.embedding) == 4

    def test_get_similar_entities(self):
        """Test finding similar entities by embedding."""
        import numpy as np

        graph = KnowledgeGraph()
        e1 = graph.add_entity("EntityA", EntityType.CLASS)
        e2 = graph.add_entity("EntityB", EntityType.CLASS)
        e3 = graph.add_entity("EntityC", EntityType.CLASS)

        # Assign embeddings: A and B are similar, C is different
        e1.embedding = np.array([1.0, 0.0, 0.0])
        e2.embedding = np.array([0.9, 0.1, 0.0])
        e3.embedding = np.array([0.0, 0.0, 1.0])

        # Query with A's embedding
        similar = graph.get_similar_entities(e1.embedding, k=2)
        assert len(similar) == 2

        # B should be most similar to A (excluding A itself when using find_similar_to_entity)
        names = [ent.name for ent, _ in similar]
        assert "EntityA" in names  # A is most similar to itself
        assert "EntityB" in names  # B is similar

    def test_find_similar_to_entity(self):
        """Test finding entities similar to a given entity."""
        import numpy as np

        graph = KnowledgeGraph()
        e1 = graph.add_entity("EntityA", EntityType.CLASS)
        e2 = graph.add_entity("EntityB", EntityType.CLASS)
        e3 = graph.add_entity("EntityC", EntityType.CLASS)

        # Assign embeddings
        e1.embedding = np.array([1.0, 0.0, 0.0])
        e2.embedding = np.array([0.9, 0.1, 0.0])
        e3.embedding = np.array([0.0, 0.0, 1.0])

        # Find entities similar to A
        similar = graph.find_similar_to_entity(e1.id, k=2)
        assert len(similar) == 2

        # A should be excluded, B should be first (most similar)
        names = [ent.name for ent, _ in similar]
        assert "EntityA" not in names
        assert "EntityB" in names

    def test_get_similar_entities_no_embeddings(self):
        """Test get_similar_entities returns empty when no embeddings."""
        import numpy as np

        graph = KnowledgeGraph()
        graph.add_entity("Test", EntityType.CLASS)

        result = graph.get_similar_entities(np.array([1.0, 0.0]), k=5)
        assert result == []


class TestBuildKnowledgeGraphWithEmbeddings:
    """Tests for building knowledge graph with embeddings."""

    def test_build_with_embedding_provider(self):
        """Test building graph with embedding provider."""
        import numpy as np

        from l3m_backend.engine.extraction import build_knowledge_graph_from_transcript

        class MockProvider:
            def embed(self, texts: list[str]) -> np.ndarray:
                return np.array([[0.1, 0.2, 0.3, 0.4]] * len(texts))

        transcript = [
            {"role": "user", "content": "Tell me about ChatEngine class"},
            {"role": "assistant", "content": "ChatEngine manages LLM interaction"},
        ]

        graph = build_knowledge_graph_from_transcript(
            transcript, embedding_provider=MockProvider()
        )

        assert len(graph.entities) > 0
        assert graph.has_embeddings

    def test_build_without_embedding_provider(self):
        """Test building graph without embedding provider."""
        from l3m_backend.engine.extraction import build_knowledge_graph_from_transcript

        transcript = [
            {"role": "user", "content": "Tell me about ChatEngine class"},
        ]

        graph = build_knowledge_graph_from_transcript(transcript)

        assert len(graph.entities) > 0
        assert not graph.has_embeddings
