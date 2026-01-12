"""Knowledge graph module for tracking entities and relationships across conversations.

This module extends the similarity graph with structured knowledge about what is
discussed in conversations - entities (people, code, topics) and their relationships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from l3m_backend.engine.similarity import SimilarityGraph


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        ...


class EntityType(str, Enum):
    """Types of entities that can be extracted from conversations."""

    # Named entities
    PERSON = "person"
    ORGANIZATION = "org"
    LOCATION = "place"
    DATETIME = "datetime"

    # Code concepts
    FUNCTION = "function"
    CLASS = "class"
    FILE = "file"
    LIBRARY = "library"
    VARIABLE = "variable"

    # Topics and themes
    TOPIC = "topic"
    THEME = "theme"
    CONCEPT = "concept"


class RelationType(str, Enum):
    """Types of relationships between entities."""

    CO_OCCURS = "co_occurs"  # Entities in same message
    SIMILAR_TO = "similar_to"  # Semantic similarity
    FOLLOWS = "follows"  # Temporal ordering
    DEPENDS_ON = "depends_on"  # Causal/dependency
    IMPLEMENTS = "implements"  # Code relationship
    MENTIONS = "mentions"  # Entity reference


@dataclass
class Entity:
    """An entity extracted from conversation."""

    id: str
    name: str
    entity_type: EntityType
    context: str = ""  # Brief context about the entity
    first_seen_msg: str | None = None  # Message ID where first mentioned
    mention_count: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    embedding: np.ndarray | None = None  # Contextual embedding vector

    def get_embedding_text(self) -> str:
        """Get text to embed for this entity (contextual format)."""
        if self.context:
            return f"{self.entity_type.value} {self.name}: {self.context}"
        return f"{self.entity_type.value} {self.name}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize entity to dictionary."""
        data = {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "context": self.context,
            "first_seen_msg": self.first_seen_msg,
            "mention_count": self.mention_count,
            "created_at": self.created_at.isoformat(),
        }
        if self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        """Deserialize entity from dictionary."""
        embedding = None
        if "embedding" in data and data["embedding"] is not None:
            embedding = np.array(data["embedding"])
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            context=data.get("context", ""),
            first_seen_msg=data.get("first_seen_msg"),
            mention_count=data.get("mention_count", 1),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            embedding=embedding,
        )


@dataclass
class KnowledgeEdge:
    """A relationship between two entities."""

    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0  # 0.0 - 1.0
    message_ids: list[str] = field(default_factory=list)  # Where observed
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize edge to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "message_ids": self.message_ids,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeEdge:
        """Deserialize edge from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            weight=data.get("weight", 1.0),
            message_ids=data.get("message_ids", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
        )


@dataclass
class KnowledgeGraph:
    """Knowledge graph for tracking entities and relationships in conversations.

    Extends SimilarityGraph with structured knowledge about what is discussed.
    Tracks entities (people, code concepts, topics) and their relationships.
    """

    entities: dict[str, Entity] = field(default_factory=dict)  # id -> entity
    edges: list[KnowledgeEdge] = field(default_factory=list)
    message_entities: dict[str, list[str]] = field(
        default_factory=dict
    )  # msg_id -> entity_ids
    _entity_name_index: dict[str, str] = field(
        default_factory=dict
    )  # name.lower() -> id

    # Optional reference to similarity graph for semantic similarity edges
    similarity_graph: "SimilarityGraph | None" = None

    def __post_init__(self) -> None:
        """Build indices after initialization."""
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """Rebuild internal indices."""
        self._entity_name_index = {
            e.name.lower(): e.id for e in self.entities.values()
        }

    def _generate_entity_id(self, name: str, entity_type: EntityType) -> str:
        """Generate a unique entity ID."""
        base = f"{entity_type.value}_{name.lower().replace(' ', '_')}"
        if base not in self.entities:
            return base
        # Handle collisions
        counter = 1
        while f"{base}_{counter}" in self.entities:
            counter += 1
        return f"{base}_{counter}"

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        context: str = "",
        message_id: str | None = None,
    ) -> Entity:
        """Add or update an entity in the graph.

        If entity with same name exists, increments mention count.
        Returns the entity (new or existing).
        """
        # Check if entity already exists by name
        existing_id = self._entity_name_index.get(name.lower())
        if existing_id and existing_id in self.entities:
            entity = self.entities[existing_id]
            entity.mention_count += 1
            if context and not entity.context:
                entity.context = context
            return entity

        # Create new entity
        entity_id = self._generate_entity_id(name, entity_type)
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            context=context,
            first_seen_msg=message_id,
        )
        self.entities[entity_id] = entity
        self._entity_name_index[name.lower()] = entity_id

        # Track which message this entity appeared in
        if message_id:
            if message_id not in self.message_entities:
                self.message_entities[message_id] = []
            self.message_entities[message_id].append(entity_id)

        return entity

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
        message_id: str | None = None,
    ) -> KnowledgeEdge | None:
        """Add or update an edge between entities.

        If edge exists, updates weight and adds message_id.
        Returns the edge or None if entities don't exist.
        """
        if source_id not in self.entities or target_id not in self.entities:
            return None

        # Check for existing edge
        for edge in self.edges:
            if (
                edge.source_id == source_id
                and edge.target_id == target_id
                and edge.relation_type == relation_type
            ):
                # Update existing edge
                edge.weight = max(edge.weight, weight)
                if message_id and message_id not in edge.message_ids:
                    edge.message_ids.append(message_id)
                return edge

        # Create new edge
        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            message_ids=[message_id] if message_id else [],
        )
        self.edges.append(edge)
        return edge

    def add_entities_from_extraction(
        self,
        extraction_result: dict[str, Any],
        message_id: str | None = None,
    ) -> list[Entity]:
        """Add entities and relationships from an extraction result.

        Expected format:
        {
            "entities": [
                {"name": "...", "type": "person|function|...", "context": "..."}
            ],
            "relationships": [
                {"from": "...", "to": "...", "type": "co_occurs|..."}
            ]
        }
        """
        added_entities: list[Entity] = []
        entity_name_to_id: dict[str, str] = {}

        # Add entities
        for ent_data in extraction_result.get("entities", []):
            try:
                entity_type = EntityType(ent_data.get("type", "topic"))
            except ValueError:
                entity_type = EntityType.TOPIC

            entity = self.add_entity(
                name=ent_data.get("name", "unknown"),
                entity_type=entity_type,
                context=ent_data.get("context", ""),
                message_id=message_id,
            )
            added_entities.append(entity)
            entity_name_to_id[ent_data.get("name", "").lower()] = entity.id

        # Add relationships
        for rel_data in extraction_result.get("relationships", []):
            from_name = rel_data.get("from", "").lower()
            to_name = rel_data.get("to", "").lower()

            from_id = entity_name_to_id.get(
                from_name, self._entity_name_index.get(from_name)
            )
            to_id = entity_name_to_id.get(
                to_name, self._entity_name_index.get(to_name)
            )

            if from_id and to_id:
                try:
                    relation_type = RelationType(
                        rel_data.get("type", "co_occurs")
                    )
                except ValueError:
                    relation_type = RelationType.CO_OCCURS

                self.add_edge(
                    source_id=from_id,
                    target_id=to_id,
                    relation_type=relation_type,
                    message_id=message_id,
                )

        return added_entities

    def build_co_occurrence_edges(self) -> int:
        """Build CO_OCCURS edges for entities in the same message.

        Returns number of edges created.
        """
        edges_created = 0
        for msg_id, entity_ids in self.message_entities.items():
            # Create edges between all pairs of entities in this message
            for i, source_id in enumerate(entity_ids):
                for target_id in entity_ids[i + 1 :]:
                    edge = self.add_edge(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=RelationType.CO_OCCURS,
                        message_id=msg_id,
                    )
                    if edge:
                        edges_created += 1
        return edges_created

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def get_entity(self, name: str) -> Entity | None:
        """Get entity by name (case-insensitive)."""
        entity_id = self._entity_name_index.get(name.lower())
        return self.entities.get(entity_id) if entity_id else None

    def get_entity_by_id(self, entity_id: str) -> Entity | None:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def get_entities_by_type(
        self, entity_type: EntityType
    ) -> list[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_related(
        self,
        entity_id: str,
        relation_type: RelationType | None = None,
    ) -> list[tuple[Entity, KnowledgeEdge]]:
        """Get entities related to the given entity.

        Args:
            entity_id: The entity to find relations for
            relation_type: Optional filter by relation type

        Returns:
            List of (entity, edge) tuples for related entities
        """
        if entity_id not in self.entities:
            return []

        related: list[tuple[Entity, KnowledgeEdge]] = []
        for edge in self.edges:
            # Check if this entity is involved in the edge
            other_id = None
            if edge.source_id == entity_id:
                other_id = edge.target_id
            elif edge.target_id == entity_id:
                other_id = edge.source_id

            if other_id is None:
                continue

            # Apply relation type filter
            if relation_type and edge.relation_type != relation_type:
                continue

            other_entity = self.entities.get(other_id)
            if other_entity:
                related.append((other_entity, edge))

        # Sort by edge weight descending
        related.sort(key=lambda x: x[1].weight, reverse=True)
        return related

    def get_messages_about(self, entity_id: str) -> list[str]:
        """Get all message IDs that mention an entity."""
        if entity_id not in self.entities:
            return []

        message_ids = set()

        # Direct mentions
        for msg_id, ent_ids in self.message_entities.items():
            if entity_id in ent_ids:
                message_ids.add(msg_id)

        # Also check edges for additional message references
        for edge in self.edges:
            if edge.source_id == entity_id or edge.target_id == entity_id:
                message_ids.update(edge.message_ids)

        return list(message_ids)

    def get_entities_in_message(self, msg_id: str) -> list[Entity]:
        """Get all entities mentioned in a specific message."""
        entity_ids = self.message_entities.get(msg_id, [])
        return [
            self.entities[eid] for eid in entity_ids if eid in self.entities
        ]

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    def get_topic_summary(self) -> dict[str, list[Entity]]:
        """Get entities grouped by topic/theme.

        Returns dict mapping topic names to related entities.
        """
        # Find all topic/theme entities
        topics: dict[str, list[Entity]] = {}

        for entity in self.entities.values():
            if entity.entity_type in (
                EntityType.TOPIC,
                EntityType.THEME,
                EntityType.CONCEPT,
            ):
                topic_name = entity.name
                topics[topic_name] = [entity]

                # Find entities related to this topic
                related = self.get_related(entity.id)
                for rel_entity, _ in related:
                    if rel_entity.id != entity.id:
                        topics[topic_name].append(rel_entity)

        return topics

    def get_entity_timeline(
        self, entity_id: str
    ) -> list[tuple[str, datetime]]:
        """Get timeline of when entity was mentioned.

        Returns list of (message_id, timestamp) tuples.
        """
        if entity_id not in self.entities:
            return []

        entity = self.entities[entity_id]
        messages = self.get_messages_about(entity_id)

        # We don't have message timestamps, so use edge timestamps as proxy
        timeline: list[tuple[str, datetime]] = []
        for msg_id in messages:
            # Use entity creation time as approximation
            timeline.append((msg_id, entity.created_at))

        return timeline

    def find_paths(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 3,
    ) -> list[list[str]]:
        """Find paths between two entities in the graph.

        Uses BFS to find shortest paths up to max_depth.

        Returns:
            List of paths, where each path is a list of entity IDs
        """
        if from_id not in self.entities or to_id not in self.entities:
            return []

        if from_id == to_id:
            return [[from_id]]

        # Build adjacency list
        adjacency: dict[str, set[str]] = {eid: set() for eid in self.entities}
        for edge in self.edges:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)

        # BFS
        paths: list[list[str]] = []
        queue: list[tuple[str, list[str]]] = [(from_id, [from_id])]
        visited: set[str] = {from_id}

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            for neighbor in adjacency[current]:
                if neighbor == to_id:
                    paths.append(path + [neighbor])
                elif neighbor not in visited and len(path) < max_depth:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return paths

    def get_most_mentioned(self, limit: int = 10) -> list[Entity]:
        """Get the most frequently mentioned entities."""
        entities = list(self.entities.values())
        entities.sort(key=lambda e: e.mention_count, reverse=True)
        return entities[:limit]

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------

    @property
    def has_embeddings(self) -> bool:
        """Check if entities have embeddings computed."""
        if not self.entities:
            return False
        # Check if at least some entities have embeddings
        return any(e.embedding is not None for e in self.entities.values())

    def compute_embeddings(self, provider: EmbeddingProvider) -> int:
        """Compute contextual embeddings for all entities.

        Args:
            provider: An object implementing the EmbeddingProvider protocol
                     (must have embed(texts: list[str]) -> np.ndarray method)

        Returns:
            Number of entities that received embeddings
        """
        if not self.entities:
            return 0

        # Build texts for all entities
        entity_ids = list(self.entities.keys())
        texts = [self.entities[eid].get_embedding_text() for eid in entity_ids]

        # Compute embeddings in batch
        embeddings = provider.embed(texts)

        # Assign embeddings to entities
        count = 0
        for i, eid in enumerate(entity_ids):
            if i < len(embeddings):
                self.entities[eid].embedding = embeddings[i]
                count += 1

        return count

    def get_similar_entities(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[Entity, float]]:
        """Find entities most similar to a query embedding.

        Uses cosine similarity to find semantically similar entities.

        Args:
            query_embedding: The embedding vector to compare against
            k: Number of similar entities to return
            exclude_ids: Entity IDs to exclude from results

        Returns:
            List of (entity, similarity_score) tuples, sorted by score descending
        """
        if not self.has_embeddings:
            return []

        exclude_ids = exclude_ids or set()

        # Compute cosine similarities
        similarities: list[tuple[Entity, float]] = []

        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_normalized = query_embedding / query_norm

        for entity in self.entities.values():
            if entity.id in exclude_ids:
                continue
            if entity.embedding is None:
                continue

            # Cosine similarity
            entity_norm = np.linalg.norm(entity.embedding)
            if entity_norm == 0:
                continue

            similarity = float(
                np.dot(query_normalized, entity.embedding / entity_norm)
            )
            similarities.append((entity, similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]

    def find_similar_to_entity(
        self,
        entity_id: str,
        k: int = 5,
    ) -> list[tuple[Entity, float]]:
        """Find entities similar to a given entity by embedding.

        Args:
            entity_id: ID of the entity to find similar ones for
            k: Number of similar entities to return

        Returns:
            List of (entity, similarity_score) tuples
        """
        entity = self.entities.get(entity_id)
        if not entity or entity.embedding is None:
            return []

        return self.get_similar_entities(
            entity.embedding, k=k, exclude_ids={entity_id}
        )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        entity_type_counts = {}
        for entity in self.entities.values():
            entity_type_counts[entity.entity_type.value] = (
                entity_type_counts.get(entity.entity_type.value, 0) + 1
            )

        relation_type_counts = {}
        for edge in self.edges:
            relation_type_counts[edge.relation_type.value] = (
                relation_type_counts.get(edge.relation_type.value, 0) + 1
            )

        return {
            "num_entities": len(self.entities),
            "num_edges": len(self.edges),
            "num_messages": len(self.message_entities),
            "entity_types": entity_type_counts,
            "relation_types": relation_type_counts,
        }

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "entities": {
                eid: entity.to_dict() for eid, entity in self.entities.items()
            },
            "edges": [edge.to_dict() for edge in self.edges],
            "message_entities": self.message_entities,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        """Deserialize graph from dictionary."""
        entities = {
            eid: Entity.from_dict(edata)
            for eid, edata in data.get("entities", {}).items()
        }
        edges = [
            KnowledgeEdge.from_dict(edata)
            for edata in data.get("edges", [])
        ]
        message_entities = data.get("message_entities", {})

        graph = cls(
            entities=entities,
            edges=edges,
            message_entities=message_entities,
        )
        return graph

    def merge(self, other: KnowledgeGraph) -> None:
        """Merge another knowledge graph into this one.

        Entities with the same name are combined (mention counts added).
        Edges are deduplicated.
        """
        # Merge entities
        for entity in other.entities.values():
            existing_id = self._entity_name_index.get(entity.name.lower())
            if existing_id and existing_id in self.entities:
                # Update existing entity
                self.entities[existing_id].mention_count += entity.mention_count
            else:
                # Add new entity
                self.entities[entity.id] = entity
                self._entity_name_index[entity.name.lower()] = entity.id

        # Merge message_entities
        for msg_id, entity_ids in other.message_entities.items():
            if msg_id not in self.message_entities:
                self.message_entities[msg_id] = []
            for eid in entity_ids:
                if eid not in self.message_entities[msg_id]:
                    self.message_entities[msg_id].append(eid)

        # Merge edges (deduplicate)
        existing_edges = {
            (e.source_id, e.target_id, e.relation_type) for e in self.edges
        }
        for edge in other.edges:
            key = (edge.source_id, edge.target_id, edge.relation_type)
            if key not in existing_edges:
                self.edges.append(edge)
                existing_edges.add(key)
