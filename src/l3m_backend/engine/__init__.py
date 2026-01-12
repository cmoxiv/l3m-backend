"""
Engine module for the l3m_backend package.

Provides the ChatEngine for LLM interaction with tool calling.
"""

from l3m_backend.engine.chat import ChatEngine
from l3m_backend.engine.contract import TOOL_CONTRACT_TEMPLATE, load_contract_template
from l3m_backend.engine.legacy_priming import LegacyPrimingGenerator, generate_legacy_priming
from l3m_backend.engine.extraction import (
    EntityExtractor,
    build_knowledge_graph_from_transcript,
)
from l3m_backend.engine.knowledge import (
    Entity,
    EntityType,
    KnowledgeEdge,
    KnowledgeGraph,
    RelationType,
)
from l3m_backend.engine.similarity import (
    LlamaEmbeddingProvider,
    SimilarityGraph,
    SimilarityNode,
)

__all__ = [
    "ChatEngine",
    "TOOL_CONTRACT_TEMPLATE",
    "load_contract_template",
    # Legacy Priming
    "LegacyPrimingGenerator",
    "generate_legacy_priming",
    # Similarity graph
    "SimilarityGraph",
    "SimilarityNode",
    "LlamaEmbeddingProvider",
    # Knowledge graph
    "KnowledgeGraph",
    "Entity",
    "EntityType",
    "KnowledgeEdge",
    "RelationType",
    "EntityExtractor",
    "build_knowledge_graph_from_transcript",
]
