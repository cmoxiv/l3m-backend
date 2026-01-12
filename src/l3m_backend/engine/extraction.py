"""Entity extraction module for knowledge graph construction.

Provides LLM-based entity extraction from conversation messages.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from llama_cpp import Llama

from l3m_backend.engine.knowledge import KnowledgeGraph


# Prompt template for entity extraction
EXTRACTION_PROMPT = """Analyze this message and extract entities.

Message: {content}

Extract:
1. Named entities (people, places, organizations)
2. Code concepts (functions, classes, files, libraries)
3. Topics/themes being discussed

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "entity_name", "type": "person|org|place|datetime|function|class|file|library|variable|topic|theme|concept", "context": "brief context"}}
  ],
  "relationships": [
    {{"from": "entity1_name", "to": "entity2_name", "type": "co_occurs|depends_on|implements|mentions"}}
  ]
}}

JSON response:"""


class EntityExtractor:
    """Extracts entities from text using LLM or pattern matching."""

    def __init__(
        self,
        llm: "Llama | None" = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ):
        """Initialize the extractor.

        Args:
            llm: Llama instance for LLM-based extraction
            max_tokens: Maximum tokens for extraction response
            temperature: Temperature for LLM generation
        """
        self.llm = llm
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Pattern-based extractors as fallback
        self._code_patterns = {
            "file": re.compile(r"[\w/-]+\.(py|js|ts|go|rs|java|c|cpp|h|hpp|md|txt|json|yaml|yml|toml)"),
            "function": re.compile(r"\b([a-z_][a-z0-9_]*)\s*\("),
            # Class: CamelCase with 2+ capitals, or explicit "class Foo" pattern
            "class_explicit": re.compile(r"\bclass\s+([A-Z][a-zA-Z0-9]*)\b"),
            "class_camel": re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b"),  # CamelCase only
            "library": re.compile(r"(?:import|from|require)\s+([a-zA-Z_][a-zA-Z0-9_]*)"),
        }

        # Common English words to filter out (not code entities)
        self._common_words = {
            # Pronouns and articles
            "the", "this", "that", "it", "i", "a", "an", "he", "she", "they",
            "we", "you", "his", "her", "their", "my", "your", "its",
            # Conjunctions and prepositions
            "and", "or", "but", "if", "for", "in", "on", "at", "to", "of",
            "with", "by", "from", "as", "into", "through", "during", "before",
            "after", "above", "below", "between", "under", "again", "further",
            # Common verbs
            "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "can", "shall",
            # Question words
            "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
            # Common adjectives/adverbs
            "here", "there", "then", "now", "just", "only", "also", "very",
            "more", "most", "other", "some", "any", "each", "every", "all",
            "both", "few", "many", "much", "such", "no", "nor", "not", "own",
            "same", "so", "than", "too", "very", "just", "even", "also", "back",
            # Time words
            "today", "yesterday", "tomorrow", "week", "weeks", "month", "year",
            "day", "days", "time", "times", "morning", "evening", "night",
            # Common nouns
            "way", "thing", "things", "people", "person", "man", "woman",
            "child", "world", "life", "hand", "part", "place", "case", "point",
            "government", "company", "number", "group", "problem", "fact",
            # Conversational
            "yes", "no", "yeah", "okay", "ok", "please", "thanks", "thank",
            "sorry", "hello", "hi", "hey", "bye", "well", "oh", "ah",
            # Programming keywords (not class names)
            "true", "false", "none", "null", "undefined", "return", "break",
            "continue", "pass", "raise", "try", "except", "finally", "else",
            "elif", "while", "async", "await", "yield", "lambda", "def", "class",
        }

    def extract(self, content: str, message_id: str | None = None) -> dict[str, Any]:
        """Extract entities from message content.

        Uses LLM if available, falls back to pattern matching.

        Args:
            content: The message content to extract from
            message_id: Optional message ID for tracking

        Returns:
            Dictionary with 'entities' and 'relationships' lists
        """
        if self.llm:
            return self._extract_with_llm(content, message_id)
        else:
            return self._extract_with_patterns(content, message_id)

    def _extract_with_llm(
        self, content: str, message_id: str | None = None
    ) -> dict[str, Any]:
        """Extract entities using the LLM."""
        if not self.llm:
            return self._extract_with_patterns(content, message_id)

        prompt = EXTRACTION_PROMPT.format(content=content[:2000])  # Limit content

        try:
            response = self.llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["```", "\n\n\n"],
            )

            # Handle both streaming and non-streaming responses
            if hasattr(response, "__getitem__"):
                text = response["choices"][0]["text"].strip()
            else:
                text = ""

            # Try to parse JSON from response
            result = self._parse_extraction_response(text)
            return result

        except Exception:
            # Fall back to pattern matching on error
            return self._extract_with_patterns(content, message_id)

    def _parse_extraction_response(self, text: str) -> dict[str, Any]:
        """Parse LLM response into structured extraction result."""
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the text
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Return empty result if parsing fails
        return {"entities": [], "relationships": []}

    def _extract_with_patterns(
        self, content: str, _message_id: str | None = None
    ) -> dict[str, Any]:
        """Extract entities using regex patterns (fallback method)."""
        entities: list[dict[str, str]] = []
        seen_names: set[str] = set()

        # Extract file paths
        for match in self._code_patterns["file"].finditer(content):
            name = match.group(0)
            if name.lower() not in seen_names:
                entities.append({
                    "name": name,
                    "type": "file",
                    "context": "File reference in message",
                })
                seen_names.add(name.lower())

        # Extract function calls (lowercase with parentheses)
        for match in self._code_patterns["function"].finditer(content):
            name = match.group(1)
            # Filter out common words and builtins
            if name.lower() not in self._common_words and name not in (
                "print", "len", "str", "int", "list", "dict", "set", "type",
                "range", "open", "input", "format", "sorted", "map", "filter",
            ):
                if name.lower() not in seen_names and len(name) > 2:
                    entities.append({
                        "name": name,
                        "type": "function",
                        "context": "Function reference",
                    })
                    seen_names.add(name.lower())

        # Extract explicit class definitions (class ClassName)
        for match in self._code_patterns["class_explicit"].finditer(content):
            name = match.group(1)
            if name.lower() not in seen_names and name.lower() not in self._common_words:
                entities.append({
                    "name": name,
                    "type": "class",
                    "context": "Class definition",
                })
                seen_names.add(name.lower())

        # Extract CamelCase class names (must have 2+ capital letters)
        for match in self._code_patterns["class_camel"].finditer(content):
            name = match.group(1)
            if name.lower() not in seen_names and name.lower() not in self._common_words:
                entities.append({
                    "name": name,
                    "type": "class",
                    "context": "Class reference",
                })
                seen_names.add(name.lower())

        # Extract library imports
        for match in self._code_patterns["library"].finditer(content):
            name = match.group(1)
            if name.lower() not in seen_names:
                entities.append({
                    "name": name,
                    "type": "library",
                    "context": "Library import",
                })
                seen_names.add(name.lower())

        # Build co-occurrence relationships for entities in same message
        relationships: list[dict[str, str]] = []
        entity_names = [e["name"] for e in entities]
        for i, name1 in enumerate(entity_names):
            for name2 in entity_names[i + 1:]:
                relationships.append({
                    "from": name1,
                    "to": name2,
                    "type": "co_occurs",
                })

        return {"entities": entities, "relationships": relationships}

    def extract_batch(
        self,
        messages: list[dict[str, Any]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract entities from a batch of messages.

        Args:
            messages: List of message dicts with 'content' key
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of extraction results, one per message
        """
        results: list[dict[str, Any]] = []
        total = len(messages)

        for i, message in enumerate(messages):
            content = message.get("content", "")
            msg_id = f"msg_{i}"

            if content:
                result = self.extract(content, msg_id)
            else:
                result = {"entities": [], "relationships": []}

            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results


def build_knowledge_graph_from_transcript(
    transcript: list[dict[str, Any]],
    llm: "Llama | None" = None,
    progress_callback: Callable[[int, int], None] | None = None,
    embedding_provider: Any = None,
) -> KnowledgeGraph:
    """Build a knowledge graph from a conversation transcript.

    Args:
        transcript: List of message dicts with 'role' and 'content'
        llm: Optional Llama instance for LLM-based extraction
        progress_callback: Optional progress callback
        embedding_provider: Optional provider implementing EmbeddingProvider protocol
                           (must have embed(texts: list[str]) -> np.ndarray method)

    Returns:
        Populated KnowledgeGraph
    """
    graph = KnowledgeGraph()
    extractor = EntityExtractor(llm=llm)

    for i, message in enumerate(transcript):
        content = message.get("content", "")
        msg_id = f"msg_{i}"

        if content:
            extraction = extractor.extract(content, msg_id)
            graph.add_entities_from_extraction(extraction, msg_id)

        if progress_callback:
            progress_callback(i + 1, len(transcript))

    # Build co-occurrence edges
    graph.build_co_occurrence_edges()

    # Compute entity embeddings if provider is available
    if embedding_provider is not None and graph.entities:
        graph.compute_embeddings(embedding_provider)

    return graph
