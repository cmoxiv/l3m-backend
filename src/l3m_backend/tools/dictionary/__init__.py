"""Dictionary/word definition tool."""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pydantic import Field

from l3m_backend.core import tool_output
from l3m_backend.tools._registry import registry


@registry.register(aliases=["define", "dict"])
@tool_output(llm_format=lambda x: f"{x['word']}: {x['definition']}" if x.get('definition') else x.get('error', 'No definition'))
def define_word(
    word: str = Field(description="Word to define"),
) -> dict[str, Any]:
    """Look up the definition of a word.

    Uses the Free Dictionary API.

    Args:
        word: The word to define.

    Returns:
        Dictionary with keys:
            - word: The word
            - definition: Primary definition
            - part_of_speech: e.g., "noun", "verb"
            - example: Example usage (if available)
        Or on error:
            - error: Error message
    """
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(word)}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())

        if data and isinstance(data, list):
            entry = data[0]
            meanings = entry.get("meanings", [])
            if meanings:
                meaning = meanings[0]
                definitions = meaning.get("definitions", [])
                if definitions:
                    defn = definitions[0]
                    return {
                        "word": entry.get("word", word),
                        "part_of_speech": meaning.get("partOfSpeech", "unknown"),
                        "definition": defn.get("definition", "No definition"),
                        "example": defn.get("example", ""),
                    }
        return {"error": f"No definition found for '{word}'"}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": f"Word '{word}' not found in dictionary"}
        return {"error": f"HTTP error: {e.code}"}
    except Exception as e:
        return {"error": str(e)}
