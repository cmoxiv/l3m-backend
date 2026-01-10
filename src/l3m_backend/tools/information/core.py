"""
Information and lookup tools implementation.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Literal

from pydantic import Field

from l3m_backend.core import tool_output
from l3m_backend.tools._registry import registry


# -----------------------------
# Wikipedia Tool
# -----------------------------

@registry.register(aliases=["wiki"])
@tool_output(llm_format=lambda x: f"{x['title']}: {x['summary']}" if x.get('summary') else x.get('error', 'No result'))
def wikipedia(
    query: str = Field(description="Topic to search on Wikipedia"),
) -> dict[str, Any]:
    """Fetch a Wikipedia summary for a topic.

    Uses the Wikipedia API to get the extract (summary) of an article.

    Args:
        query: The topic to search for (e.g., "Eiffel Tower", "Python programming").

    Returns:
        Dictionary with keys:
            - title: Article title
            - summary: First paragraph summary
            - url: Link to full article
        Or on error:
            - error: Error message
    """
    try:
        encoded = urllib.parse.quote(query)
        url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "LLMTools/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        return {
            "title": data.get("title", query),
            "summary": data.get("extract", "No summary available."),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": f"No Wikipedia article found for '{query}'"}
        return {"error": f"HTTP error: {e.code}"}
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Dictionary Tool
# -----------------------------

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


# -----------------------------
# Unit Conversion Tool
# -----------------------------

_UNIT_CONVERSIONS = {
    # Length
    ("km", "miles"): lambda x: x * 0.621371,
    ("miles", "km"): lambda x: x * 1.60934,
    ("m", "ft"): lambda x: x * 3.28084,
    ("ft", "m"): lambda x: x * 0.3048,
    ("cm", "in"): lambda x: x * 0.393701,
    ("in", "cm"): lambda x: x * 2.54,
    # Weight
    ("kg", "lb"): lambda x: x * 2.20462,
    ("lb", "kg"): lambda x: x * 0.453592,
    ("g", "oz"): lambda x: x * 0.035274,
    ("oz", "g"): lambda x: x * 28.3495,
    # Temperature
    ("c", "f"): lambda x: (x * 9/5) + 32,
    ("f", "c"): lambda x: (x - 32) * 5/9,
    ("c", "k"): lambda x: x + 273.15,
    ("k", "c"): lambda x: x - 273.15,
    # Volume
    ("l", "gal"): lambda x: x * 0.264172,
    ("gal", "l"): lambda x: x * 3.78541,
    ("ml", "floz"): lambda x: x * 0.033814,
    ("floz", "ml"): lambda x: x * 29.5735,
    # Speed
    ("kmh", "mph"): lambda x: x * 0.621371,
    ("mph", "kmh"): lambda x: x * 1.60934,
    # Data
    ("mb", "gb"): lambda x: x / 1024,
    ("gb", "mb"): lambda x: x * 1024,
    ("gb", "tb"): lambda x: x / 1024,
    ("tb", "gb"): lambda x: x * 1024,
}


@registry.register(aliases=["convert", "unit"])
@tool_output(llm_format=lambda x: f"{x['value']} {x['from_unit']} = {x['result']} {x['to_unit']}" if 'result' in x else x.get('error'))
def unit_convert(
    value: float = Field(description="Numeric value to convert"),
    from_unit: str = Field(description="Source unit (e.g., km, lb, c)"),
    to_unit: str = Field(description="Target unit (e.g., miles, kg, f)"),
) -> dict[str, Any]:
    """Convert between units.

    Supported conversions:
    - Length: km/miles, m/ft, cm/in
    - Weight: kg/lb, g/oz
    - Temperature: c/f/k (Celsius/Fahrenheit/Kelvin)
    - Volume: l/gal, ml/floz
    - Speed: kmh/mph
    - Data: mb/gb/tb

    Args:
        value: The numeric value to convert.
        from_unit: Source unit abbreviation.
        to_unit: Target unit abbreviation.

    Returns:
        Dictionary with conversion result or error.
    """
    from_u = from_unit.lower().strip()
    to_u = to_unit.lower().strip()

    converter = _UNIT_CONVERSIONS.get((from_u, to_u))
    if converter:
        result = converter(value)
        return {
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "result": round(result, 4),
        }
    else:
        available = sorted(set(f"{a}->{b}" for a, b in _UNIT_CONVERSIONS.keys()))
        return {
            "error": f"Unknown conversion: {from_unit} -> {to_unit}",
            "available": available,
        }


# -----------------------------
# Currency Conversion Tool
# -----------------------------

@registry.register(aliases=["currency", "fx"])
@tool_output(llm_format=lambda x: f"{x['amount']} {x['from_currency']} = {x['result']} {x['to_currency']}" if 'result' in x else x.get('error'))
def currency_convert(
    amount: float = Field(description="Amount to convert"),
    from_currency: str = Field(description="Source currency code (e.g., USD, EUR)"),
    to_currency: str = Field(description="Target currency code (e.g., GBP, JPY)"),
) -> dict[str, Any]:
    """Convert between currencies using live exchange rates.

    Uses the free exchangerate-api.com API.

    Args:
        amount: The amount to convert.
        from_currency: Source currency code (e.g., USD, EUR, GBP).
        to_currency: Target currency code.

    Returns:
        Dictionary with conversion result or error.
    """
    try:
        from_c = from_currency.upper().strip()
        to_c = to_currency.upper().strip()

        url = f"https://api.exchangerate-api.com/v4/latest/{from_c}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())

        rates = data.get("rates", {})
        if to_c not in rates:
            return {"error": f"Unknown currency: {to_c}"}

        rate = rates[to_c]
        result = amount * rate

        return {
            "amount": amount,
            "from_currency": from_c,
            "to_currency": to_c,
            "rate": rate,
            "result": round(result, 2),
        }
    except urllib.error.HTTPError as e:
        return {"error": f"Unknown currency or API error: {e.code}"}
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Web Search Tool (Mock)
# -----------------------------

@registry.register(aliases=["search", "google"])
@tool_output(llm_format=lambda x: "\n".join(f"* {r['title']}: {r['snippet']}" for r in x.get('results', [])) if x.get('results') else x.get('error', 'No results'))
def web_search(
    query: str,
    num_results: int = 5,
) -> dict[str, Any]:
    """Search the web for information.

    Note: This is a mock implementation. For real search, integrate with
    SerpAPI, Brave Search API, or similar service.

    Args:
        query: The search query.
        num_results: Number of results to return (default 5, max 10).

    Returns:
        Dictionary with search results or info about setting up real search.
    """
    # Check for API key in environment
    api_key = os.environ.get("SERPAPI_KEY") or os.environ.get("BRAVE_API_KEY")

    if not api_key:
        return {
            "info": "Web search requires an API key. Set SERPAPI_KEY or BRAVE_API_KEY environment variable.",
            "query": query,
            "results": [
                {
                    "title": f"Mock result for: {query}",
                    "snippet": "This is a placeholder. Configure a search API for real results.",
                    "url": "https://example.com",
                }
            ],
        }

    # If we have SerpAPI key
    if os.environ.get("SERPAPI_KEY"):
        try:
            params = urllib.parse.urlencode({
                "q": query,
                "api_key": os.environ["SERPAPI_KEY"],
                "num": min(num_results, 10),
            })
            url = f"https://serpapi.com/search.json?{params}"
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())

            results = []
            for item in data.get("organic_results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                })
            return {"query": query, "results": results}
        except Exception as e:
            return {"error": f"Search failed: {e}"}

    return {"error": "No valid search API configured"}
