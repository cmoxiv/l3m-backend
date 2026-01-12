"""Wikipedia lookup tool."""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pydantic import Field

from l3m_backend.core import tool_output
from l3m_backend.tools._registry import registry


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
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
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
