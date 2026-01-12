"""Web search tool using DuckDuckGo."""

import json
import re
import urllib.parse
import urllib.request
from typing import Any

from pydantic import Field

from l3m_backend.core import tool_output
from l3m_backend.engine.context import get_current_engine
from l3m_backend.tools._registry import registry


def _fetch_page_content(url: str, max_chars: int = 2000) -> str:
    """Fetch and extract readable text from a URL.

    Args:
        url: The URL to fetch.
        max_chars: Maximum characters to return.

    Returns:
        Extracted text content, or empty string on failure.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LLMTools/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Strip script and style tags first
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        # Strip all remaining HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text[:max_chars]
    except Exception:
        return ""


def _fetch_all_results(results: list[dict], max_per_page: int = 1500) -> str:
    """Fetch content from all result URLs and concatenate.

    Args:
        results: List of search result dicts with 'url' and 'title' keys.
        max_per_page: Maximum characters to fetch per page.

    Returns:
        Concatenated content from all pages.
    """
    contents = []
    for r in results:
        url = r.get("url", "")
        title = r.get("title", "")
        if url:
            content = _fetch_page_content(url, max_per_page)
            if content:
                contents.append(f"## {title}\n{content}")
    return "\n\n".join(contents)


def _parse_ddg_html(html: str, num_results: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML lite results."""
    results = []

    # Find result blocks - DDG lite uses simple HTML structure
    # Pattern: <a rel="nofollow" href="URL">TITLE</a>
    pattern = r'<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
    matches = re.findall(pattern, html)

    for url, title in matches[:num_results * 2]:  # Get extra to account for skips
        # Extract actual URL from DDG redirect (//duckduckgo.com/l/?uddg=...)
        if 'uddg=' in url:
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            if 'uddg' in parsed:
                url = parsed['uddg'][0]
        # Skip internal DDG links (but allow redirects we just extracted)
        elif 'duckduckgo.com' in url:
            continue

        # Decode HTML entities
        title = title.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&#x27;', "'")
        results.append({
            "title": title.strip(),
            "url": url,
            "snippet": "",  # Lite version has limited snippets
        })

        if len(results) >= num_results:
            break

    return results


def _ddg_instant_answer(query: str) -> dict[str, Any] | None:
    """Try DuckDuckGo Instant Answer API first."""
    try:
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
        url = f"https://api.duckduckgo.com/?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "LLMTools/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        # Check for abstract (Wikipedia-style answer)
        if data.get("Abstract"):
            return {
                "type": "instant_answer",
                "title": data.get("Heading", query),
                "answer": data.get("Abstract"),
                "source": data.get("AbstractSource", ""),
                "url": data.get("AbstractURL", ""),
            }

        # Check for direct answer
        if data.get("Answer"):
            return {
                "type": "instant_answer",
                "title": query,
                "answer": data.get("Answer"),
                "source": "DuckDuckGo",
                "url": "",
            }

        # Check for related topics
        if data.get("RelatedTopics"):
            results = []
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:100],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                    })
            if results:
                return {"type": "related", "query": query, "results": results}

    except Exception:
        pass
    return None


def _summarize_content(content: str, query: str, max_chars: int = 3000) -> str:
    """Summarize content using the LLM if it's too long.

    Args:
        content: The content to potentially summarize.
        query: The original search query for context.
        max_chars: Threshold above which to summarize.

    Returns:
        Original content if short, or summarized content.
    """
    if len(content) <= max_chars:
        return content

    engine = get_current_engine()
    if engine is None:
        # No engine available, truncate instead
        return content[:max_chars] + "..."

    try:
        summary = engine.chat(
            f"Search query: {query}\n\nContent:\n{content}",
            ignore_history=True,
            temp_system="Summarize this web search content concisely. Keep key facts and information relevant to the search query. Be brief but informative.",
        )
        return str(summary)  # ignore_history always returns str
    except Exception:
        # Fallback to truncation
        return content[:max_chars] + "..."


def _format_search_results(x: dict) -> str:
    """Format search results for LLM consumption."""
    # For instant answers, return the answer directly
    if x.get('type') == 'instant_answer':
        return x.get('answer', '')

    # For fetched content, return the combined content
    if x.get('content'):
        return x['content']

    # Fallback to titles if no content
    if x.get('results'):
        return "\n".join(f"* {r['title']}: {r.get('url', '')}" for r in x['results'])

    return x.get('error', 'No results')


@registry.register(aliases=["search", "ddg", "google"])
@tool_output(llm_format=_format_search_results)
def web_search(
    query: str = Field(description="Search query - use the user's question or topic as the query"),
    num_results: int = 5,
) -> dict[str, Any]:
    """Search the web for real-time information, or facts beyond your cut-off date.

    Use this tool when:
    - User asks about current events, news, or future dates
    - User asks about something you're unsure of or don't have info on
    - User asks about somthing beyond your cut-off date
    - User explicitly asks to search or check the web

    Args:
        query: The search query - can be the user's question directly.
        num_results: Number of results to return (default 5, max 10).

    Returns:
        Dictionary with search results or instant answer.
    """
    num_results = min(max(num_results, 1), 10)

    # Try instant answer first
    instant = _ddg_instant_answer(query)
    if instant:
        return instant

    # Fall back to HTML search (lite version, no JS required)
    try:
        params = urllib.parse.urlencode({"q": query})
        url = f"https://lite.duckduckgo.com/lite/?{params}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; LLMTools/1.0)",
                "Accept": "text/html",
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        results = _parse_ddg_html(html, num_results)

        if results:
            # Fetch content from all result URLs
            content = _fetch_all_results(results)
            # Summarize if content is too long
            content = _summarize_content(content, query)
            return {"query": query, "results": results, "content": content}

        return {
            "query": query,
            "results": [],
            "content": "",
            "info": "No results found. Try a different query.",
        }

    except Exception as e:
        return {"error": f"Search failed: {e}"}
