"""
Time tool implementation.
"""

from datetime import datetime

from l3m_backend.core import tool_output
from l3m_backend.tools._registry import registry


@registry.register(aliases=["time", "t"])
@tool_output(llm_format="{current_time}")
def get_time() -> dict[str, str]:
    """Get the current system time.

    Returns:
        Dictionary with key:
            - current_time: Formatted timestamp string (YYYY-MM-DD HH:MM:SS)

    Example:
        >>> get_time()
        {"current_time": "2026-01-06 12:34:56"}
    """
    now = datetime.now()
    return {"current_time": now.strftime("%Y-%m-%d %H:%M:%S")}
