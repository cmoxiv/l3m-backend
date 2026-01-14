"""
Example user-defined tool for l3m-chat.

This file demonstrates how to create custom tools.
Place your tool in ~/.l3m/tools/<name>/__init__.py

Tools use the @registry.register decorator to make them available
to the LLM, and @tool_output to format the response.
"""

from typing import Any

from l3m_backend.core import tool_output
from l3m_backend.tools._registry import registry


@registry.register(
    name="example_tool",
    aliases=["ext"],
    description="Example custom tool that echoes input",
)
@tool_output(llm_format="{message}")
def example_tool(
    text: str,
    uppercase: bool = False,
) -> dict[str, Any]:
    """
    Example tool that echoes the input text.

    Args:
        text: The text to echo back.
        uppercase: If True, convert text to uppercase.

    Returns:
        Dictionary with the echoed message.
    """
    message = text.upper() if uppercase else text
    return {
        "message": f"[Example Tool] {message}",
        "original": text,
        "uppercase": uppercase,
    }
