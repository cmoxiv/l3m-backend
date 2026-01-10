"""
Tool scaffold generator for creating new user tools.

Usage:
    l3m-tools create my_tool
    l3m-tools create git-status --wrap "git status"
"""

from l3m_backend.tools.generator.core import (
    BASIC_TOOL_TEMPLATE,
    USER_TOOLS_DIR,
    WRAPPER_ARGS_TOOL_TEMPLATE,
    WRAPPER_TOOL_TEMPLATE,
    generate_tool,
    generate_tool_name,
)

__all__ = [
    "USER_TOOLS_DIR",
    "BASIC_TOOL_TEMPLATE",
    "WRAPPER_TOOL_TEMPLATE",
    "WRAPPER_ARGS_TOOL_TEMPLATE",
    "generate_tool_name",
    "generate_tool",
]
