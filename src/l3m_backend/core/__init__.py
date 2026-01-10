"""
Core module for the l3m_backend package.

Provides the Tool Registry and related models for managing LLM tool calling.
"""

from l3m_backend.core.decorators import tool_output
from l3m_backend.core.exceptions import (
    ToolError,
    ToolNotFoundError,
    ToolParseError,
    ToolValidationError,
)
from l3m_backend.core.helpers import parse_model_response
from l3m_backend.core.datamodels import ToolEntry, ToolOutput, ToolResult
from l3m_backend.core.registry import ToolRegistry

__all__ = [
    # Registry
    "ToolRegistry",
    # Models
    "ToolEntry",
    "ToolOutput",
    "ToolResult",
    # Exceptions
    "ToolError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolParseError",
    # Decorators
    "tool_output",
    # Helpers
    "parse_model_response",
]
