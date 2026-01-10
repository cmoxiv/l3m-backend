"""
Exception classes for the tool registry.
"""


class ToolError(Exception):
    """Base exception for tool-related errors."""


class ToolNotFoundError(ToolError):
    """Tool not found in registry."""


class ToolValidationError(ToolError):
    """Tool argument validation failed."""


class ToolParseError(ToolError):
    """Failed to parse model output."""
