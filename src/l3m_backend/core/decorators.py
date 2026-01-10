"""
Decorators for the tool registry.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from l3m_backend.core.datamodels import ToolOutput


def tool_output(
    llm_format: str | Callable[[Any], str] | None = None,
    gui_format: str | Callable[[Any], str] | None = None,
) -> Callable:
    """
    Decorator that wraps function return value in ToolOutput.

    Usage:
        @tool_output(llm_format="{location}: {temperature}°{unit}")
        def get_weather(...): ...

        @tool_output(llm_format=lambda x: f"Temp: {x['temperature']}")
        def get_weather(...): ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> ToolOutput:
            result = fn(*args, **kwargs)

            # Resolve format strings/callables
            llm_str = _resolve_format(llm_format, result)
            gui_str = _resolve_format(gui_format, result)

            return ToolOutput.create(result, llm_format=llm_str, gui_format=gui_str)

        return wrapper
    return decorator


def _resolve_format(fmt: str | Callable[[Any], str] | None, data: Any) -> str | None:
    """Resolve format string or callable to final string."""
    if fmt is None:
        return None
    if callable(fmt):
        return fmt(data)
    if isinstance(fmt, str) and isinstance(data, dict):
        try:
            return fmt.format(**data)
        except KeyError:
            return fmt
    return str(fmt)
