"""
Helper functions for the tool registry.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, create_model

from l3m_backend.core.exceptions import ToolParseError


def _normalize_name(name: str) -> str:
    """Normalize tool name to valid identifier."""
    name = (name or "").strip()
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name)


def _create_params_model(fn: Callable, tool_name: str) -> type[BaseModel]:
    """Create a Pydantic model from function signature."""
    sig = inspect.signature(fn)
    hints = _safe_get_type_hints(fn)

    fields: dict[str, Any] = {}

    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = hints.get(param.name, str)

        if param.default is inspect.Parameter.empty:
            fields[param.name] = (annotation, ...)
        else:
            fields[param.name] = (annotation, param.default)

    return create_model(f"{tool_name}_Params", **fields)


def _safe_get_type_hints(fn: Callable) -> dict[str, Any]:
    """Get type hints with fallback for unresolvable annotations."""
    try:
        return get_type_hints(fn) or {}
    except Exception:
        return getattr(fn, "__annotations__", {}) or {}


def parse_model_response(text: str) -> dict[str, Any]:
    """Parse JSON response from model."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ToolParseError(f"Invalid JSON: {e}\n---\n{text}") from e
