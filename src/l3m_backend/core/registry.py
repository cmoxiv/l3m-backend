"""
Tool Registry for managing and executing tools.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import ValidationError

from l3m_backend.core.exceptions import (
    ToolError,
    ToolNotFoundError,
    ToolParseError,
    ToolValidationError,
)
from l3m_backend.core.helpers import _normalize_name
from l3m_backend.core.datamodels import ToolEntry, ToolResult


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        fn: Callable | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        aliases: list[str] | None = None,
    ) -> Callable:
        """
        Register a tool. Can be used as decorator with or without arguments.

        Usage:
            @registry.register
            def my_tool(x: int) -> str: ...

            @registry.register(name="custom_name", aliases=["mt"])
            def my_tool(x: int) -> str: ...
        """
        def decorator(func: Callable) -> Callable:
            canonical = _normalize_name(name or func.__name__)
            tool_aliases = [_normalize_name(a) for a in (aliases or [])]

            if canonical in self._tools:
                raise ToolError(f"Tool name collision: {canonical}")

            entry = ToolEntry(
                name=canonical,
                callable_fn=func,
                aliases=tool_aliases,
                description=description,
            )
            self._tools[canonical] = entry

            for alias in tool_aliases:
                if alias in self._aliases and self._aliases[alias] != canonical:
                    raise ToolError(f"Alias collision: {alias} -> {self._aliases[alias]}")
                self._aliases[alias] = canonical

            # Attach metadata to function
            func.__tool_name__ = canonical
            func.__tool_aliases__ = tool_aliases

            return func

        # Handle @registry.register vs @registry.register(...)
        if fn is not None:
            return decorator(fn)
        return decorator

    def get(self, name_or_alias: str) -> ToolEntry:
        """Resolve a tool by name or alias."""
        key = _normalize_name(name_or_alias)
        canonical = self._aliases.get(key, key)

        if canonical not in self._tools:
            raise ToolNotFoundError(f"Unknown tool: {name_or_alias}")

        return self._tools[canonical]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool with the given arguments."""
        entry = self.get(name)
        params_model = entry.get_params_model()

        try:
            validated = params_model.model_validate(arguments)
        except ValidationError as e:
            raise ToolValidationError(str(e)) from e

        output = entry.callable_fn(**validated.model_dump())

        return ToolResult(
            name=entry.name,
            arguments=arguments,
            output=output,
        )

    def execute_call(self, call: dict[str, Any]) -> ToolResult:
        """
        Execute a tool call dict.
        Expected format: {"type": "tool_call", "name": "...", "arguments": {...}}
        """
        if call.get("type") != "tool_call":
            raise ToolParseError("Not a tool_call object")

        name = call.get("name")
        if not name:
            raise ToolParseError("tool_call missing 'name'")

        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ToolParseError("'arguments' must be a dict")

        return self.execute(name, arguments)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Get all tools as OpenAI-style tool specs."""
        return [entry.to_openai_spec() for entry in self._tools.values()]

    def to_registry_json(self) -> str:
        """Get registry as JSON for system prompts."""
        tools = []
        for entry in self._tools.values():
            spec = entry.to_openai_spec()
            tools.append({
                "name": spec["function"]["name"],
                "description": spec["function"]["description"],
                "parameters": spec["function"]["parameters"],
                "aliases": entry.aliases,
            })
        return json.dumps({"tools": tools}, indent=2)

    def to_minimal_list(self) -> str:
        """Generate compact tool list (name + one-line description).

        Returns a minimal representation suitable for smaller context windows.
        Format: "- tool_name(param1, param2): Brief description"
        """
        lines = []
        for name, entry in self._tools.items():
            # Get parameter names from params model schema
            schema = entry.get_params_model().model_json_schema()
            params = ", ".join(schema.get("properties", {}).keys())
            # Get first line of description, truncated to 80 chars
            desc = entry.get_description().split("\n")[0]
            if len(desc) > 80:
                desc = desc[:77] + "..."
            lines.append(f"- {name}({params}): {desc}")
        return "\n".join(lines)

    def build_system_prompt(self) -> str:
        """Build the system prompt with tool contract."""
        registry_json = self.to_registry_json()
        return f"""You have access to tools to help you with your response.

TOOL_REGISTRY_JSON:
{registry_json}

OUTPUT CONTRACT (STRICT):
- If you decide to call a tool, output ONLY one JSON object (no extra text):
  {{"type": "tool_call", "name": "<tool name or alias>", "arguments": {{...}}}}

- Otherwise output ONLY:
  {{"type": "final", "content": "..."}}

RULES:
- Use ONLY tool names/aliases from TOOL_REGISTRY_JSON.
- Arguments must satisfy the tool's JSON Schema.
- Never guess missing required arguments. Ask the user if needed."""

    def __contains__(self, name: str) -> bool:
        try:
            self.get(name)
            return True
        except ToolNotFoundError:
            return False

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
