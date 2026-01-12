"""
Legacy priming generator for tool contracts.

Generates contextual examples based on currently enabled tools
to help the LLM understand how to use them effectively.

Examples are loaded from ~/.l3m/legacy_priming.yaml and filtered based
on which tools are actually available in the registry.

Note: This is the legacy implementation that consumes context window.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from l3m_backend.core import ToolRegistry

DEFAULT_LEGACY_PRIMING_FILE = Path.home() / ".l3m" / "legacy_priming.yaml"
DEFAULT_PRIMING_JSON_FILE = Path.home() / ".l3m" / "tool-priming.json"

DEFAULT_PRIMING_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "What time is it?"},
    {"role": "assistant", "content": '{"type": "tool_call", "name": "get_time", "arguments": {}}'},
    {"role": "tool", "content": '{"type": "tool_result", "name": "get_time", "content": "14:30"}'},
    {"role": "assistant", "content": "It's 2:30 PM."},
    {"role": "user", "content": "What's 25 * 4?"},
    {"role": "assistant", "content": '{"type": "tool_call", "name": "calculate", "arguments": {"expression": "25 * 4"}}'},
    {"role": "tool", "content": '{"type": "tool_result", "name": "calculate", "content": "100"}'},
    {"role": "assistant", "content": "25 times 4 equals 100."},
    {"role": "user", "content": "What major events are planned for next year?"},
    {"role": "assistant", "content": '{"type": "tool_call", "name": "web_search", "arguments": {"query": "major events next year"}}'},
    {"role": "tool", "content": '{"type": "tool_result", "name": "web_search", "content": "* FIFA World Cup 2026\\n* Winter Olympics 2026"}'},
    {"role": "assistant", "content": "Some major events planned include the FIFA World Cup 2026 and the Winter Olympics 2026."},
    {"role": "user", "content": "I need to convert currency and then calculate a tip. Help me plan this."},
    {"role": "assistant", "content": '{"type": "tool_call", "name": "plan", "arguments": {"task": "convert currency and calculate tip"}}'},
    {"role": "tool", "content": '{"type": "tool_result", "name": "plan", "content": "1. Use currency_convert to convert the amount\\n2. Use calculate to compute the tip percentage"}'},
    {"role": "assistant", "content": "Here\\'s the plan:\\n1. First convert the currency using currency_convert\\n2. Then calculate the tip using calculate\\n\\nWhat amount and currencies would you like to convert?"},
]


def load_priming_messages(path: Path | None = None) -> list[dict[str, str]]:
    """Load priming messages from JSON file, creating default if needed.

    Args:
        path: Path to priming JSON file. Defaults to ~/.l3m/tool-priming.json

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    priming_path = path or DEFAULT_PRIMING_JSON_FILE

    # Create default file if it doesn't exist
    if not priming_path.exists():
        priming_path.parent.mkdir(parents=True, exist_ok=True)
        priming_path.write_text(json.dumps(DEFAULT_PRIMING_MESSAGES, indent=2))

    try:
        content = priming_path.read_text().strip()
        if content:
            return json.loads(content)
    except (json.JSONDecodeError, OSError):
        pass

    return DEFAULT_PRIMING_MESSAGES


class LegacyPrimingGenerator:
    """Generates dynamic priming examples based on available tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        priming_file: Path | None = None,
    ):
        """Initialize the legacy priming generator.

        Args:
            registry: The tool registry to check for available tools.
            priming_file: Path to the priming YAML file.
                         Defaults to ~/.l3m/legacy_priming.yaml
        """
        self.registry = registry
        self.priming_file = priming_file or DEFAULT_LEGACY_PRIMING_FILE
        self._available_tools: set[str] = set()
        self._refresh_available_tools()

    def _refresh_available_tools(self) -> None:
        """Update the set of available tool names from registry."""
        self._available_tools = {entry.name for entry in self.registry}

    def _load_examples(self) -> list[dict[str, Any]]:
        """Load examples from YAML file.

        Returns:
            List of example dictionaries, or empty list if file doesn't exist.
        """
        if not self.priming_file.exists():
            return []

        try:
            with open(self.priming_file) as f:
                data = yaml.safe_load(f)
            return data.get("examples", []) if data else []
        except (yaml.YAMLError, OSError):
            return []

    def _tools_available(self, example: dict[str, Any]) -> bool:
        """Check if all tools in an example are registered.

        Args:
            example: Example dictionary with 'steps' containing tool calls.

        Returns:
            True if all tools in the example are available.
        """
        for step in example.get("steps", []):
            tool_name = step.get("tool")
            if tool_name and tool_name not in self._available_tools:
                return False
        return True

    def _format_example(self, example: dict[str, Any]) -> str:
        """Format a single example for the contract.

        Args:
            example: Example dictionary with user query, steps, and response.

        Returns:
            Formatted example string showing the conversation flow.
        """
        lines = [f'user: "{example["user"]}"']

        # Add planning thought if present (for chaining examples)
        if "planning" in example:
            lines.append(f'assistant (thinking): {example["planning"]}')

        # Add each tool call and result
        for step in example.get("steps", []):
            tool_name = step["tool"]
            args = step.get("args", {})
            result = step.get("result", "")

            args_json = json.dumps(args)
            lines.append(
                f'assistant: {{"type": "tool_call", "name": "{tool_name}", "arguments": {args_json}}}'
            )
            result_json = json.dumps({"type": "tool_result", "name": tool_name, "content": result})
            lines.append(f'tool: {result_json}')

        # Add final response
        lines.append(f'assistant: {example["response"]}')

        return "\n".join(lines)

    def generate_priming_section(self) -> str:
        """Generate the complete priming section.

        Selects one example from each complexity level (simple, multi, chaining)
        where all required tools are available.

        Returns:
            Formatted priming section string, or empty string if no examples.
        """
        self._refresh_available_tools()
        examples = self._load_examples()

        if not examples:
            return ""

        sections = []

        # Select one example per level, in order of complexity
        for level in ["simple", "multi", "chaining"]:
            for example in examples:
                if example.get("level") == level and self._tools_available(example):
                    sections.append(self._format_example(example))
                    break  # One example per level

        if not sections:
            return ""

        return "EXAMPLES:\n\n" + "\n\n".join(sections)


def generate_legacy_priming(registry: ToolRegistry) -> list[dict[str, str]]:
    """Load priming messages from JSON file.

    Args:
        registry: The tool registry (unused, kept for compatibility).

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    return load_priming_messages()
