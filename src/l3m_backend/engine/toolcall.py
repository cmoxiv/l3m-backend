"""
Tool call parsing module.

Provides unified parsing for tool calls across different LLM formats:
- Hermes/ChatML: <tool_call>...</tool_call>
- OpenAI/Contract: {"type": "tool_call"|"function", ...}
- Llama 3: [func(arg=val)] (future)
- Mistral: [TOOL_CALLS] [...] (future)
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolCallFormat(Enum):
    """Supported tool call formats."""

    UNKNOWN = "unknown"
    CONTRACT = "contract"       # {"type": "tool_call", "name": ..., "arguments": ...}
    OPENAI = "openai"           # {"type": "function", "name": ..., "parameters": ...}
    HERMES = "hermes"           # <tool_call>{"name": ..., "arguments": ...}</tool_call>
    LLAMA3 = "llama3"           # [func(arg=val)] - future
    MISTRAL = "mistral"         # [TOOL_CALLS] [...] - future
    NATIVE = "native"           # llama-cpp-python native format


@dataclass
class ToolCall:
    """Normalized tool call representation."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str | None = None
    format: ToolCallFormat = ToolCallFormat.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard dict format."""
        return {
            "type": "tool_call",
            "name": self.name,
            "arguments": self.arguments,
        }


class JsonCleaner:
    """Clean malformed JSON from LLM output."""

    @staticmethod
    def clean(text: str) -> str:
        """Clean and normalize JSON text.

        Handles common LLM JSON issues:
        1. HTML entity decoding (&#34; -> ")
        2. Single quotes -> double quotes (with care for nested strings)
        3. Trailing commas
        4. Unquoted keys
        5. Python-style booleans (True/False -> true/false)
        """
        if not text:
            return text

        # 1. HTML entity decode
        text = html.unescape(text)

        # 2. Python booleans -> JSON booleans
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        text = re.sub(r'\bNone\b', 'null', text)

        # 3. Try to fix single quotes (careful with apostrophes)
        # Only convert if it looks like JSON with single quotes
        if text.startswith("{'") or "': '" in text or "': {" in text:
            # Simple single-to-double quote conversion for dict-like strings
            text = JsonCleaner._convert_single_quotes(text)

        # 4. Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        # 5. Try to quote unquoted keys (simple cases)
        # Match: {key: or , key: where key is unquoted
        text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', text)

        return text

    @staticmethod
    def _convert_single_quotes(text: str) -> str:
        """Convert single quotes to double quotes in JSON-like strings."""
        result = []
        in_string = False
        string_char = None
        i = 0

        while i < len(text):
            char = text[i]

            if not in_string:
                if char == '"':
                    in_string = True
                    string_char = '"'
                    result.append(char)
                elif char == "'":
                    # Start of single-quoted string, convert to double
                    in_string = True
                    string_char = "'"
                    result.append('"')
                else:
                    result.append(char)
            else:
                if char == '\\' and i + 1 < len(text):
                    # Escape sequence
                    result.append(char)
                    result.append(text[i + 1])
                    i += 1
                elif char == string_char:
                    # End of string
                    in_string = False
                    result.append('"' if string_char == "'" else char)
                    string_char = None
                elif char == '"' and string_char == "'":
                    # Double quote inside single-quoted string, escape it
                    result.append('\\"')
                else:
                    result.append(char)
            i += 1

        return ''.join(result)


class FormatParser:
    """Parse tool calls from various LLM output formats."""

    # Type field aliases that indicate a tool call
    TYPE_ALIASES = {"tool_call", "function", "function_call", "tool_use", "tool"}

    # Name key aliases
    NAME_ALIASES = ("name", "function_name", "tool_name", "tool")

    # Arguments key aliases
    ARGS_ALIASES = ("arguments", "parameters", "params", "args", "input")

    def __init__(self):
        self.cleaner = JsonCleaner()

    def detect_format(self, content: str) -> ToolCallFormat:
        """Detect the tool call format from content."""
        if not content:
            return ToolCallFormat.UNKNOWN

        # Check for Hermes XML tags
        if "<tool_call>" in content.lower():
            return ToolCallFormat.HERMES

        # Check for Mistral format
        if "[TOOL_CALLS]" in content:
            return ToolCallFormat.MISTRAL

        # Check for Llama 3 pythonic format
        if re.search(r'\[\s*\w+\s*\(', content):
            return ToolCallFormat.LLAMA3

        # Try to parse as JSON and check type
        try:
            # Find JSON in content
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end > start:
                data = json.loads(content[start:end + 1])
                if isinstance(data, dict):
                    type_val = data.get("type", "")
                    if type_val == "tool_call":
                        return ToolCallFormat.CONTRACT
                    if type_val in ("function", "function_call"):
                        return ToolCallFormat.OPENAI
                    # Has name field but no type - could be either
                    if "name" in data:
                        return ToolCallFormat.CONTRACT
        except json.JSONDecodeError:
            pass

        return ToolCallFormat.UNKNOWN

    def parse(self, content: str) -> ToolCall | None:
        """Parse tool call from content string.

        Returns normalized ToolCall or None if not a tool call.
        """
        if not content:
            return None

        fmt = self.detect_format(content)

        if fmt == ToolCallFormat.HERMES:
            return self._parse_hermes(content)
        elif fmt in (ToolCallFormat.CONTRACT, ToolCallFormat.OPENAI):
            return self._parse_json(content, fmt)
        elif fmt == ToolCallFormat.LLAMA3:
            return self._parse_llama3(content)
        elif fmt == ToolCallFormat.MISTRAL:
            return self._parse_mistral(content)

        # Try generic JSON parsing as fallback
        return self._parse_json(content, ToolCallFormat.UNKNOWN)

    def _parse_hermes(self, content: str) -> ToolCall | None:
        """Parse Hermes <tool_call> format."""
        # Extract content between tags
        match = re.search(
            r'<tool_call>\s*(.*?)\s*</tool_call>',
            content,
            re.DOTALL | re.IGNORECASE
        )
        if not match:
            return None

        json_str = self.cleaner.clean(match.group(1).strip())
        try:
            data = json.loads(json_str)
            return self._normalize(data, ToolCallFormat.HERMES, content)
        except json.JSONDecodeError:
            return None

    def _parse_json(self, content: str, fmt: ToolCallFormat) -> ToolCall | None:
        """Parse JSON-based tool call formats."""
        # Try multiple extraction strategies
        strategies = [
            self._extract_direct,
            self._extract_code_block,
            self._extract_braces,
            self._extract_any_json,
        ]

        for strategy in strategies:
            data = strategy(content)
            if data:
                result = self._normalize(data, fmt, content)
                if result:
                    return result

        return None

    def _extract_direct(self, content: str) -> dict | None:
        """Try direct JSON parsing."""
        cleaned = self.cleaner.clean(content.strip())
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def _extract_code_block(self, content: str) -> dict | None:
        """Extract JSON from markdown code blocks."""
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if match:
            cleaned = self.cleaner.clean(match.group(1).strip())
            try:
                data = json.loads(cleaned)
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
        return None

    def _extract_braces(self, content: str) -> dict | None:
        """Extract JSON from outermost braces."""
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end > start:
            cleaned = self.cleaner.clean(content[start:end + 1])
            try:
                data = json.loads(cleaned)
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
        return None

    def _extract_any_json(self, content: str) -> dict | None:
        """Find any valid JSON object in content."""
        # Match nested braces
        pattern = re.compile(r'\{[^{}]*\}|\{(?:[^{}]|\{[^{}]*\})*\}')
        for match in pattern.finditer(content):
            cleaned = self.cleaner.clean(match.group())
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict) and self._looks_like_tool_call(data):
                    return data
            except json.JSONDecodeError:
                continue
        return None

    def _looks_like_tool_call(self, data: dict) -> bool:
        """Check if dict looks like a tool call."""
        # Has a type we recognize
        if data.get("type") in self.TYPE_ALIASES:
            return True
        # Has a name and arguments-like key
        if any(k in data for k in self.NAME_ALIASES):
            if any(k in data for k in self.ARGS_ALIASES):
                return True
            # Just name is also valid (no-arg tool)
            return True
        return False

    def _parse_llama3(self, content: str) -> ToolCall | None:
        """Parse Llama 3 pythonic format: [func(arg=val)].

        Handles:
        - [get_weather(location='Paris')]
        - [calculate(x=10, y=20, precise=True)]
        - [get_weather(location=Paris)]  # unquoted strings
        """
        # Extract function call from brackets
        match = re.search(r'\[\s*(\w+)\s*\(([^)]*)\)\s*\]', content)
        if not match:
            return None

        name = match.group(1)
        params_str = match.group(2).strip()

        # Parse parameters
        args = {}
        if params_str:
            # Split on comma, but be careful with nested strings
            # Pattern: key=value pairs
            param_pattern = re.compile(
                r"(\w+)\s*=\s*"  # key=
                r"(?:"
                r"'([^']*)'"     # single-quoted string
                r"|\"([^\"]*)\""  # double-quoted string
                r"|(\d+\.?\d*)"  # number
                r"|(True|False|None)"  # Python booleans
                r"|(\w+)"        # unquoted identifier
                r")"
            )
            for m in param_pattern.finditer(params_str):
                key = m.group(1)
                # Check which capture group matched
                if m.group(2) is not None:  # single-quoted
                    args[key] = m.group(2)
                elif m.group(3) is not None:  # double-quoted
                    args[key] = m.group(3)
                elif m.group(4) is not None:  # number
                    num = m.group(4)
                    args[key] = float(num) if '.' in num else int(num)
                elif m.group(5) is not None:  # Python booleans
                    val = m.group(5)
                    args[key] = True if val == "True" else False if val == "False" else None
                elif m.group(6) is not None:  # unquoted identifier
                    args[key] = m.group(6)

        return ToolCall(
            name=name,
            arguments=args,
            raw=content,
            format=ToolCallFormat.LLAMA3,
        )

    def _parse_mistral(self, content: str) -> ToolCall | None:
        """Parse Mistral format: [TOOL_CALLS] [{"name": ..., "arguments": ...}].

        The format is:
        [TOOL_CALLS] [{"name": "func", "arguments": {"param": "value"}}]
        """
        # Strip [TOOL_CALLS] prefix and find the JSON array
        if "[TOOL_CALLS]" not in content:
            return None

        # Find the JSON array after [TOOL_CALLS]
        idx = content.find("[TOOL_CALLS]")
        rest = content[idx + len("[TOOL_CALLS]"):].strip()

        # Find the array
        start = rest.find('[')
        if start == -1:
            return None

        # Extract and parse JSON array
        cleaned = self.cleaner.clean(rest[start:])
        try:
            data = json.loads(cleaned)
            if not isinstance(data, list) or not data:
                return None

            # Take first tool call
            tc = data[0]
            if not isinstance(tc, dict):
                return None

            name = tc.get("name", "")
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if name:
                return ToolCall(
                    name=name,
                    arguments=args if isinstance(args, dict) else {},
                    raw=content,
                    format=ToolCallFormat.MISTRAL,
                )
        except json.JSONDecodeError:
            pass

        return None

    def _normalize(self, data: dict, fmt: ToolCallFormat, raw: str) -> ToolCall | None:
        """Normalize parsed dict to ToolCall."""
        if not isinstance(data, dict):
            return None

        # Check type field
        type_val = data.get("type", "")
        has_valid_type = type_val in self.TYPE_ALIASES

        # Find name
        name = None
        for key in self.NAME_ALIASES:
            if key in data and isinstance(data[key], str) and data[key]:
                name = data[key]
                break

        # Find arguments
        args = {}
        for key in self.ARGS_ALIASES:
            if key in data:
                val = data[key]
                if isinstance(val, dict):
                    args = val
                    break
                elif isinstance(val, str):
                    # Arguments as JSON string
                    try:
                        args = json.loads(val)
                    except json.JSONDecodeError:
                        pass
                    break

        # Valid if: has recognized type OR has name
        if (has_valid_type or not type_val) and name:
            return ToolCall(
                name=name,
                arguments=args,
                raw=raw,
                format=fmt if fmt != ToolCallFormat.UNKNOWN else (
                    ToolCallFormat.CONTRACT if type_val == "tool_call"
                    else ToolCallFormat.OPENAI if type_val in ("function", "function_call")
                    else ToolCallFormat.CONTRACT
                ),
            )

        return None


# Singleton instance
_handler: ToolCallHandler | None = None


class ToolCallHandler:
    """Main interface for tool call parsing.

    Use get_handler() to get the singleton instance.
    """

    def __init__(self):
        self.parser = FormatParser()
        self.cleaner = JsonCleaner()

    def parse(self, content: str) -> ToolCall | None:
        """Parse tool call from any format.

        Args:
            content: Raw LLM output string.

        Returns:
            Normalized ToolCall or None if not a tool call.
        """
        return self.parser.parse(content)

    def parse_native_chunks(self, tool_calls_data: list[dict]) -> list[ToolCall]:
        """Parse native tool calls from llama-cpp-python streaming chunks.

        Args:
            tool_calls_data: Accumulated tool call data from streaming.
                Format: [{"id": "...", "function": {"name": "...", "arguments": "..."}}]

        Returns:
            List of normalized ToolCall objects.
        """
        results = []
        for tc in tool_calls_data:
            func = tc.get("function", {})
            name = func.get("name", "")
            args_raw = func.get("arguments", "{}")

            if not name:
                continue

            # Parse arguments
            args = {}
            if isinstance(args_raw, str):
                cleaned = self.cleaner.clean(args_raw)
                try:
                    args = json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
            elif isinstance(args_raw, dict):
                args = args_raw

            results.append(ToolCall(
                name=name,
                arguments=args,
                raw=json.dumps(tc),
                format=ToolCallFormat.NATIVE,
            ))

        return results

    def format_result(
        self,
        tool_name: str,
        result: Any,
        fmt: ToolCallFormat = ToolCallFormat.CONTRACT
    ) -> str:
        """Format tool result for the model.

        Args:
            tool_name: Name of the tool that was called.
            result: Result from tool execution.
            fmt: Format to use for the result.

        Returns:
            Formatted result string.
        """
        # Convert result to string if needed
        if isinstance(result, dict):
            result_str = json.dumps(result)
        else:
            result_str = str(result)

        # Format based on expected model format
        if fmt == ToolCallFormat.HERMES:
            return f"<tool_response>\n{result_str}\n</tool_response>"
        else:
            # Default contract format
            return f"[Tool Result]: {result_str}"


def get_handler() -> ToolCallHandler:
    """Get the singleton ToolCallHandler instance."""
    global _handler
    if _handler is None:
        _handler = ToolCallHandler()
    return _handler
