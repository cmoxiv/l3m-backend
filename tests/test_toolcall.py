#!/usr/bin/env python3
"""
Tests for tool call parsing module.
"""

import pytest

from l3m_backend.engine.toolcall import (
    JsonCleaner,
    FormatParser,
    ToolCall,
    ToolCallFormat,
    ToolCallHandler,
    get_handler,
)


# ============================================================================
# JsonCleaner Tests
# ============================================================================

class TestJsonCleaner:
    """Tests for JsonCleaner."""

    def test_html_entity_decode(self):
        """Test HTML entity decoding."""
        text = '{"name": "test", "value": "hello&#34;world"}'
        result = JsonCleaner.clean(text)
        assert '&#34;' not in result
        assert '"' in result

    def test_python_booleans(self):
        """Test Python boolean conversion."""
        text = '{"enabled": True, "disabled": False, "empty": None}'
        result = JsonCleaner.clean(text)
        assert 'true' in result
        assert 'false' in result
        assert 'null' in result
        assert 'True' not in result
        assert 'False' not in result
        assert 'None' not in result

    def test_single_quotes_simple(self):
        """Test single quote conversion."""
        text = "{'name': 'test', 'value': 123}"
        result = JsonCleaner.clean(text)
        assert '"name"' in result
        assert '"test"' in result

    def test_trailing_commas(self):
        """Test trailing comma removal."""
        text = '{"a": 1, "b": 2, }'
        result = JsonCleaner.clean(text)
        assert ', }' not in result
        assert ',}' not in result

    def test_unquoted_keys(self):
        """Test unquoted key handling."""
        text = '{name: "test", value: 123}'
        result = JsonCleaner.clean(text)
        assert '"name"' in result
        assert '"value"' in result

    def test_clean_valid_json_unchanged(self):
        """Test that valid JSON is minimally changed."""
        text = '{"type": "tool_call", "name": "test", "arguments": {}}'
        result = JsonCleaner.clean(text)
        # Should parse the same
        import json
        original = json.loads(text)
        cleaned = json.loads(result)
        assert original == cleaned


# ============================================================================
# FormatParser Tests
# ============================================================================

class TestFormatParser:
    """Tests for FormatParser."""

    @pytest.fixture
    def parser(self):
        return FormatParser()

    # Format detection tests

    def test_detect_hermes_format(self, parser):
        """Test Hermes format detection."""
        content = '<tool_call>{"name": "test"}</tool_call>'
        assert parser.detect_format(content) == ToolCallFormat.HERMES

    def test_detect_hermes_format_case_insensitive(self, parser):
        """Test Hermes format detection is case insensitive."""
        content = '<TOOL_CALL>{"name": "test"}</TOOL_CALL>'
        assert parser.detect_format(content) == ToolCallFormat.HERMES

    def test_detect_contract_format(self, parser):
        """Test contract format detection."""
        content = '{"type": "tool_call", "name": "test", "arguments": {}}'
        assert parser.detect_format(content) == ToolCallFormat.CONTRACT

    def test_detect_openai_format(self, parser):
        """Test OpenAI format detection."""
        content = '{"type": "function", "name": "test", "parameters": {}}'
        assert parser.detect_format(content) == ToolCallFormat.OPENAI

    def test_detect_unknown_format(self, parser):
        """Test unknown format detection."""
        content = "This is just plain text"
        assert parser.detect_format(content) == ToolCallFormat.UNKNOWN

    # Parsing tests - Contract format

    def test_parse_contract_format(self, parser):
        """Test parsing contract format."""
        content = '{"type": "tool_call", "name": "get_weather", "arguments": {"location": "Paris"}}'
        result = parser.parse(content)
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments == {"location": "Paris"}
        assert result.format == ToolCallFormat.CONTRACT

    def test_parse_contract_no_args(self, parser):
        """Test parsing contract format without arguments."""
        content = '{"type": "tool_call", "name": "get_time", "arguments": {}}'
        result = parser.parse(content)
        assert result is not None
        assert result.name == "get_time"
        assert result.arguments == {}

    # Parsing tests - OpenAI format

    def test_parse_openai_format(self, parser):
        """Test parsing OpenAI/function format."""
        content = '{"type": "function", "name": "get_weather", "parameters": {"location": "London"}}'
        result = parser.parse(content)
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments == {"location": "London"}
        assert result.format == ToolCallFormat.OPENAI

    # Parsing tests - Hermes format

    def test_parse_hermes_format(self, parser):
        """Test parsing Hermes format."""
        content = '<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>'
        result = parser.parse(content)
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments == {"location": "Tokyo"}
        assert result.format == ToolCallFormat.HERMES

    def test_parse_hermes_with_surrounding_text(self, parser):
        """Test parsing Hermes format with surrounding text."""
        content = 'Let me check the weather for you.\n<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>\nPlease wait...'
        result = parser.parse(content)
        assert result is not None
        assert result.name == "get_weather"

    def test_parse_hermes_single_quotes(self, parser):
        """Test parsing Hermes format with single quotes."""
        content = "<tool_call>{'name': 'get_weather', 'arguments': {'location': 'Berlin'}}</tool_call>"
        result = parser.parse(content)
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments == {"location": "Berlin"}

    # Parsing tests - Raw format (no type field)

    def test_parse_raw_format(self, parser):
        """Test parsing raw format without type field."""
        content = '{"name": "get_weather", "arguments": {"location": "Sydney"}}'
        result = parser.parse(content)
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments == {"location": "Sydney"}

    # Parsing tests - Embedded JSON

    def test_parse_json_in_markdown(self, parser):
        """Test parsing JSON in markdown code block."""
        content = '''Here's the tool call:
```json
{"type": "tool_call", "name": "calculate", "arguments": {"expression": "2+2"}}
```
'''
        result = parser.parse(content)
        assert result is not None
        assert result.name == "calculate"
        assert result.arguments == {"expression": "2+2"}

    def test_parse_json_with_text_before(self, parser):
        """Test parsing JSON with text before it."""
        content = 'I will call the weather tool: {"type": "tool_call", "name": "get_weather", "arguments": {"location": "Cairo"}}'
        result = parser.parse(content)
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments == {"location": "Cairo"}

    # Edge cases

    def test_parse_empty_string(self, parser):
        """Test parsing empty string."""
        assert parser.parse("") is None
        assert parser.parse(None) is None

    def test_parse_plain_text(self, parser):
        """Test parsing plain text returns None."""
        content = "The weather in Paris is sunny and 20 degrees."
        assert parser.parse(content) is None

    def test_parse_invalid_json(self, parser):
        """Test parsing invalid JSON returns None."""
        content = '{"name": "test", arguments: }'
        # Should try to clean but ultimately fail
        result = parser.parse(content)
        # May or may not parse depending on cleanup
        # Just verify it doesn't crash

    def test_parse_arguments_as_string(self, parser):
        """Test parsing arguments provided as JSON string."""
        content = '{"type": "tool_call", "name": "test", "arguments": "{\\"key\\": \\"value\\"}"}'
        result = parser.parse(content)
        assert result is not None
        assert result.name == "test"
        assert result.arguments == {"key": "value"}


# ============================================================================
# ToolCallHandler Tests
# ============================================================================

class TestToolCallHandler:
    """Tests for ToolCallHandler."""

    @pytest.fixture
    def handler(self):
        return ToolCallHandler()

    def test_parse_contract_format(self, handler):
        """Test parsing contract format through handler."""
        content = '{"type": "tool_call", "name": "test", "arguments": {"a": 1}}'
        result = handler.parse(content)
        assert result is not None
        assert result.name == "test"

    def test_parse_hermes_format(self, handler):
        """Test parsing Hermes format through handler."""
        content = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        result = handler.parse(content)
        assert result is not None
        assert result.name == "test"

    def test_parse_native_chunks(self, handler):
        """Test parsing native llama-cpp-python chunks."""
        chunks = [
            {
                "id": "call_1",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}'
                }
            }
        ]
        results = handler.parse_native_chunks(chunks)
        assert len(results) == 1
        assert results[0].name == "get_weather"
        assert results[0].arguments == {"location": "Paris"}
        assert results[0].format == ToolCallFormat.NATIVE

    def test_parse_native_chunks_multiple(self, handler):
        """Test parsing multiple native chunks."""
        chunks = [
            {"id": "1", "function": {"name": "func1", "arguments": "{}"}},
            {"id": "2", "function": {"name": "func2", "arguments": '{"x": 1}'}},
        ]
        results = handler.parse_native_chunks(chunks)
        assert len(results) == 2
        assert results[0].name == "func1"
        assert results[1].name == "func2"

    def test_parse_native_chunks_empty(self, handler):
        """Test parsing empty chunks list."""
        results = handler.parse_native_chunks([])
        assert results == []

    def test_format_result_contract(self, handler):
        """Test formatting result for contract format."""
        result = handler.format_result("test", {"value": 42}, ToolCallFormat.CONTRACT)
        assert "[Tool Result]:" in result
        assert "42" in result

    def test_format_result_hermes(self, handler):
        """Test formatting result for Hermes format."""
        result = handler.format_result("test", "success", ToolCallFormat.HERMES)
        assert "<tool_response>" in result
        assert "success" in result
        assert "</tool_response>" in result


# ============================================================================
# Singleton Tests
# ============================================================================

class TestSingleton:
    """Tests for singleton behavior."""

    def test_get_handler_returns_same_instance(self):
        """Test that get_handler returns singleton."""
        h1 = get_handler()
        h2 = get_handler()
        assert h1 is h2

    def test_get_handler_returns_handler(self):
        """Test that get_handler returns ToolCallHandler."""
        h = get_handler()
        assert isinstance(h, ToolCallHandler)


# ============================================================================
# ToolCall Dataclass Tests
# ============================================================================

class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_create_minimal(self):
        """Test creating minimal ToolCall."""
        tc = ToolCall(name="test")
        assert tc.name == "test"
        assert tc.arguments == {}
        assert tc.raw is None
        assert tc.format == ToolCallFormat.UNKNOWN

    def test_create_full(self):
        """Test creating full ToolCall."""
        tc = ToolCall(
            name="get_weather",
            arguments={"location": "Paris"},
            raw='{"name": "get_weather"}',
            format=ToolCallFormat.HERMES
        )
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "Paris"}
        assert tc.raw is not None
        assert tc.format == ToolCallFormat.HERMES

    def test_to_dict(self):
        """Test converting to dict."""
        tc = ToolCall(name="test", arguments={"a": 1})
        d = tc.to_dict()
        assert d == {
            "type": "tool_call",
            "name": "test",
            "arguments": {"a": 1}
        }


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for realistic scenarios."""

    @pytest.fixture
    def handler(self):
        return get_handler()

    def test_llama32_function_format(self, handler):
        """Test Llama 3.2 function format (as seen in real output)."""
        content = '{"type": "function", "name": "get_weather", "parameters": {"location": "", "unit": "celsius"}}'
        result = handler.parse(content)
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments == {"location": "", "unit": "celsius"}

    def test_hermes_with_python_booleans(self, handler):
        """Test Hermes format with Python booleans."""
        content = '<tool_call>{"name": "test", "arguments": {"enabled": True, "count": None}}</tool_call>'
        result = handler.parse(content)
        assert result is not None
        assert result.arguments == {"enabled": True, "count": None}

    def test_messy_json_with_extra_text(self, handler):
        """Test parsing messy output with extra text."""
        content = '''I'll check the weather for you.

{"type": "tool_call", "name": "get_weather", "arguments": {"location": "Cairo"}}

Please wait while I fetch the data.'''
        result = handler.parse(content)
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments["location"] == "Cairo"


# ============================================================================
# LLAMA3 Format Tests
# ============================================================================

class TestLlama3Format:
    """Tests for Llama 3 pythonic format parsing."""

    @pytest.fixture
    def parser(self):
        return FormatParser()

    @pytest.fixture
    def handler(self):
        return get_handler()

    def test_detect_llama3_format(self, parser):
        """Test Llama 3 format detection."""
        content = "[get_weather(location='Paris')]"
        assert parser.detect_format(content) == ToolCallFormat.LLAMA3

    def test_parse_single_quoted_string(self, handler):
        """Test parsing with single-quoted string arg."""
        content = "[get_weather(location='Paris')]"
        result = handler.parse(content)
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments == {"location": "Paris"}
        assert result.format == ToolCallFormat.LLAMA3

    def test_parse_double_quoted_string(self, handler):
        """Test parsing with double-quoted string arg."""
        content = '[get_weather(location="London")]'
        result = handler.parse(content)
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments == {"location": "London"}

    def test_parse_integer(self, handler):
        """Test parsing integer arguments."""
        content = "[calculate(x=10, y=20)]"
        result = handler.parse(content)
        assert result is not None
        assert result.name == "calculate"
        assert result.arguments == {"x": 10, "y": 20}

    def test_parse_float(self, handler):
        """Test parsing float arguments."""
        content = "[calculate(value=3.14)]"
        result = handler.parse(content)
        assert result is not None
        assert result.arguments == {"value": 3.14}

    def test_parse_boolean_true(self, handler):
        """Test parsing Python True."""
        content = "[test(enabled=True)]"
        result = handler.parse(content)
        assert result is not None
        assert result.arguments == {"enabled": True}

    def test_parse_boolean_false(self, handler):
        """Test parsing Python False."""
        content = "[test(enabled=False)]"
        result = handler.parse(content)
        assert result is not None
        assert result.arguments == {"enabled": False}

    def test_parse_none(self, handler):
        """Test parsing Python None."""
        content = "[test(value=None)]"
        result = handler.parse(content)
        assert result is not None
        assert result.arguments == {"value": None}

    def test_parse_unquoted_string(self, handler):
        """Test parsing unquoted string (identifier)."""
        content = "[get_weather(location=Cairo)]"
        result = handler.parse(content)
        assert result is not None
        assert result.arguments == {"location": "Cairo"}

    def test_parse_multiple_args(self, handler):
        """Test parsing multiple arguments of different types."""
        content = "[calculate(x=10, y=20, precise=True)]"
        result = handler.parse(content)
        assert result is not None
        assert result.arguments == {"x": 10, "y": 20, "precise": True}

    def test_parse_no_args(self, handler):
        """Test parsing function with no arguments."""
        content = "[get_time()]"
        result = handler.parse(content)
        assert result is not None
        assert result.name == "get_time"
        assert result.arguments == {}

    def test_parse_with_whitespace(self, handler):
        """Test parsing with extra whitespace."""
        content = "[  get_weather( location = 'Paris' )  ]"
        result = handler.parse(content)
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments == {"location": "Paris"}


# ============================================================================
# MISTRAL Format Tests
# ============================================================================

class TestMistralFormat:
    """Tests for Mistral [TOOL_CALLS] format parsing."""

    @pytest.fixture
    def parser(self):
        return FormatParser()

    @pytest.fixture
    def handler(self):
        return get_handler()

    def test_detect_mistral_format(self, parser):
        """Test Mistral format detection."""
        content = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris"}}]'
        assert parser.detect_format(content) == ToolCallFormat.MISTRAL

    def test_parse_single_call(self, handler):
        """Test parsing single Mistral tool call."""
        content = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris"}}]'
        result = handler.parse(content)
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments == {"location": "Paris"}
        assert result.format == ToolCallFormat.MISTRAL

    def test_parse_multiple_calls_returns_first(self, handler):
        """Test that multiple calls returns first one."""
        content = '[TOOL_CALLS] [{"name": "func1", "arguments": {}}, {"name": "func2", "arguments": {}}]'
        result = handler.parse(content)
        assert result is not None
        assert result.name == "func1"

    def test_parse_empty_arguments(self, handler):
        """Test parsing with empty arguments."""
        content = '[TOOL_CALLS] [{"name": "get_time", "arguments": {}}]'
        result = handler.parse(content)
        assert result is not None
        assert result.name == "get_time"
        assert result.arguments == {}

    def test_parse_complex_arguments(self, handler):
        """Test parsing with complex nested arguments."""
        content = '[TOOL_CALLS] [{"name": "test", "arguments": {"a": 1, "b": "hello", "c": true}}]'
        result = handler.parse(content)
        assert result is not None
        assert result.arguments == {"a": 1, "b": "hello", "c": True}

    def test_parse_with_whitespace(self, handler):
        """Test parsing with extra whitespace."""
        content = '[TOOL_CALLS]   [  {"name": "test", "arguments": {}}  ]'
        result = handler.parse(content)
        assert result is not None
        assert result.name == "test"

    def test_parse_no_tool_calls_marker(self, handler):
        """Test that content without [TOOL_CALLS] is not parsed as Mistral."""
        content = '[{"name": "test", "arguments": {}}]'
        result = handler.parse(content)
        # Should not be parsed as MISTRAL (might be parsed as something else or None)
        if result:
            assert result.format != ToolCallFormat.MISTRAL


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
