#!/usr/bin/env python3
"""
Comprehensive tests for frontend.py - Chat Engine with tool calling.
"""

import json
import pytest
from io import StringIO
from unittest.mock import patch, MagicMock, Mock

from l3m_backend.core import ToolRegistry, ToolOutput, tool_output
from l3m_backend.engine import ChatEngine, TOOL_CONTRACT_TEMPLATE
from l3m_backend.cli import print_tools, simple_repl as repl


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_registry():
    """Create a mock tool registry with sample tools."""
    registry = ToolRegistry()

    @registry.register(aliases=["calc", "c"])
    @tool_output(llm_format=lambda x: f"Result: {x['result']}")
    def calculate(expression: str) -> dict:
        """Evaluate a math expression."""
        try:
            allowed = set("0123456789+-*/.() ")
            if all(c in allowed for c in expression):
                result = eval(expression)
            else:
                result = "Invalid expression"
        except Exception as e:
            result = f"Error: {e}"
        return {"expression": expression, "result": result}

    @registry.register(aliases=["time", "t"])
    @tool_output(llm_format="{current_time}")
    def get_time() -> dict:
        """Get current time."""
        return {"current_time": "2026-01-06 12:00:00"}

    @registry.register
    def simple_tool() -> str:
        """A simple tool that returns a string."""
        return "simple result"

    return registry


def _make_stream_response(content: str):
    """Create a mock streaming response from content."""
    # Yield the content in small chunks to simulate streaming
    for char in content:
        yield {
            "choices": [{
                "delta": {"content": char},
                "finish_reason": None
            }]
        }
    # Final chunk with finish_reason
    yield {
        "choices": [{
            "delta": {},
            "finish_reason": "stop"
        }]
    }


def _create_chat_completion_handler(return_value):
    """Create a handler that returns streaming or non-streaming response based on kwargs."""
    def handler(*args, **kwargs):
        if kwargs.get("stream"):
            # Extract content from non-streaming format
            content = return_value["choices"][0]["message"]["content"]
            return _make_stream_response(content)
        return return_value
    return handler


@pytest.fixture
def mock_llm():
    """Create a mock Llama instance."""
    mock = MagicMock()
    mock.reset = MagicMock()
    # Set default return values for context management
    mock.n_ctx.return_value = 32768
    mock.tokenize.return_value = list(range(10))  # 10 tokens by default
    return mock


@pytest.fixture
def chat_engine(mock_registry, mock_llm):
    """Create a ChatEngine with mocked Llama."""
    with patch('llama_cpp.Llama', return_value=mock_llm):
        engine = ChatEngine(
            model_path="/fake/model.gguf",
            registry=mock_registry,
            n_ctx=1024,
            n_gpu_layers=0,
            verbose=False,
        )
    return engine


# ============================================================================
# TOOL_CONTRACT_TEMPLATE Tests
# ============================================================================

class TestToolContractTemplate:
    """Tests for the tool contract template."""

    def test_template_has_placeholder(self):
        """Test template has registry_json placeholder."""
        assert "{registry_json}" in TOOL_CONTRACT_TEMPLATE

    def test_template_has_output_formats(self):
        """Test template describes output formats."""
        assert "tool_call" in TOOL_CONTRACT_TEMPLATE
        assert "plain text" in TOOL_CONTRACT_TEMPLATE

    def test_template_has_examples(self):
        """Test template includes examples."""
        assert "EXAMPLES" in TOOL_CONTRACT_TEMPLATE
        assert "get_time" in TOOL_CONTRACT_TEMPLATE
        assert "get_weather" in TOOL_CONTRACT_TEMPLATE

    def test_template_has_tool_result_documentation(self):
        """Test template documents tool_result format and handling."""
        # Must document the tool result format the model will receive
        assert "TOOL RESULTS" in TOOL_CONTRACT_TEMPLATE or "Tool Result" in TOOL_CONTRACT_TEMPLATE
        # Must show complete flow: tool_call -> [Tool Result] -> plain text
        assert "[Tool Result]" in TOOL_CONTRACT_TEMPLATE
        # Must instruct to respond in plain text after receiving tool result
        assert "plain text" in TOOL_CONTRACT_TEMPLATE


# ============================================================================
# ChatEngine Initialization Tests
# ============================================================================

class TestChatEngineInit:
    """Tests for ChatEngine initialization."""

    def test_init_with_defaults(self, mock_registry):
        """Test ChatEngine initialization with defaults."""
        with patch('llama_cpp.Llama') as MockLlama:
            engine = ChatEngine(
                model_path="/fake/model.gguf",
                registry=mock_registry,
            )

            MockLlama.assert_called_once()
            call_kwargs = MockLlama.call_args[1]
            assert call_kwargs["model_path"] == "/fake/model.gguf"
            assert call_kwargs["n_ctx"] == 32768
            assert call_kwargs["n_gpu_layers"] == -1
            assert call_kwargs["verbose"] is False

    def test_init_with_custom_params(self, mock_registry):
        """Test ChatEngine initialization with custom parameters."""
        with patch('llama_cpp.Llama') as MockLlama:
            engine = ChatEngine(
                model_path="/fake/model.gguf",
                registry=mock_registry,
                system_prompt="Custom prompt",
                n_ctx=8192,
                n_gpu_layers=10,
                verbose=True,
            )

            call_kwargs = MockLlama.call_args[1]
            assert call_kwargs["n_ctx"] == 8192
            assert call_kwargs["n_gpu_layers"] == 10
            assert call_kwargs["verbose"] is True
            assert engine.system_prompt == "Custom prompt"

    def test_init_default_system_prompt(self, mock_registry):
        """Test ChatEngine uses default system prompt."""
        with patch('llama_cpp.Llama'):
            engine = ChatEngine(
                model_path="/fake/model.gguf",
                registry=mock_registry,
            )
            assert "helpful assistant" in engine.system_prompt

    def test_init_sets_up_tools(self, chat_engine, mock_registry):
        """Test ChatEngine sets up tools from registry."""
        assert len(chat_engine.tools) == 3
        tool_names = {t["function"]["name"] for t in chat_engine.tools}
        assert "calculate" in tool_names
        assert "get_time" in tool_names

    def test_init_builds_contract(self, chat_engine):
        """Test ChatEngine builds tools contract."""
        assert "calculate" in chat_engine.tools_contract
        assert "get_time" in chat_engine.tools_contract

    def test_init_empty_history(self, chat_engine):
        """Test ChatEngine starts with empty history."""
        assert chat_engine.history == []


# ============================================================================
# ChatEngine _build_system_message Tests
# ============================================================================

class TestBuildSystemMessage:
    """Tests for _build_system_message method."""

    def test_build_system_message_format(self, chat_engine):
        """Test system message has correct format."""
        msg = chat_engine._build_system_message()
        assert msg["role"] == "system"
        assert "content" in msg

    def test_build_system_message_includes_prompt(self, chat_engine):
        """Test system message includes system prompt."""
        msg = chat_engine._build_system_message()
        assert chat_engine.system_prompt in msg["content"]

    def test_build_system_message_includes_contract(self, chat_engine):
        """Test system message includes tools contract."""
        msg = chat_engine._build_system_message()
        assert "tool_call" in msg["content"]
        assert "plain text" in msg["content"]


# ============================================================================
# ChatEngine _build_messages Tests
# ============================================================================

class TestBuildMessages:
    """Tests for _build_messages method."""

    def test_build_messages_system_only(self, chat_engine):
        """Test _build_messages with no history."""
        messages = chat_engine._build_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_build_messages_with_history(self, chat_engine):
        """Test _build_messages includes history."""
        chat_engine.history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        messages = chat_engine._build_messages()
        assert len(messages) == 3
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_build_messages_with_pending(self, chat_engine):
        """Test _build_messages includes pending messages."""
        pending = [{"role": "user", "content": "New message"}]
        messages = chat_engine._build_messages(pending)
        assert len(messages) == 2
        assert messages[1]["content"] == "New message"

    def test_build_messages_order(self, chat_engine):
        """Test _build_messages maintains correct order."""
        chat_engine.history = [
            {"role": "user", "content": "First"},
        ]
        pending = [{"role": "user", "content": "Second"}]
        messages = chat_engine._build_messages(pending)

        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "First"
        assert messages[2]["content"] == "Second"


# ============================================================================
# ChatEngine _execute_tool_by_name Tests
# ============================================================================

class TestExecuteToolByName:
    """Tests for _execute_tool_by_name method."""

    def test_execute_tool_success(self, chat_engine):
        """Test executing a tool successfully."""
        result = chat_engine._execute_tool_by_name("calculate", {"expression": "2 + 2"})
        assert result["name"] == "calculate"
        assert result["arguments"] == {"expression": "2 + 2"}
        assert chat_engine._result2llm(result["output"]) == "Result: 4"

    def test_execute_tool_via_alias(self, chat_engine):
        """Test executing a tool via alias."""
        result = chat_engine._execute_tool_by_name("c", {"expression": "3 * 3"})
        assert result["name"] == "calculate"
        assert chat_engine._result2llm(result["output"]) == "Result: 9"

    def test_execute_tool_no_params(self, chat_engine):
        """Test executing a tool with no parameters."""
        result = chat_engine._execute_tool_by_name("get_time", {})
        assert result["name"] == "get_time"
        assert "2026-01-06" in chat_engine._result2llm(result["output"])

    def test_execute_tool_returns_llm_format(self, chat_engine):
        """Test that tool returns llm_format when available."""
        result = chat_engine._execute_tool_by_name("calculate", {"expression": "10 / 2"})
        assert chat_engine._result2llm(result["output"]) == "Result: 5.0"

    def test_execute_tool_not_found(self, chat_engine):
        """Test executing non-existent tool returns error."""
        result = chat_engine._execute_tool_by_name("nonexistent", {})
        assert "error" in result["output"]

    def test_execute_tool_validation_error(self, chat_engine):
        """Test executing tool with invalid args returns error."""
        # calculate requires 'expression' parameter
        result = chat_engine._execute_tool_by_name("calculate", {})
        assert "error" in result["output"]

    def test_execute_simple_tool_json_output(self, chat_engine):
        """Test tool without ToolOutput returns JSON."""
        result = chat_engine._execute_tool_by_name("simple_tool", {})
        # simple_tool returns a string, so it should be JSON serialized
        assert chat_engine._result2llm(result["output"]) == '"simple result"'


# ============================================================================
# ChatEngine _execute_tool Tests
# ============================================================================

class TestExecuteTool:
    """Tests for _execute_tool method (OpenAI format)."""

    def test_execute_tool_openai_format(self, chat_engine):
        """Test executing tool with OpenAI format."""
        tool_call = {
            "function": {
                "name": "calculate",
                "arguments": {"expression": "5 + 5"}
            }
        }
        result = chat_engine._execute_tool(tool_call)
        assert result["name"] == "calculate"
        assert chat_engine._result2llm(result["output"]) == "Result: 10"

    def test_execute_tool_string_arguments(self, chat_engine):
        """Test executing tool with string arguments (JSON)."""
        tool_call = {
            "function": {
                "name": "calculate",
                "arguments": '{"expression": "7 - 2"}'
            }
        }
        result = chat_engine._execute_tool(tool_call)
        assert result["name"] == "calculate"
        assert chat_engine._result2llm(result["output"]) == "Result: 5"


# ============================================================================
# ChatEngine _parse_tool_call Tests
# ============================================================================

class TestParseContentJson:
    """Tests for _parse_tool_call method."""

    def test_parse_tool_call(self, chat_engine):
        """Test parsing tool_call JSON."""
        content = '{"type": "tool_call", "name": "calc", "arguments": {"expression": "1+1"}}'
        result = chat_engine._parse_tool_call(content)

        assert result is not None
        assert result["type"] == "tool_call"
        assert result["name"] == "calc"

    def test_parse_final_returns_none(self, chat_engine):
        """Test parsing final JSON returns None (only tool_call supported)."""
        content = '{"type": "final", "content": "The answer is 42"}'
        result = chat_engine._parse_tool_call(content)
        # final type is no longer supported, should return None
        assert result is None

    def test_parse_tool_call_with_whitespace(self, chat_engine):
        """Test parsing tool call JSON with whitespace."""
        content = '  {"type": "tool_call", "name": "calc", "arguments": {}}  '
        result = chat_engine._parse_tool_call(content)

        assert result is not None
        assert result["type"] == "tool_call"

    def test_parse_html_entities(self, chat_engine):
        """Test parsing JSON with HTML entities."""
        # Some models output &#34; instead of "
        content = '{&#34;type&#34;: &#34;tool_call&#34;, &#34;name&#34;: &#34;get_time&#34;, &#34;arguments&#34;: {}}'
        result = chat_engine._parse_tool_call(content)

        assert result is not None
        assert result["type"] == "tool_call"
        assert result["name"] == "get_time"

    def test_parse_invalid_json(self, chat_engine):
        """Test parsing invalid JSON returns None."""
        result = chat_engine._parse_tool_call("not valid json")
        assert result is None

    def test_parse_wrong_type(self, chat_engine):
        """Test parsing JSON with wrong type returns None."""
        content = '{"type": "other", "content": "hello"}'
        result = chat_engine._parse_tool_call(content)
        assert result is None

    def test_parse_empty_string(self, chat_engine):
        """Test parsing empty string returns None."""
        result = chat_engine._parse_tool_call("")
        assert result is None

    def test_parse_none(self, chat_engine):
        """Test parsing None returns None."""
        result = chat_engine._parse_tool_call(None)
        assert result is None

    def test_parse_non_dict_json(self, chat_engine):
        """Test parsing non-dict JSON returns None."""
        result = chat_engine._parse_tool_call('["list", "items"]')
        assert result is None

    # ---- JSON extraction from mixed content ----

    def test_parse_text_before_json(self, chat_engine):
        """Test parsing JSON with text before it."""
        content = 'Sure! Here is the response: {"type": "tool_call", "name": "calc", "arguments": {}}'
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert result["type"] == "tool_call"
        assert result["name"] == "calc"

    def test_parse_text_after_json(self, chat_engine):
        """Test parsing JSON with text after it."""
        content = '{"type": "tool_call", "name": "calc", "arguments": {}} Let me know!'
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert result["type"] == "tool_call"
        assert result["name"] == "calc"

    def test_parse_markdown_json_block(self, chat_engine):
        """Test parsing JSON in markdown code block with json tag."""
        content = '```json\n{"type": "tool_call", "name": "get_time", "arguments": {}}\n```'
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert result["type"] == "tool_call"
        assert result["name"] == "get_time"

    def test_parse_markdown_block_no_tag(self, chat_engine):
        """Test parsing tool call JSON in markdown code block without json tag."""
        content = '```\n{"type": "tool_call", "name": "get_time", "arguments": {}}\n```'
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert result["type"] == "tool_call"
        assert result["name"] == "get_time"

    def test_parse_text_surrounding_json(self, chat_engine):
        """Test parsing JSON with text both before and after."""
        content = 'I will call: {"type": "tool_call", "name": "x", "arguments": {}} now.'
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert result["type"] == "tool_call"
        assert result["name"] == "x"

    def test_parse_multiple_json_picks_valid(self, chat_engine):
        """Test parsing picks JSON with valid type when multiple objects exist."""
        content = '{"other": 1} and {"type": "tool_call", "name": "calc", "arguments": {}}'
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert result["type"] == "tool_call"
        assert result["name"] == "calc"

    def test_parse_nested_braces_in_arguments(self, chat_engine):
        """Test parsing JSON with nested braces in arguments."""
        content = '{"type": "tool_call", "name": "calc", "arguments": {"expr": "1+1"}}'
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert result["arguments"]["expr"] == "1+1"

    def test_parse_markdown_with_surrounding_text(self, chat_engine):
        """Test parsing markdown code block with surrounding text."""
        content = 'Here is the result:\n```json\n{"type": "tool_call", "name": "get_time", "arguments": {}}\n```\nLet me know if you need more.'
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert result["name"] == "get_time"

    def test_parse_tool_call_in_markdown(self, chat_engine):
        """Test parsing tool call in markdown block."""
        content = '```json\n{"type": "tool_call", "name": "get_time", "arguments": {}}\n```'
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert result["type"] == "tool_call"
        assert result["name"] == "get_time"

    def test_parse_invalid_json_in_markdown_falls_through(self, chat_engine):
        """Test that invalid JSON in markdown tries other strategies."""
        content = '```json\n{broken json}\n``` but here is valid: {"type": "tool_call", "name": "calc", "arguments": {}}'
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert result["type"] == "tool_call"

    def test_parse_no_valid_json_returns_none(self, chat_engine):
        """Test that content with no valid JSON returns None."""
        content = 'Just some text with no JSON at all'
        result = chat_engine._parse_tool_call(content)
        assert result is None

    def test_parse_json_without_valid_type_returns_none(self, chat_engine):
        """Test that JSON without valid type returns None even after extraction."""
        content = 'Here: {"other_key": "value", "data": 123}'
        result = chat_engine._parse_tool_call(content)
        assert result is None


# ============================================================================
# ChatEngine chat() Tests
# ============================================================================

class TestChat:
    """Tests for chat() method."""

    def test_chat_final_response(self, chat_engine, mock_llm):
        """Test chat with immediate plain text response."""
        mock_llm.create_chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": "Hello there!"
                }
            }]
        }

        response = chat_engine.chat("Hi")

        assert response == "Hello there!"
        assert len(chat_engine.history) == 2  # user + assistant

    def test_chat_plain_text_response(self, chat_engine, mock_llm):
        """Test chat with plain text response (no JSON)."""
        mock_llm.create_chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": "Plain text response"
                }
            }]
        }

        response = chat_engine.chat("Hi")

        assert response == "Plain text response"
        assert len(chat_engine.history) == 2

    def test_chat_tool_call_then_final(self, chat_engine, mock_llm):
        """Test chat with tool call followed by plain text response."""
        mock_llm.create_chat_completion.side_effect = [
            # First call: tool call
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "calculate", "arguments": {"expression": "2+2"}}'
                    }
                }]
            },
            # Second call: plain text response
            {
                "choices": [{
                    "message": {
                        "content": "2+2 equals 4"
                    }
                }]
            },
        ]

        response = chat_engine.chat("What is 2+2?")

        assert response == "2+2 equals 4"
        assert mock_llm.create_chat_completion.call_count == 2
        assert mock_llm.reset.call_count == 2

    def test_chat_multiple_tool_calls(self, chat_engine, mock_llm):
        """Test chat with multiple tool calls."""
        mock_llm.create_chat_completion.side_effect = [
            # First tool call
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "get_time", "arguments": {}}'
                    }
                }]
            },
            # Second tool call
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "calculate", "arguments": {"expression": "5*5"}}'
                    }
                }]
            },
            # Plain text response
            {
                "choices": [{
                    "message": {
                        "content": "Done"
                    }
                }]
            },
        ]

        response = chat_engine.chat("Get time and calculate 5*5")

        assert response == "Done"
        assert mock_llm.create_chat_completion.call_count == 3

    def test_chat_max_rounds(self, chat_engine, mock_llm):
        """Test chat respects max_tool_rounds."""
        # Always return tool call (never final) - but with DIFFERENT args each time
        call_count = [0]

        def return_unique_tool_call(**kwargs):
            call_count[0] += 1
            return {
                "choices": [{
                    "message": {
                        "content": f'{{"type": "tool_call", "name": "calculate", "arguments": {{"expression": "{call_count[0]}+1"}}}}'
                    }
                }]
            }

        mock_llm.create_chat_completion.side_effect = return_unique_tool_call

        response = chat_engine.chat("Loop forever", max_tool_rounds=3)

        assert response == "[Max tool rounds reached]"
        assert mock_llm.create_chat_completion.call_count == 3

    def test_chat_duplicate_tool_call_detected(self, chat_engine, mock_llm):
        """Test chat detects duplicate tool calls and exits to prevent infinite loops."""
        # Model keeps calling the same tool with same arguments (infinite loop scenario)
        mock_llm.create_chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": '{"type": "tool_call", "name": "get_time", "arguments": {}}'
                }
            }]
        }

        response = chat_engine.chat("Loop forever", max_tool_rounds=10)

        # Should exit after detecting duplicate, not after 10 rounds
        assert response == "[Tool already executed - please provide a response]"
        # First call executes the tool, second call is detected as duplicate
        assert mock_llm.create_chat_completion.call_count == 2

    def test_chat_resets_llm(self, chat_engine, mock_llm):
        """Test chat resets LLM before each generation."""
        mock_llm.create_chat_completion.return_value = {
            "choices": [{
                "message": {"content": "response"}
            }]
        }

        chat_engine.chat("Hello")

        mock_llm.reset.assert_called_once()

    def test_chat_history_updated(self, chat_engine, mock_llm):
        """Test chat updates history correctly."""
        mock_llm.create_chat_completion.return_value = {
            "choices": [{
                "message": {"content": "Response"}
            }]
        }

        chat_engine.chat("First message")
        chat_engine.chat("Second message")

        assert len(chat_engine.history) == 4  # 2 user + 2 assistant

    def test_chat_tool_result_format(self, chat_engine, mock_llm):
        """Test that tool results are presented correctly."""
        calls = []

        def capture_calls(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return {
                    "choices": [{
                        "message": {
                            "content": '{"type": "tool_call", "name": "calculate", "arguments": {"expression": "1+1"}}'
                        }
                    }]
                }
            else:
                return {
                    "choices": [{
                        "message": {"content": "Done"}
                    }]
                }

        mock_llm.create_chat_completion.side_effect = capture_calls

        chat_engine.chat("Calculate 1+1")

        # Check second call includes tool result
        second_call_messages = calls[1]["messages"]
        tool_result_msg = second_call_messages[-1]
        assert "[Tool Result]: Result: 2" in tool_result_msg["content"]


# ============================================================================
# ChatEngine clear() Tests
# ============================================================================

class TestClear:
    """Tests for clear() method."""

    def test_clear_empties_history(self, chat_engine, mock_llm):
        """Test clear empties history."""
        mock_llm.create_chat_completion.return_value = {
            "choices": [{
                "message": {"content": "response"}
            }]
        }

        chat_engine.chat("Hello")
        assert len(chat_engine.history) > 0

        chat_engine.clear()
        assert chat_engine.history == []

    def test_clear_allows_new_conversation(self, chat_engine, mock_llm):
        """Test clear allows starting fresh conversation."""
        mock_llm.create_chat_completion.return_value = {
            "choices": [{
                "message": {"content": "Hi"}
            }]
        }

        chat_engine.chat("First convo")
        chat_engine.clear()
        chat_engine.chat("Second convo")

        # Only messages from second conversation
        assert len(chat_engine.history) == 2


# ============================================================================
# ChatEngine messages Property Tests
# ============================================================================

class TestMessagesProperty:
    """Tests for messages property."""

    def test_messages_includes_system(self, chat_engine):
        """Test messages includes system message."""
        messages = chat_engine.messages
        assert len(messages) >= 1
        assert messages[0]["role"] == "system"

    def test_messages_includes_history(self, chat_engine):
        """Test messages includes history."""
        chat_engine.history = [
            {"role": "user", "content": "Test"},
        ]
        messages = chat_engine.messages
        assert len(messages) == 2
        assert messages[1]["content"] == "Test"


# ============================================================================
# print_tools Tests
# ============================================================================

class TestPrintTools:
    """Tests for print_tools function."""

    def test_print_tools_output(self, mock_registry, capsys):
        """Test print_tools produces correct output."""
        print_tools(mock_registry)
        captured = capsys.readouterr()

        assert "calculate" in captured.out
        assert "get_time" in captured.out
        assert "calc" in captured.out  # alias
        assert "time" in captured.out  # alias

    def test_print_tools_shows_descriptions(self, mock_registry, capsys):
        """Test print_tools shows tool descriptions."""
        print_tools(mock_registry)
        captured = capsys.readouterr()

        assert "math expression" in captured.out.lower() or "evaluate" in captured.out.lower()


# ============================================================================
# repl Tests
# ============================================================================

class TestRepl:
    """Tests for repl function."""

    def test_repl_quit_command(self, chat_engine, capsys):
        """Test /quit exits repl."""
        with patch('builtins.input', side_effect=["/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_repl_exit_command(self, chat_engine, capsys):
        """Test /exit exits repl."""
        with patch('builtins.input', side_effect=["/exit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_repl_q_command(self, chat_engine, capsys):
        """Test /q exits repl."""
        with patch('builtins.input', side_effect=["/q"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_repl_clear_command(self, chat_engine, mock_llm, capsys):
        """Test /clear clears history."""
        chat_engine.history = [{"role": "user", "content": "Old"}]

        with patch('builtins.input', side_effect=["/clear", "/quit"]):
            repl(chat_engine)

        assert chat_engine.history == []
        captured = capsys.readouterr()
        assert "cleared" in captured.out.lower()

    def test_repl_tools_command(self, chat_engine, capsys):
        """Test /tools lists tools."""
        with patch('builtins.input', side_effect=["/tools", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "calculate" in captured.out

    def test_repl_system_command(self, chat_engine, capsys):
        """Test /system shows system prompt."""
        with patch('builtins.input', side_effect=["/system", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert chat_engine.system_prompt in captured.out

    def test_repl_schema_command(self, chat_engine, capsys):
        """Test /schema shows tool schema."""
        with patch('builtins.input', side_effect=["/schema", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "function" in captured.out
        assert "calculate" in captured.out

    def test_repl_contract_command(self, chat_engine, capsys):
        """Test /contract shows full system message."""
        with patch('builtins.input', side_effect=["/contract", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "tool_call" in captured.out
        assert "plain text" in captured.out

    def test_repl_history_command(self, chat_engine, capsys):
        """Test /history shows message history."""
        chat_engine.history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        with patch('builtins.input', side_effect=["/history", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "user" in captured.out
        assert "assistant" in captured.out

    def test_repl_unknown_command(self, chat_engine, capsys):
        """Test unknown command shows error."""
        with patch('builtins.input', side_effect=["/unknown", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_repl_empty_input(self, chat_engine, mock_llm, capsys):
        """Test empty input is ignored."""
        with patch('builtins.input', side_effect=["", "/quit"]):
            repl(chat_engine)

        mock_llm.create_chat_completion.assert_not_called()

    def test_repl_chat_response(self, chat_engine, mock_llm, capsys):
        """Test regular chat input gets response."""
        response = {
            "choices": [{
                "message": {"content": "Test response"}
            }]
        }
        mock_llm.create_chat_completion.side_effect = _create_chat_completion_handler(response)

        with patch('builtins.input', side_effect=["Hello", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Test response" in captured.out

    def test_repl_chat_error(self, chat_engine, mock_llm, capsys):
        """Test chat error is handled gracefully."""
        mock_llm.create_chat_completion.side_effect = Exception("Model error")

        with patch('builtins.input', side_effect=["Hello", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_repl_keyboard_interrupt(self, chat_engine, capsys):
        """Test Ctrl+C continues, then /quit exits."""
        with patch('builtins.input', side_effect=[KeyboardInterrupt(), "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_repl_eof(self, chat_engine, capsys):
        """Test EOF requires two presses to exit."""
        # First EOF shows message, second EOF exits
        with patch('builtins.input', side_effect=[EOFError(), EOFError()]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Press Ctrl+D again to exit" in captured.out
        assert "Goodbye" in captured.out

    def test_repl_eof_reset_on_input(self, chat_engine, capsys):
        """Test EOF state is reset when user types input."""
        # First EOF, then user types something, then /quit
        with patch('builtins.input', side_effect=[EOFError(), "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Press Ctrl+D again to exit" in captured.out
        assert "Goodbye" in captured.out

    def test_repl_history_with_tool_calls(self, chat_engine, capsys):
        """Test /history handles messages with tool_calls."""
        chat_engine.history = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "calculate"}}]
            },
        ]

        with patch('builtins.input', side_effect=["/history", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "calculate" in captured.out

    def test_repl_history_with_tool_role(self, chat_engine, capsys):
        """Test /history handles tool role messages."""
        chat_engine.history = [
            {
                "role": "tool",
                "content": "Tool result",
                "tool_call_id": "abc123"
            },
        ]

        with patch('builtins.input', side_effect=["/history", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "tool" in captured.out

    def test_repl_shows_banner(self, chat_engine, capsys):
        """Test repl shows banner on start."""
        with patch('builtins.input', side_effect=["/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "LlamaCpp Chat REPL" in captured.out
        assert "/help" in captured.out
        assert "Tab: completion" in captured.out

    def test_repl_shell_command(self, chat_engine, capsys):
        """Test ! runs shell command."""
        with patch('builtins.input', side_effect=["!echo hello", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_repl_shell_command_stderr(self, chat_engine, capsys):
        """Test ! shows stderr output."""
        with patch('builtins.input', side_effect=["!ls /nonexistent_path_12345", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        # Should show some error about nonexistent path
        assert "No such file" in captured.out or "nonexistent" in captured.out.lower()

    def test_repl_shell_command_empty(self, chat_engine, mock_llm, capsys):
        """Test ! with no command does nothing."""
        with patch('builtins.input', side_effect=["!", "/quit"]):
            repl(chat_engine)

        # Should not call chat
        mock_llm.create_chat_completion.assert_not_called()


# ============================================================================
# main() Function Tests
# ============================================================================

class TestMainFunction:
    """Tests for main() entry point function."""

    def test_main_parses_model_path(self, tmp_path):
        """Test main parses model path argument."""
        # Create a temporary model file
        model_file = tmp_path / "model.gguf"
        model_file.touch()
        test_args = ['frontend.py', str(model_file)]
        with patch('sys.argv', test_args):
            with patch('l3m_backend.engine.ChatEngine') as MockEngine:
                with patch('l3m_backend.cli._repl.repl') as mock_repl:
                    from l3m_backend.cli.chat_repl import main
                    main()

                    MockEngine.assert_called_once()
                    call_kwargs = MockEngine.call_args[1]
                    assert call_kwargs["model_path"] == str(model_file)
                    mock_repl.assert_called_once()

    def test_main_custom_context_size(self, tmp_path):
        """Test main passes custom context size."""
        model_file = tmp_path / "model.gguf"
        model_file.touch()
        test_args = ['frontend.py', str(model_file), '--ctx', '8192']
        with patch('sys.argv', test_args):
            with patch('l3m_backend.engine.ChatEngine') as MockEngine:
                with patch('l3m_backend.cli._repl.repl'):
                    from l3m_backend.cli.chat_repl import main
                    main()

                    call_kwargs = MockEngine.call_args[1]
                    assert call_kwargs["n_ctx"] == 8192

    def test_main_custom_gpu_layers(self, tmp_path):
        """Test main passes custom GPU layers."""
        model_file = tmp_path / "model.gguf"
        model_file.touch()
        test_args = ['frontend.py', str(model_file), '--gpu', '0']
        with patch('sys.argv', test_args):
            with patch('l3m_backend.engine.ChatEngine') as MockEngine:
                with patch('l3m_backend.cli._repl.repl'):
                    from l3m_backend.cli.chat_repl import main
                    main()

                    call_kwargs = MockEngine.call_args[1]
                    assert call_kwargs["n_gpu_layers"] == 0

    def test_main_verbose_mode(self, tmp_path):
        """Test main passes verbose flag."""
        model_file = tmp_path / "model.gguf"
        model_file.touch()
        test_args = ['frontend.py', str(model_file), '-v']
        with patch('sys.argv', test_args):
            with patch('l3m_backend.engine.ChatEngine') as MockEngine:
                with patch('l3m_backend.cli._repl.repl'):
                    from l3m_backend.cli.chat_repl import main
                    main()

                    call_kwargs = MockEngine.call_args[1]
                    assert call_kwargs["verbose"] is True

    def test_main_uses_tools_registry(self, tmp_path):
        """Test main uses the tools registry."""
        model_file = tmp_path / "model.gguf"
        model_file.touch()
        test_args = ['frontend.py', str(model_file)]
        with patch('sys.argv', test_args):
            with patch('l3m_backend.engine.ChatEngine') as MockEngine:
                with patch('l3m_backend.cli._repl.repl'):
                    from l3m_backend.cli.chat_repl import main
                    from l3m_backend.tools import registry
                    main()

                    call_kwargs = MockEngine.call_args[1]
                    assert call_kwargs["registry"] is registry

    def test_main_default_values(self, tmp_path):
        """Test main uses default values when not specified."""
        model_file = tmp_path / "model.gguf"
        model_file.touch()
        test_args = ['frontend.py', str(model_file)]
        with patch('sys.argv', test_args):
            with patch('l3m_backend.engine.ChatEngine') as MockEngine:
                with patch('l3m_backend.cli._repl.repl'):
                    from l3m_backend.cli.chat_repl import main
                    main()

                    call_kwargs = MockEngine.call_args[1]
                    assert call_kwargs["n_ctx"] == 8192  # Default from config
                    assert call_kwargs["n_gpu_layers"] == -1
                    assert call_kwargs["verbose"] is False


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestChatEngineEdgeCases:
    """Additional edge case tests for ChatEngine."""

    def test_chat_with_tool_error_during_execution(self, chat_engine, mock_llm):
        """Test chat handles tool execution errors gracefully."""
        # Setup tool call that will fail
        mock_llm.create_chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "nonexistent_tool", "arguments": {}}'
                    }
                }]
            },
            {
                "choices": [{
                    "message": {
                        "content": "Tool not found, sorry"
                    }
                }]
            },
        ]

        response = chat_engine.chat("Use a bad tool")

        # Should continue and get final response
        assert "sorry" in response.lower() or "Tool not found" in response

    def test_chat_with_empty_tool_result(self, chat_engine, mock_llm):
        """Test chat handles empty tool results."""
        mock_llm.create_chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "simple_tool", "arguments": {}}'
                    }
                }]
            },
            {
                "choices": [{
                    "message": {
                        "content": "Done"
                    }
                }]
            },
        ]

        response = chat_engine.chat("Run simple tool")
        assert response == "Done"

    def test_chat_empty_response(self, chat_engine, mock_llm):
        """Test chat handles empty content."""
        mock_llm.create_chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": ""
                }
            }]
        }

        response = chat_engine.chat("Test")
        assert response == ""

    def test_chat_preserves_tool_chain_in_history(self, chat_engine, mock_llm):
        """Test that tool call chain is preserved in history."""
        mock_llm.create_chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "calculate", "arguments": {"expression": "1+1"}}'
                    }
                }]
            },
            {
                "choices": [{
                    "message": {
                        "content": "Result is 2"
                    }
                }]
            },
        ]

        chat_engine.chat("What is 1+1?")

        # History should contain: user, assistant (tool_call), user (tool result), assistant (final)
        assert len(chat_engine.history) == 4
        assert chat_engine.history[0]["role"] == "user"
        assert chat_engine.history[0]["content"] == "What is 1+1?"

    def test_execute_tool_by_name_with_none_args(self, chat_engine):
        """Test _execute_tool_by_name handles None arguments."""
        # get_time takes no arguments, so empty dict should work
        result = chat_engine._execute_tool_by_name("get_time", {})
        assert "2026-01-06" in chat_engine._result2llm(result["output"])

    def test_parse_tool_call_with_nested_quotes(self, chat_engine):
        """Test parsing JSON with nested quotes in arguments."""
        content = '{"type": "tool_call", "name": "test", "arguments": {"msg": "He said \\"hello\\""}}'
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert result["arguments"]["msg"] == 'He said "hello"'

    def test_parse_tool_call_with_unicode(self, chat_engine):
        """Test parsing JSON with unicode characters."""
        content = '{"type": "tool_call", "name": "test", "arguments": {"temp": "25°C"}}'
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert "25°C" in result["arguments"]["temp"]

    def test_parse_tool_call_with_tool_call_tags(self, chat_engine):
        """Test parsing JSON wrapped in <tool_call> tags with HTML entities."""
        content = '''&lt;tool_call&gt;
{&#34;name&#34;: &#34;get_weather&#34;, &#34;arguments&#34;: {&#34;location&#34;: &#34;Canberra&#34;}}
&lt;/tool_call&gt;'''
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert result["type"] == "tool_call"
        assert result["name"] == "get_weather"
        assert result["arguments"]["location"] == "Canberra"

    def test_parse_tool_call_raw_tool_call_format(self, chat_engine):
        """Test parsing raw tool call JSON without type field."""
        content = '{"name": "get_time", "arguments": {}}'
        result = chat_engine._parse_tool_call(content)
        assert result is not None
        assert result["type"] == "tool_call"
        assert result["name"] == "get_time"


class TestReplEdgeCases:
    """Additional edge case tests for repl."""

    def test_repl_whitespace_only_input(self, chat_engine, mock_llm, capsys):
        """Test whitespace-only input is ignored."""
        with patch('builtins.input', side_effect=["   ", "/quit"]):
            repl(chat_engine)

        mock_llm.create_chat_completion.assert_not_called()

    def test_repl_command_case_insensitive(self, chat_engine, capsys):
        """Test commands are case insensitive."""
        with patch('builtins.input', side_effect=["/QUIT"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_repl_multiline_response(self, chat_engine, mock_llm, capsys):
        """Test repl handles multiline responses."""
        response = {
            "choices": [{
                "message": {"content": "Line 1\nLine 2\nLine 3"}
            }]
        }
        mock_llm.create_chat_completion.side_effect = _create_chat_completion_handler(response)

        with patch('builtins.input', side_effect=["Hello", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Line 1" in captured.out

    def test_repl_empty_history_display(self, chat_engine, capsys):
        """Test /history with empty history."""
        chat_engine.history = []

        with patch('builtins.input', side_effect=["/history", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        # Should show system message at minimum
        assert "system" in captured.out


# ============================================================================
# Integration Tests
# ============================================================================

class TestChatEngineIntegration:
    """Integration tests for ChatEngine."""

    def test_full_conversation_flow(self, chat_engine, mock_llm):
        """Test complete conversation with tool use."""
        mock_llm.create_chat_completion.side_effect = [
            # User asks for calculation
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "calculate", "arguments": {"expression": "10 * 5"}}'
                    }
                }]
            },
            # Model provides answer
            {
                "choices": [{
                    "message": {
                        "content": "10 times 5 equals 50"
                    }
                }]
            },
            # User asks follow-up
            {
                "choices": [{
                    "message": {
                        "content": "Yes, that is correct!"
                    }
                }]
            },
        ]

        # First turn with tool use
        response1 = chat_engine.chat("What is 10 * 5?")
        assert "50" in response1

        # Follow-up
        response2 = chat_engine.chat("Is that right?")
        assert "correct" in response2.lower()

        # History should have 4 entries (2 turns x 2 messages)
        # But tool intermediate messages are also saved
        assert len(chat_engine.history) >= 4

    def test_multi_tool_conversation(self, chat_engine, mock_llm):
        """Test conversation using multiple different tools."""
        mock_llm.create_chat_completion.side_effect = [
            # First: time tool
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "get_time", "arguments": {}}'
                    }
                }]
            },
            # Then: calculate tool
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "calculate", "arguments": {"expression": "2*2"}}'
                    }
                }]
            },
            # Final response
            {
                "choices": [{
                    "message": {
                        "content": "The time is 12:00 and 2*2=4"
                    }
                }]
            },
        ]

        response = chat_engine.chat("What time is it and what is 2*2?")

        assert "12:00" in response
        assert "4" in response

    def test_conversation_clear_and_restart(self, chat_engine, mock_llm):
        """Test clearing history and starting new conversation."""
        mock_llm.create_chat_completion.return_value = {
            "choices": [{
                "message": {"content": "Response"}
            }]
        }

        # First conversation
        chat_engine.chat("Hello")
        assert len(chat_engine.history) == 2

        # Clear
        chat_engine.clear()
        assert len(chat_engine.history) == 0

        # New conversation
        chat_engine.chat("New conversation")
        assert len(chat_engine.history) == 2
        assert "New conversation" in chat_engine.history[0]["content"]

    def test_tool_alias_resolution(self, chat_engine, mock_llm):
        """Test tools can be called by alias."""
        mock_llm.create_chat_completion.side_effect = [
            # Use alias "c" for calculate
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "c", "arguments": {"expression": "3+3"}}'
                    }
                }]
            },
            {
                "choices": [{
                    "message": {
                        "content": "3+3 is 6"
                    }
                }]
            },
        ]

        response = chat_engine.chat("Calculate 3+3")
        assert "6" in response

    def test_llm_reset_on_each_generation(self, chat_engine, mock_llm):
        """Test LLM reset is called before each generation."""
        mock_llm.create_chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "get_time", "arguments": {}}'
                    }
                }]
            },
            {
                "choices": [{
                    "message": {
                        "content": "Done"
                    }
                }]
            },
        ]

        chat_engine.chat("Get time")

        # Reset should be called once per generation (2 times total)
        assert mock_llm.reset.call_count == 2

    def test_error_recovery_in_tool_chain(self, chat_engine, mock_llm):
        """Test recovery when tool in chain fails."""
        mock_llm.create_chat_completion.side_effect = [
            # Call tool that will return error
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "calculate", "arguments": {}}'
                    }
                }]
            },
            # Model should still be able to respond
            {
                "choices": [{
                    "message": {
                        "content": "I encountered an error"
                    }
                }]
            },
        ]

        response = chat_engine.chat("Calculate something")
        assert "error" in response.lower()


class TestEndToEndScenarios:
    """End-to-end scenario tests."""

    def test_time_query_scenario(self, chat_engine, mock_llm):
        """Simulate a user asking for the current time."""
        mock_llm.create_chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "time", "arguments": {}}'
                    }
                }]
            },
            {
                "choices": [{
                    "message": {
                        "content": "The current time is 2026-01-06 12:00:00"
                    }
                }]
            },
        ]

        response = chat_engine.chat("What time is it?")
        assert "2026-01-06" in response

    def test_calculation_scenario(self, chat_engine, mock_llm):
        """Simulate a user asking for a calculation."""
        mock_llm.create_chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "calc", "arguments": {"expression": "15 / 3"}}'
                    }
                }]
            },
            {
                "choices": [{
                    "message": {
                        "content": "15 divided by 3 equals 5"
                    }
                }]
            },
        ]

        response = chat_engine.chat("What is 15 divided by 3?")
        assert "5" in response

    def test_clarification_scenario(self, chat_engine, mock_llm):
        """Simulate model asking for clarification instead of calling tool."""
        mock_llm.create_chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": "Which city would you like the weather for?"
                }
            }]
        }

        response = chat_engine.chat("What's the weather?")
        assert "city" in response.lower() or "weather" in response.lower()

    def test_direct_answer_scenario(self, chat_engine, mock_llm):
        """Simulate model answering without tools."""
        mock_llm.create_chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": "Hello! How can I help you today?"
                }
            }]
        }

        response = chat_engine.chat("Hi!")
        assert "Hello" in response or "help" in response


# ============================================================================
# Magic Commands Tests
# ============================================================================

class TestMagicCommands:
    """Tests for magic command functionality.

    Magic commands all add results to conversation history.
    Pure commands like /undo, /pop, /context, /model are tested in TestRepl.
    """

    def test_magic_shell_appends_to_history(self, chat_engine, capsys):
        """Test %! runs shell and appends user/assistant exchange to history."""
        from l3m_backend.cli._magic import MagicCommands

        magic = MagicCommands(chat_engine)
        magic.execute("!echo hello")

        # Magic commands add user message + assistant response
        assert len(chat_engine.history) == 2
        assert chat_engine.history[0]["role"] == "user"
        assert "%!echo hello" in chat_engine.history[0]["content"]
        assert chat_engine.history[1]["role"] == "assistant"
        assert "hello" in chat_engine.history[1]["content"]

    def test_magic_tool_invokes_tool(self, chat_engine, capsys):
        """Test %tool invokes tool and appends user/assistant exchange."""
        from l3m_backend.cli._magic import MagicCommands

        magic = MagicCommands(chat_engine)
        magic.execute("tool get_time {}")

        # Magic commands add user message + assistant response
        assert len(chat_engine.history) == 2
        assert chat_engine.history[0]["role"] == "user"
        assert "%tool get_time" in chat_engine.history[0]["content"]
        assert chat_engine.history[1]["role"] == "assistant"

    def test_magic_time_adds_to_history(self, chat_engine, capsys):
        """Test %time reports current time and adds user/assistant exchange."""
        from l3m_backend.cli._magic import MagicCommands

        magic = MagicCommands(chat_engine)
        magic.execute("time")

        captured = capsys.readouterr()
        assert "Current time" in captured.out
        # Magic commands add user message + assistant response
        assert len(chat_engine.history) == 2
        assert chat_engine.history[0]["role"] == "user"
        assert "%time" in chat_engine.history[0]["content"]
        assert chat_engine.history[1]["role"] == "assistant"
        assert "Current time" in chat_engine.history[1]["content"]

    def test_magic_execute_unknown_command(self, chat_engine, capsys):
        """Test unknown magic command shows error."""
        from l3m_backend.cli._magic import MagicCommands

        magic = MagicCommands(chat_engine)
        magic.execute("unknown_cmd")

        captured = capsys.readouterr()
        assert "Unknown" in captured.out
        assert "%help" in captured.out

    def test_magic_execute_empty_shows_help(self, chat_engine, capsys):
        """Test empty magic command shows help."""
        from l3m_backend.cli._magic import MagicCommands

        magic = MagicCommands(chat_engine)
        magic.execute("")

        captured = capsys.readouterr()
        assert "Magic Commands" in captured.out

    def test_magic_save_creates_file_and_adds_to_history(self, chat_engine, tmp_path, capsys):
        """Test %save creates file and adds user/assistant exchange to history."""
        from l3m_backend.cli._magic import MagicCommands

        chat_engine.history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        magic = MagicCommands(chat_engine)
        save_path = tmp_path / "conversation.md"
        magic.execute(f"save {save_path}")

        assert save_path.exists()
        content = save_path.read_text()
        assert "USER" in content
        assert "ASSISTANT" in content
        assert "Hello" in content

        # Check that save was added to history as user/assistant exchange
        assert len(chat_engine.history) == 4
        assert chat_engine.history[-2]["role"] == "user"
        assert "%save" in chat_engine.history[-2]["content"]
        assert chat_engine.history[-1]["role"] == "assistant"
        assert "Saved" in chat_engine.history[-1]["content"]

    def test_magic_load_adds_file_to_history(self, chat_engine, tmp_path, capsys):
        """Test %load adds file content as user/assistant exchange to history."""
        from l3m_backend.cli._magic import MagicCommands

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content")

        magic = MagicCommands(chat_engine)
        magic.execute(f"load {test_file}")

        # Magic commands add user message + assistant response
        assert len(chat_engine.history) == 2
        assert chat_engine.history[0]["role"] == "user"
        assert "%load" in chat_engine.history[0]["content"]
        assert chat_engine.history[1]["role"] == "assistant"
        assert "test content" in chat_engine.history[1]["content"]

    def test_magic_load_nonexistent_file(self, chat_engine, capsys):
        """Test %load with nonexistent file shows error."""
        from l3m_backend.cli._magic import MagicCommands

        magic = MagicCommands(chat_engine)
        magic.execute("load /nonexistent/path/file.txt")

        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_magic_tool_invalid_json(self, chat_engine, capsys):
        """Test %tool with invalid JSON shows error."""
        from l3m_backend.cli._magic import MagicCommands

        magic = MagicCommands(chat_engine)
        magic.execute("tool get_time {invalid json}")

        captured = capsys.readouterr()
        assert "Invalid JSON" in captured.out

    def test_magic_tool_nonexistent_tool(self, chat_engine, capsys):
        """Test %tool with nonexistent tool shows error."""
        from l3m_backend.cli._magic import MagicCommands

        magic = MagicCommands(chat_engine)
        magic.execute("tool nonexistent_tool {}")

        captured = capsys.readouterr()
        assert "error" in captured.out.lower()

    def test_magic_edit_response_no_history(self, chat_engine, capsys):
        """Test %edit-response with no history shows error."""
        from l3m_backend.cli._magic import MagicCommands

        chat_engine.history = []
        magic = MagicCommands(chat_engine)
        magic.execute("edit-response")

        captured = capsys.readouterr()
        assert "No conversation history" in captured.out

    def test_magic_edit_response_no_assistant(self, chat_engine, capsys):
        """Test %edit-response with no assistant message shows error."""
        from l3m_backend.cli._magic import MagicCommands

        chat_engine.history = [{"role": "user", "content": "hello"}]
        magic = MagicCommands(chat_engine)
        magic.execute("edit-response")

        captured = capsys.readouterr()
        assert "No assistant response to edit" in captured.out

    def test_magic_edit_response_handles_hyphen(self, chat_engine, capsys):
        """Test execute handles edit-response with hyphen."""
        from l3m_backend.cli._magic import MagicCommands

        chat_engine.history = []
        magic = MagicCommands(chat_engine)
        # This should be handled even though method is magic_edit_response
        magic.execute("edit-response")

        captured = capsys.readouterr()
        assert "No conversation history" in captured.out


class TestReplMagicIntegration:
    """Tests for magic command integration with REPL."""

    def test_repl_magic_command(self, chat_engine, capsys):
        """Test % (empty) in REPL shows magic commands."""
        with patch('builtins.input', side_effect=["%", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Magic Commands" in captured.out

    def test_repl_magic_shell_command(self, chat_engine, capsys):
        """Test %!echo in REPL runs shell and adds user/assistant exchange to history."""
        with patch('builtins.input', side_effect=["%!echo test_magic", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "test_magic" in captured.out
        # Check history was updated (user message + assistant response)
        assert len(chat_engine.history) == 2
        assert chat_engine.history[0]["role"] == "user"
        assert chat_engine.history[1]["role"] == "assistant"

    def test_repl_magic_time(self, chat_engine, capsys):
        """Test %time in REPL adds user/assistant exchange to history."""
        with patch('builtins.input', side_effect=["%time", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Current time" in captured.out
        # Magic commands add user message + assistant response
        assert len(chat_engine.history) == 2
        assert chat_engine.history[0]["role"] == "user"
        assert chat_engine.history[1]["role"] == "assistant"


class TestReplPureCommands:
    """Tests for pure / commands (no history modification)."""

    def test_repl_undo_command(self, chat_engine, capsys):
        """Test /undo removes last exchange."""
        chat_engine.history = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
        with patch('builtins.input', side_effect=["/undo", "/quit"]):
            repl(chat_engine)

        assert len(chat_engine.history) == 0
        captured = capsys.readouterr()
        assert "Removed last exchange" in captured.out

    def test_repl_undo_single_message(self, chat_engine, capsys):
        """Test /undo removes single message if only one exists."""
        chat_engine.history = [
            {"role": "user", "content": "test"},
        ]
        with patch('builtins.input', side_effect=["/undo", "/quit"]):
            repl(chat_engine)

        assert len(chat_engine.history) == 0
        captured = capsys.readouterr()
        assert "Removed last message" in captured.out

    def test_repl_undo_empty_history(self, chat_engine, capsys):
        """Test /undo on empty history shows message."""
        chat_engine.history = []
        with patch('builtins.input', side_effect=["/undo", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    def test_repl_pop_command(self, chat_engine, capsys):
        """Test /pop removes n message pairs."""
        chat_engine.history = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"},
        ]
        with patch('builtins.input', side_effect=["/pop 2", "/quit"]):
            repl(chat_engine)

        assert len(chat_engine.history) == 0
        captured = capsys.readouterr()
        assert "Removed 2 message pair(s)" in captured.out

    def test_repl_pop_default_one(self, chat_engine, capsys):
        """Test /pop without argument removes 1 message pair."""
        chat_engine.history = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
        ]
        with patch('builtins.input', side_effect=["/pop", "/quit"]):
            repl(chat_engine)

        assert len(chat_engine.history) == 0
        captured = capsys.readouterr()
        assert "Removed 1 message pair(s)" in captured.out

    def test_repl_context_command(self, chat_engine, capsys):
        """Test /context shows context estimate."""
        with patch('builtins.input', side_effect=["/context", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Context" in captured.out
        assert "tokens" in captured.out

    def test_repl_model_command(self, chat_engine, capsys):
        """Test /model shows model info."""
        with patch('builtins.input', side_effect=["/model", "/quit"]):
            repl(chat_engine)

        captured = capsys.readouterr()
        assert "Model" in captured.out


# ============================================================================
# Context Management Tests
# ============================================================================

class TestContextManagement:
    """Tests for ChatEngine context management."""

    def test_count_tokens_returns_positive(self, chat_engine):
        """Token count should be positive for non-empty messages."""
        messages = [{"role": "user", "content": "Hello world"}]
        count = chat_engine._count_tokens(messages)
        assert count > 0

    def test_count_tokens_empty_returns_small(self, chat_engine):
        """Token count for empty messages should be small."""
        messages = [{"role": "user", "content": ""}]
        count = chat_engine._count_tokens(messages)
        # "user: \n" is minimal - should be small but positive
        assert count >= 0

    def test_count_tokens_uses_tokenizer(self, chat_engine, mock_llm):
        """Token count should use llm.tokenize when available."""
        mock_llm.tokenize.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        messages = [{"role": "user", "content": "test"}]
        count = chat_engine._count_tokens(messages)
        assert count == 5
        mock_llm.tokenize.assert_called_once()

    def test_count_tokens_fallback_on_error(self, chat_engine, mock_llm):
        """Token count should fallback to char estimate on tokenizer error."""
        mock_llm.tokenize.side_effect = Exception("Tokenizer failed")
        messages = [{"role": "user", "content": "test message here"}]
        count = chat_engine._count_tokens(messages)
        # Fallback is len(text) // 4
        assert count > 0

    def test_trim_history_empty_stays_empty(self, chat_engine):
        """Empty history should stay empty after trim."""
        chat_engine.history = []
        chat_engine._trim_history()
        assert chat_engine.history == []

    def test_trim_history_removes_oldest(self, chat_engine, mock_llm):
        """Oldest history messages should be dropped first."""
        # Setup: small context limit
        mock_llm.n_ctx.return_value = 100
        # Make tokenizer return predictable counts
        mock_llm.tokenize.side_effect = lambda x: list(range(len(x) // 4))

        chat_engine.history = [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "first response"},
            {"role": "user", "content": "second message"},
            {"role": "assistant", "content": "second response"},
        ]

        chat_engine._trim_history(reserve_tokens=50)

        # Should have dropped some messages, newest should remain
        # The exact number depends on token counts
        if chat_engine.history:
            assert chat_engine.history[-1]["content"] == "second response"

    def test_trim_history_keeps_recent(self, chat_engine, mock_llm):
        """Most recent history messages should be preserved when possible."""
        mock_llm.n_ctx.return_value = 10000  # Large context
        mock_llm.tokenize.return_value = [1, 2, 3]  # Small token count

        chat_engine.history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "response"},
        ]

        chat_engine._trim_history()

        # With large context, should keep all messages
        assert len(chat_engine.history) == 2

    def test_trim_history_never_touches_system(self, chat_engine, mock_llm):
        """System message should never be affected by trim (it's not in history)."""
        mock_llm.n_ctx.return_value = 100
        mock_llm.tokenize.return_value = list(range(50))  # 50 tokens each

        original_system = chat_engine.system_prompt
        chat_engine.history = [
            {"role": "user", "content": "message"},
        ]

        chat_engine._trim_history(reserve_tokens=10)

        # System prompt unchanged
        assert chat_engine.system_prompt == original_system

    def test_chat_trims_before_generation(self, chat_engine, mock_llm):
        """chat() should call _trim_history before generating."""
        mock_llm.n_ctx.return_value = 32768
        mock_llm.tokenize.return_value = [1, 2, 3]
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }

        # Pre-fill history
        chat_engine.history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old response"},
        ]

        # Chat should work and history should be managed
        result = chat_engine.chat("new message")
        assert result == "response"


# ============================================================================
# Native Tool Calling Tests
# ============================================================================

class TestNativeToolCalling:
    """Tests for native function calling support."""

    def test_init_passes_chat_format_to_llama(self, mock_registry):
        """Test ChatEngine passes chat_format to Llama constructor."""
        with patch('llama_cpp.Llama') as MockLlama:
            engine = ChatEngine(
                model_path="/fake/model.gguf",
                registry=mock_registry,
                chat_format="chatml-function-calling",
            )

            call_kwargs = MockLlama.call_args[1]
            assert call_kwargs["chat_format"] == "chatml-function-calling"

    def test_init_default_chat_format(self, mock_registry):
        """Test ChatEngine uses None (model default) as default chat format."""
        with patch('llama_cpp.Llama') as MockLlama:
            engine = ChatEngine(
                model_path="/fake/model.gguf",
                registry=mock_registry,
            )

            call_kwargs = MockLlama.call_args[1]
            assert call_kwargs["chat_format"] is None

    def test_init_use_native_tools_default_true(self, mock_registry):
        """Test use_native_tools defaults to True."""
        with patch('llama_cpp.Llama'):
            engine = ChatEngine(
                model_path="/fake/model.gguf",
                registry=mock_registry,
            )
            assert engine.use_native_tools is True

    def test_init_use_native_tools_disabled(self, mock_registry):
        """Test use_native_tools can be disabled."""
        with patch('llama_cpp.Llama'):
            engine = ChatEngine(
                model_path="/fake/model.gguf",
                registry=mock_registry,
                use_native_tools=False,
            )
            assert engine.use_native_tools is False

    def test_chat_passes_tools_when_native_enabled(self, chat_engine, mock_llm):
        """Test tools are passed to create_chat_completion when native enabled."""
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello"}}]
        }

        chat_engine.chat("Hi")

        call_kwargs = mock_llm.create_chat_completion.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"

    def test_chat_no_tools_when_native_disabled(self, mock_registry, mock_llm):
        """Test tools not passed when use_native_tools=False."""
        with patch('llama_cpp.Llama', return_value=mock_llm):
            engine = ChatEngine(
                model_path="/fake/model.gguf",
                registry=mock_registry,
                use_native_tools=False,
            )

        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello"}}]
        }

        engine.chat("Hi")

        call_kwargs = mock_llm.create_chat_completion.call_args[1]
        assert "tools" not in call_kwargs

    def test_chat_handles_native_tool_calls(self, chat_engine, mock_llm):
        """Test chat handles native tool_calls from LLM response."""
        mock_llm.create_chat_completion.side_effect = [
            # First response: native tool call
            {
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [{
                            "id": "call_123",
                            "function": {
                                "name": "get_time",
                                "arguments": "{}"
                            }
                        }]
                    }
                }]
            },
            # Second response: final answer
            {
                "choices": [{
                    "message": {
                        "content": "The time is 2026-01-06 12:00:00"
                    }
                }]
            },
        ]

        response = chat_engine.chat("What time is it?")

        assert "2026-01-06" in response
        # Should have been called twice (tool call + final)
        assert mock_llm.create_chat_completion.call_count == 2

    def test_chat_handles_multiple_native_tool_calls(self, chat_engine, mock_llm):
        """Test chat handles multiple native tool_calls in one response."""
        mock_llm.create_chat_completion.side_effect = [
            # First response: multiple native tool calls
            {
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "get_time",
                                    "arguments": "{}"
                                }
                            },
                            {
                                "id": "call_2",
                                "function": {
                                    "name": "calculate",
                                    "arguments": '{"expression": "2+2"}'
                                }
                            }
                        ]
                    }
                }]
            },
            # Second response: final answer
            {
                "choices": [{
                    "message": {
                        "content": "Time is 12:00 and 2+2=4"
                    }
                }]
            },
        ]

        response = chat_engine.chat("Time and math?")

        assert "12:00" in response or "4" in response

    def test_chat_native_tool_result_uses_tool_role(self, chat_engine, mock_llm):
        """Test native tool results use 'tool' role with tool_call_id."""
        mock_llm.create_chat_completion.side_effect = [
            # First: native tool call
            {
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [{
                            "id": "call_abc123",
                            "function": {
                                "name": "get_time",
                                "arguments": "{}"
                            }
                        }]
                    }
                }]
            },
            # Second: final answer
            {
                "choices": [{
                    "message": {"content": "Done"}
                }]
            },
        ]

        chat_engine.chat("Time?")

        # Check that history contains tool role message
        tool_msgs = [m for m in chat_engine.history if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_abc123"

    def test_chat_fallback_to_contract_based(self, chat_engine, mock_llm):
        """Test fallback to contract-based when no native tool_calls."""
        # Response without tool_calls but with contract-based JSON
        mock_llm.create_chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "content": '{"type": "tool_call", "name": "get_time", "arguments": {}}'
                    }
                }]
            },
            {
                "choices": [{
                    "message": {
                        "content": "The time is 12:00"
                    }
                }]
            },
        ]

        response = chat_engine.chat("What time is it?")

        assert "12:00" in response

    def test_native_tool_call_with_string_arguments(self, chat_engine, mock_llm):
        """Test native tool call handles string arguments (needs JSON parsing)."""
        mock_llm.create_chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "function": {
                                "name": "calculate",
                                "arguments": '{"expression": "5*5"}'  # String JSON
                            }
                        }]
                    }
                }]
            },
            {
                "choices": [{
                    "message": {"content": "5*5 = 25"}
                }]
            },
        ]

        response = chat_engine.chat("Calculate 5*5")

        assert "25" in response


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
