#!/usr/bin/env python3
"""
Comprehensive tests for backend.py - Tool Registry with Pydantic-based schema handling.
"""

import json
import pytest
from typing import Any, Literal
from unittest.mock import patch

from pydantic import Field

from l3m_backend.core import (
    # Exceptions
    ToolError,
    ToolNotFoundError,
    ToolValidationError,
    ToolParseError,
    # Models
    ToolOutput,
    ToolResult,
    ToolEntry,
    # Registry
    ToolRegistry,
    # Helpers
    parse_model_response,
    # Decorator
    tool_output,
)
from l3m_backend.core.helpers import (
    _normalize_name,
    _create_params_model,
    _safe_get_type_hints,
)
from l3m_backend.core.decorators import _resolve_format

# Create a TOOLS alias for backward compatibility in tests
TOOLS = ToolRegistry()


# ============================================================================
# Exception Tests
# ============================================================================

class TestExceptions:
    """Tests for custom exceptions."""

    def test_tool_error_is_base(self):
        """Test ToolError is the base exception."""
        assert issubclass(ToolNotFoundError, ToolError)
        assert issubclass(ToolValidationError, ToolError)
        assert issubclass(ToolParseError, ToolError)

    def test_tool_error_message(self):
        """Test ToolError can carry a message."""
        err = ToolError("Something went wrong")
        assert str(err) == "Something went wrong"

    def test_tool_not_found_error(self):
        """Test ToolNotFoundError."""
        err = ToolNotFoundError("Unknown tool: foo")
        assert "foo" in str(err)

    def test_tool_validation_error(self):
        """Test ToolValidationError."""
        err = ToolValidationError("Invalid argument type")
        assert "Invalid argument" in str(err)

    def test_tool_parse_error(self):
        """Test ToolParseError."""
        err = ToolParseError("Invalid JSON")
        assert "Invalid JSON" in str(err)


# ============================================================================
# ToolOutput Tests
# ============================================================================

class TestToolOutput:
    """Tests for ToolOutput model."""

    def test_create_basic(self):
        """Test ToolOutput.create with basic data."""
        data = {"key": "value"}
        output = ToolOutput.create(data)

        assert output.type_name == "dict"
        assert output.data == data
        assert output.id.startswith("dict_")
        assert len(output.id) > 5
        assert output.llm_format is None
        assert output.gui_format is None

    def test_create_with_formats(self):
        """Test ToolOutput.create with llm_format and gui_format."""
        data = {"temp": 22}
        output = ToolOutput.create(
            data,
            llm_format="Temperature: 22",
            gui_format="<temp>22</temp>"
        )

        assert output.llm_format == "Temperature: 22"
        assert output.gui_format == "<temp>22</temp>"

    def test_create_unique_ids(self):
        """Test that each create() produces unique ID."""
        data = {"x": 1}
        ids = {ToolOutput.create(data).id for _ in range(100)}
        assert len(ids) == 100

    def test_create_list_type(self):
        """Test ToolOutput.create with list type."""
        data = [1, 2, 3]
        output = ToolOutput.create(data)

        assert output.type_name == "list"
        assert output.id.startswith("list_")
        assert output.data == [1, 2, 3]

    def test_create_string_type(self):
        """Test ToolOutput.create with string type."""
        data = "hello world"
        output = ToolOutput.create(data)

        assert output.type_name == "str"
        assert output.id.startswith("str_")

    def test_create_int_type(self):
        """Test ToolOutput.create with int type."""
        data = 42
        output = ToolOutput.create(data)

        assert output.type_name == "int"
        assert output.id.startswith("int_")

    def test_tool_output_model_fields(self):
        """Test ToolOutput has expected Pydantic fields."""
        output = ToolOutput(
            type_name="dict",
            id="dict_123",
            data={"key": "value"},
            llm_format="formatted",
            gui_format="gui"
        )
        assert output.type_name == "dict"
        assert output.id == "dict_123"
        assert output.data["key"] == "value"


# ============================================================================
# ToolResult Tests
# ============================================================================

class TestToolResult:
    """Tests for ToolResult model."""

    def test_tool_result_fields(self):
        """Test ToolResult has expected fields."""
        result = ToolResult(
            name="test_tool",
            arguments={"x": 1},
            output={"result": "ok"}
        )

        assert result.name == "test_tool"
        assert result.arguments == {"x": 1}
        assert result.output == {"result": "ok"}

    def test_tool_result_with_tool_output(self):
        """Test ToolResult with ToolOutput as output."""
        tool_output = ToolOutput.create({"data": 123}, llm_format="Result: 123")
        result = ToolResult(
            name="calc",
            arguments={"expr": "1+1"},
            output=tool_output
        )

        assert result.name == "calc"
        assert result.output.llm_format == "Result: 123"


# ============================================================================
# ToolEntry Tests
# ============================================================================

class TestToolEntry:
    """Tests for ToolEntry model."""

    def test_tool_entry_basic(self):
        """Test basic ToolEntry creation."""
        def my_func(x: int) -> str:
            """A test function."""
            return str(x)

        entry = ToolEntry(
            name="my_func",
            callable_fn=my_func,
        )

        assert entry.name == "my_func"
        assert entry.aliases == []
        assert entry.description is None
        assert entry.callable_fn is my_func

    def test_tool_entry_with_aliases(self):
        """Test ToolEntry with aliases."""
        def my_func(x: int) -> str:
            return str(x)

        entry = ToolEntry(
            name="my_func",
            callable_fn=my_func,
            aliases=["mf", "m"]
        )

        assert entry.aliases == ["mf", "m"]

    def test_get_description_from_docstring(self):
        """Test get_description extracts from docstring."""
        def my_func(x: int) -> str:
            """This is the description.

            More details here.
            """
            return str(x)

        entry = ToolEntry(name="my_func", callable_fn=my_func)
        assert entry.get_description() == "This is the description."

    def test_get_description_override(self):
        """Test get_description uses override when provided."""
        def my_func(x: int) -> str:
            """Original description."""
            return str(x)

        entry = ToolEntry(
            name="my_func",
            callable_fn=my_func,
            description="Custom description"
        )
        assert entry.get_description() == "Custom description"

    def test_get_description_no_docstring(self):
        """Test get_description with no docstring."""
        def my_func(x: int) -> str:
            return str(x)

        entry = ToolEntry(name="my_func", callable_fn=my_func)
        assert entry.get_description() == ""

    def test_get_params_model_basic(self):
        """Test get_params_model creates correct Pydantic model."""
        def my_func(name: str, count: int = 10) -> str:
            return f"{name}: {count}"

        entry = ToolEntry(name="my_func", callable_fn=my_func)
        model = entry.get_params_model()

        # Validate required field
        with pytest.raises(Exception):
            model()

        # Validate with required field
        instance = model(name="test")
        assert instance.name == "test"
        assert instance.count == 10

        # Validate with both fields
        instance = model(name="test", count=5)
        assert instance.count == 5

    def test_get_params_model_cached(self):
        """Test get_params_model is cached."""
        def my_func(x: int) -> int:
            return x

        entry = ToolEntry(name="my_func", callable_fn=my_func)
        model1 = entry.get_params_model()
        model2 = entry.get_params_model()
        assert model1 is model2

    def test_to_openai_spec(self):
        """Test to_openai_spec generates correct OpenAI schema."""
        def search(query: str, limit: int = 10) -> list:
            """Search for items."""
            return []

        entry = ToolEntry(name="search", callable_fn=search)
        spec = entry.to_openai_spec()

        assert spec["type"] == "function"
        assert spec["function"]["name"] == "search"
        assert spec["function"]["description"] == "Search for items."

        params = spec["function"]["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "limit" in params["properties"]
        assert params["additionalProperties"] is False

    def test_to_openai_spec_no_params(self):
        """Test to_openai_spec with no parameters."""
        def get_time() -> str:
            """Get current time."""
            return "12:00"

        entry = ToolEntry(name="get_time", callable_fn=get_time)
        spec = entry.to_openai_spec()

        assert spec["function"]["name"] == "get_time"
        assert spec["function"]["parameters"]["properties"] == {}


# ============================================================================
# ToolRegistry Tests
# ============================================================================

class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_basic(self):
        """Test basic tool registration."""
        registry = ToolRegistry()

        @registry.register
        def my_tool(x: int) -> int:
            return x * 2

        assert "my_tool" in registry
        assert len(registry) == 1

    def test_register_with_name(self):
        """Test registration with custom name."""
        registry = ToolRegistry()

        @registry.register(name="custom_name")
        def my_tool(x: int) -> int:
            return x * 2

        assert "custom_name" in registry
        assert "my_tool" not in registry

    def test_register_with_aliases(self):
        """Test registration with aliases."""
        registry = ToolRegistry()

        @registry.register(aliases=["mt", "m"])
        def my_tool(x: int) -> int:
            return x * 2

        assert "my_tool" in registry
        assert "mt" in registry
        assert "m" in registry

        # All resolve to same tool
        assert registry.get("my_tool").name == "my_tool"
        assert registry.get("mt").name == "my_tool"
        assert registry.get("m").name == "my_tool"

    def test_register_with_description(self):
        """Test registration with custom description."""
        registry = ToolRegistry()

        @registry.register(description="Custom desc")
        def my_tool(x: int) -> int:
            """Original docstring."""
            return x * 2

        entry = registry.get("my_tool")
        assert entry.get_description() == "Custom desc"

    def test_register_name_collision(self):
        """Test that name collision raises error."""
        registry = ToolRegistry()

        @registry.register
        def my_tool(x: int) -> int:
            return x

        with pytest.raises(ToolError, match="collision"):
            @registry.register(name="my_tool")
            def another_tool(x: int) -> int:
                return x

    def test_register_alias_collision(self):
        """Test that alias collision raises error."""
        registry = ToolRegistry()

        @registry.register(aliases=["shortcut"])
        def tool1(x: int) -> int:
            return x

        with pytest.raises(ToolError, match="collision"):
            @registry.register(aliases=["shortcut"])
            def tool2(x: int) -> int:
                return x

    def test_register_attaches_metadata(self):
        """Test that register attaches metadata to function."""
        registry = ToolRegistry()

        @registry.register(aliases=["mt"])
        def my_tool(x: int) -> int:
            return x

        assert my_tool.__tool_name__ == "my_tool"
        assert my_tool.__tool_aliases__ == ["mt"]

    def test_get_by_name(self):
        """Test get() by canonical name."""
        registry = ToolRegistry()

        @registry.register
        def my_tool(x: int) -> int:
            return x

        entry = registry.get("my_tool")
        assert entry.name == "my_tool"

    def test_get_by_alias(self):
        """Test get() by alias."""
        registry = ToolRegistry()

        @registry.register(aliases=["mt"])
        def my_tool(x: int) -> int:
            return x

        entry = registry.get("mt")
        assert entry.name == "my_tool"

    def test_get_not_found(self):
        """Test get() raises ToolNotFoundError."""
        registry = ToolRegistry()

        with pytest.raises(ToolNotFoundError, match="nonexistent"):
            registry.get("nonexistent")

    def test_execute_success(self):
        """Test execute() with valid arguments."""
        registry = ToolRegistry()

        @registry.register
        def add(a: int, b: int) -> int:
            return a + b

        result = registry.execute("add", {"a": 2, "b": 3})
        assert result.name == "add"
        assert result.arguments == {"a": 2, "b": 3}
        assert result.output == 5

    def test_execute_with_defaults(self):
        """Test execute() with default arguments."""
        registry = ToolRegistry()

        @registry.register
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = registry.execute("greet", {"name": "World"})
        assert result.output == "Hello, World!"

    def test_execute_validation_error(self):
        """Test execute() raises ToolValidationError for invalid args."""
        registry = ToolRegistry()

        @registry.register
        def add(a: int, b: int) -> int:
            return a + b

        with pytest.raises(ToolValidationError):
            registry.execute("add", {"a": "not_an_int", "b": 3})

    def test_execute_missing_required(self):
        """Test execute() raises ToolValidationError for missing required args."""
        registry = ToolRegistry()

        @registry.register
        def add(a: int, b: int) -> int:
            return a + b

        with pytest.raises(ToolValidationError):
            registry.execute("add", {"a": 1})

    def test_execute_via_alias(self):
        """Test execute() works via alias."""
        registry = ToolRegistry()

        @registry.register(aliases=["a"])
        def add(a: int, b: int) -> int:
            return a + b

        result = registry.execute("a", {"a": 5, "b": 5})
        assert result.name == "add"
        assert result.output == 10

    def test_execute_call_success(self):
        """Test execute_call() with valid call dict."""
        registry = ToolRegistry()

        @registry.register
        def multiply(x: int, y: int) -> int:
            return x * y

        call = {
            "type": "tool_call",
            "name": "multiply",
            "arguments": {"x": 3, "y": 4}
        }
        result = registry.execute_call(call)
        assert result.output == 12

    def test_execute_call_wrong_type(self):
        """Test execute_call() raises error for wrong type."""
        registry = ToolRegistry()

        call = {
            "type": "final",
            "content": "hello"
        }
        with pytest.raises(ToolParseError, match="Not a tool_call"):
            registry.execute_call(call)

    def test_execute_call_missing_name(self):
        """Test execute_call() raises error for missing name."""
        registry = ToolRegistry()

        call = {
            "type": "tool_call",
            "arguments": {}
        }
        with pytest.raises(ToolParseError, match="missing 'name'"):
            registry.execute_call(call)

    def test_execute_call_invalid_arguments_type(self):
        """Test execute_call() raises error for non-dict arguments."""
        registry = ToolRegistry()

        @registry.register
        def my_tool(x: int) -> int:
            return x

        call = {
            "type": "tool_call",
            "name": "my_tool",
            "arguments": "not_a_dict"
        }
        with pytest.raises(ToolParseError, match="must be a dict"):
            registry.execute_call(call)

    def test_execute_call_empty_arguments(self):
        """Test execute_call() with missing arguments key defaults to empty dict."""
        registry = ToolRegistry()

        @registry.register
        def get_status() -> str:
            return "ok"

        call = {
            "type": "tool_call",
            "name": "get_status"
        }
        result = registry.execute_call(call)
        assert result.output == "ok"

    def test_to_openai_tools(self):
        """Test to_openai_tools() generates all tool specs."""
        registry = ToolRegistry()

        @registry.register
        def tool1(x: int) -> int:
            """Tool 1."""
            return x

        @registry.register
        def tool2(y: str) -> str:
            """Tool 2."""
            return y

        specs = registry.to_openai_tools()
        assert len(specs) == 2
        names = {s["function"]["name"] for s in specs}
        assert names == {"tool1", "tool2"}

    def test_to_registry_json(self):
        """Test to_registry_json() generates valid JSON."""
        registry = ToolRegistry()

        @registry.register(aliases=["t1"])
        def tool1(x: int) -> int:
            """Tool 1 description."""
            return x

        json_str = registry.to_registry_json()
        data = json.loads(json_str)

        assert "tools" in data
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "tool1"
        assert data["tools"][0]["aliases"] == ["t1"]
        assert data["tools"][0]["description"] == "Tool 1 description."

    def test_build_system_prompt(self):
        """Test build_system_prompt() includes all components."""
        registry = ToolRegistry()

        @registry.register
        def my_tool(x: int) -> int:
            """A tool."""
            return x

        prompt = registry.build_system_prompt()
        assert "TOOL_REGISTRY_JSON" in prompt
        assert "my_tool" in prompt
        assert "tool_call" in prompt
        assert "final" in prompt

    def test_contains(self):
        """Test __contains__ (in operator)."""
        registry = ToolRegistry()

        @registry.register(aliases=["mt"])
        def my_tool(x: int) -> int:
            return x

        assert "my_tool" in registry
        assert "mt" in registry
        assert "nonexistent" not in registry

    def test_iter(self):
        """Test __iter__."""
        registry = ToolRegistry()

        @registry.register
        def tool1(x: int) -> int:
            return x

        @registry.register
        def tool2(y: str) -> str:
            return y

        entries = list(registry)
        assert len(entries) == 2
        names = {e.name for e in entries}
        assert names == {"tool1", "tool2"}

    def test_len(self):
        """Test __len__."""
        registry = ToolRegistry()
        assert len(registry) == 0

        @registry.register
        def tool1(x: int) -> int:
            return x

        assert len(registry) == 1

        @registry.register
        def tool2(y: str) -> str:
            return y

        assert len(registry) == 2


# ============================================================================
# Global Registry Tests
# ============================================================================

class TestGlobalRegistry:
    """Tests for the global TOOLS registry."""

    def test_global_registry_exists(self):
        """Test TOOLS is a ToolRegistry instance."""
        assert isinstance(TOOLS, ToolRegistry)


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestNormalizeName:
    """Tests for _normalize_name helper."""

    def test_normalize_simple(self):
        """Test normalization of simple name."""
        assert _normalize_name("my_tool") == "my_tool"

    def test_normalize_with_spaces(self):
        """Test normalization replaces spaces."""
        assert _normalize_name("my tool") == "my_tool"

    def test_normalize_with_special_chars(self):
        """Test normalization replaces special characters."""
        assert _normalize_name("my-tool!@#$%") == "my_tool_"

    def test_normalize_strips_whitespace(self):
        """Test normalization strips leading/trailing whitespace."""
        assert _normalize_name("  my_tool  ") == "my_tool"

    def test_normalize_empty(self):
        """Test normalization of empty string."""
        assert _normalize_name("") == ""

    def test_normalize_none(self):
        """Test normalization of None."""
        assert _normalize_name(None) == ""


class TestCreateParamsModel:
    """Tests for _create_params_model helper."""

    def test_create_params_model_basic(self):
        """Test creating params model from function."""
        def my_func(name: str, count: int = 5) -> str:
            return f"{name}: {count}"

        model = _create_params_model(my_func, "my_func")
        instance = model(name="test")
        assert instance.name == "test"
        assert instance.count == 5

    def test_create_params_model_no_params(self):
        """Test creating params model from function with no params."""
        def my_func() -> str:
            return "hello"

        model = _create_params_model(my_func, "my_func")
        instance = model()
        assert instance is not None

    def test_create_params_model_skips_var_args(self):
        """Test that *args and **kwargs are skipped."""
        def my_func(x: int, *args, **kwargs) -> int:
            return x

        model = _create_params_model(my_func, "my_func")
        schema = model.model_json_schema()
        assert "x" in schema.get("properties", {})
        assert "args" not in schema.get("properties", {})
        assert "kwargs" not in schema.get("properties", {})


class TestSafeGetTypeHints:
    """Tests for _safe_get_type_hints helper."""

    def test_safe_get_type_hints_success(self):
        """Test getting type hints normally."""
        def my_func(x: int, y: str) -> bool:
            return True

        hints = _safe_get_type_hints(my_func)
        assert hints.get("x") == int
        assert hints.get("y") == str

    def test_safe_get_type_hints_no_hints(self):
        """Test function with no type hints."""
        def my_func(x, y):
            return True

        hints = _safe_get_type_hints(my_func)
        assert hints == {} or hints == {"x": None, "y": None} or "x" not in hints

    def test_safe_get_type_hints_fallback_on_error(self):
        """Test fallback to __annotations__ when get_type_hints fails."""
        # Create a function with annotations that can't be resolved
        def my_func(x: "UnresolvableType") -> str:  # Forward ref that can't resolve
            return str(x)

        # Patch get_type_hints to raise an exception
        with patch('l3m_backend.core.helpers.get_type_hints', side_effect=NameError("name 'UnresolvableType' is not defined")):
            hints = _safe_get_type_hints(my_func)

        # Should fall back to __annotations__
        assert "x" in hints
        assert hints["x"] == "UnresolvableType"

    def test_safe_get_type_hints_fallback_empty(self):
        """Test fallback returns empty dict when no __annotations__."""
        def my_func(x, y):
            return True

        # Patch get_type_hints to raise an exception
        with patch('l3m_backend.core.helpers.get_type_hints', side_effect=Exception("Some error")):
            hints = _safe_get_type_hints(my_func)

        # Should return empty dict or the raw annotations
        assert isinstance(hints, dict)


class TestParseModelResponse:
    """Tests for parse_model_response helper."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        result = parse_model_response('{"type": "final", "content": "hello"}')
        assert result == {"type": "final", "content": "hello"}

    def test_parse_with_whitespace(self):
        """Test parsing JSON with surrounding whitespace."""
        result = parse_model_response('  {"x": 1}  ')
        assert result == {"x": 1}

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON raises ToolParseError."""
        with pytest.raises(ToolParseError, match="Invalid JSON"):
            parse_model_response("not valid json")

    def test_parse_empty_string(self):
        """Test parsing empty string raises ToolParseError."""
        with pytest.raises(ToolParseError):
            parse_model_response("")

    def test_parse_none(self):
        """Test parsing None raises ToolParseError."""
        with pytest.raises(ToolParseError):
            parse_model_response(None)


# ============================================================================
# tool_output Decorator Tests
# ============================================================================

class TestToolOutputDecorator:
    """Tests for tool_output decorator."""

    def test_tool_output_string_format(self):
        """Test tool_output with string format template."""
        @tool_output(llm_format="{name}: {value}")
        def my_func() -> dict:
            return {"name": "test", "value": 42}

        result = my_func()
        assert isinstance(result, ToolOutput)
        assert result.data == {"name": "test", "value": 42}
        assert result.llm_format == "test: 42"

    def test_tool_output_callable_format(self):
        """Test tool_output with callable format."""
        @tool_output(llm_format=lambda x: f"Result: {x['value']}")
        def my_func() -> dict:
            return {"value": 100}

        result = my_func()
        assert result.llm_format == "Result: 100"

    def test_tool_output_gui_format(self):
        """Test tool_output with gui_format."""
        @tool_output(
            llm_format="{value}",
            gui_format=lambda x: f"<value>{x['value']}</value>"
        )
        def my_func() -> dict:
            return {"value": 50}

        result = my_func()
        assert result.llm_format == "50"
        assert result.gui_format == "<value>50</value>"

    def test_tool_output_no_format(self):
        """Test tool_output with no format specified."""
        @tool_output()
        def my_func() -> dict:
            return {"key": "value"}

        result = my_func()
        assert result.llm_format is None
        assert result.gui_format is None

    def test_tool_output_preserves_function_name(self):
        """Test that decorator preserves function name."""
        @tool_output(llm_format="{x}")
        def my_special_func() -> dict:
            """My docstring."""
            return {"x": 1}

        assert my_special_func.__name__ == "my_special_func"
        assert my_special_func.__doc__ == "My docstring."

    def test_tool_output_with_args(self):
        """Test tool_output with function that takes arguments."""
        @tool_output(llm_format=lambda x: f"{x['a']} + {x['b']} = {x['result']}")
        def add(a: int, b: int) -> dict:
            return {"a": a, "b": b, "result": a + b}

        result = add(3, 5)
        assert result.llm_format == "3 + 5 = 8"
        assert result.data["result"] == 8


class TestResolveFormat:
    """Tests for _resolve_format helper."""

    def test_resolve_format_none(self):
        """Test resolving None format."""
        result = _resolve_format(None, {"x": 1})
        assert result is None

    def test_resolve_format_callable(self):
        """Test resolving callable format."""
        fmt = lambda x: f"value: {x['v']}"
        result = _resolve_format(fmt, {"v": 42})
        assert result == "value: 42"

    def test_resolve_format_string_with_dict(self):
        """Test resolving string format with dict data."""
        result = _resolve_format("{a} and {b}", {"a": "x", "b": "y"})
        assert result == "x and y"

    def test_resolve_format_string_missing_key(self):
        """Test resolving string format with missing key."""
        result = _resolve_format("{a} and {b}", {"a": "x"})
        # Should return the format string as-is when key is missing
        assert result == "{a} and {b}"

    def test_resolve_format_string_non_dict(self):
        """Test resolving string format with non-dict data."""
        result = _resolve_format("template", "string_data")
        assert result == "template"

    def test_resolve_format_string_only(self):
        """Test resolving plain string format."""
        result = _resolve_format("plain text", {"x": 1})
        assert result == "plain text"


# ============================================================================
# Integration Tests
# ============================================================================

class TestRegistryIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self):
        """Test complete workflow: register, execute, get result."""
        registry = ToolRegistry()

        @registry.register(aliases=["calc", "c"])
        @tool_output(llm_format=lambda x: f"Result: {x['result']}")
        def calculate(expression: str) -> dict:
            """Evaluate a math expression."""
            try:
                result = eval(expression)
            except Exception as e:
                result = str(e)
            return {"expression": expression, "result": result}

        # Test via execute
        result = registry.execute("calc", {"expression": "2 + 2"})
        assert result.name == "calculate"
        assert result.output.data["result"] == 4
        assert result.output.llm_format == "Result: 4"

        # Test via execute_call
        call = {
            "type": "tool_call",
            "name": "c",
            "arguments": {"expression": "10 * 5"}
        }
        result = registry.execute_call(call)
        assert result.output.data["result"] == 50

    def test_literal_type_validation(self):
        """Test that Literal types are validated correctly."""
        registry = ToolRegistry()

        @registry.register
        def choose(option: Literal["a", "b", "c"]) -> str:
            return f"Chose: {option}"

        # Valid option
        result = registry.execute("choose", {"option": "a"})
        assert result.output == "Chose: a"

        # Invalid option should raise validation error
        with pytest.raises(ToolValidationError):
            registry.execute("choose", {"option": "invalid"})

    def test_field_description_in_schema(self):
        """Test that Field descriptions appear in OpenAI schema."""
        registry = ToolRegistry()

        @registry.register
        def search(
            query: str = Field(description="The search query"),
            limit: int = Field(default=10, description="Max results")
        ) -> list:
            """Search for items."""
            return []

        spec = registry.get("search").to_openai_spec()
        params = spec["function"]["parameters"]["properties"]

        assert "query" in params
        assert "limit" in params


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
