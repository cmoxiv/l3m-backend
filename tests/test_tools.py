#!/usr/bin/env python3
"""
Comprehensive tests for tools.py - LLM tool-calling system.
"""

import json
import os
import sys
import pytest
import urllib.error
from datetime import datetime
from unittest.mock import patch, Mock, MagicMock
from urllib.error import URLError, HTTPError

from l3m_backend.tools import (
    get_weather,
    calculate,
    get_time,
    registry,
)
from l3m_backend.tools.weather import _geocode, _get_weather_code_description
from l3m_backend.core import ToolOutput, ToolRegistry


# ============================================================================
# Weather Tool Tests
# ============================================================================

class TestWeatherTool:
    """Tests for get_weather() function."""

    def test_get_weather_success(self):
        """Test get_weather with valid city."""
        # Arrange
        mock_geocode_response = json.dumps({
            "results": [{
                "latitude": 48.8566,
                "longitude": 2.3522,
                "name": "Paris"
            }]
        }).encode()

        mock_weather_response = json.dumps({
            "current": {
                "temperature_2m": 22.6,
                "weather_code": 1
            }
        }).encode()

        # Act
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value.read.side_effect = [
                mock_geocode_response,
                mock_weather_response
            ]
            mock_urlopen.return_value = mock_context_manager

            result = get_weather("Paris")

        # Assert
        assert isinstance(result, ToolOutput)
        assert result.data["location"] == "Paris"
        assert result.data["temperature"] == 23  # rounded
        assert result.data["condition"] == "mainly clear"
        assert result.llm_format == "Paris: 23°C, mainly clear"

    def test_get_weather_city_not_found(self):
        """Test get_weather with non-existent city."""
        # Arrange
        mock_geocode_response = json.dumps({"results": []}).encode()

        # Act
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value.read.return_value = mock_geocode_response
            mock_urlopen.return_value = mock_context_manager

            result = get_weather("NonExistentCity12345")

        # Assert
        assert result.data["location"] == "NonExistentCity12345"
        assert result.data["temperature"] == "unknown"
        assert "error" in result.data["condition"]

    def test_get_weather_network_error(self):
        """Test get_weather handles network errors gracefully."""
        # Arrange & Act
        with patch('urllib.request.urlopen', side_effect=URLError("Network error")):
            result = get_weather("Paris")

        # Assert
        assert result.data["location"] == "Paris"
        assert result.data["temperature"] == "unknown"
        assert "error" in result.data["condition"]

    def test_get_weather_timeout(self):
        """Test get_weather handles timeout errors."""
        # Arrange & Act
        with patch('urllib.request.urlopen', side_effect=TimeoutError("Request timeout")):
            result = get_weather("London")

        # Assert
        assert result.data["location"] == "London"
        assert result.data["temperature"] == "unknown"
        assert "error" in result.data["condition"]

    def test_get_weather_registered_with_aliases(self):
        """Test get_weather is registered with correct aliases."""
        # Assert
        assert "get_weather" in registry
        assert "weather" in registry
        assert "w" in registry

        # Verify all resolve to same tool
        tool1 = registry.get("get_weather")
        tool2 = registry.get("weather")
        tool3 = registry.get("w")
        assert tool1.name == tool2.name == tool3.name == "get_weather"


class TestWeatherHelpers:
    """Tests for weather helper functions."""

    def test_geocode_success(self):
        """Test _geocode returns correct coordinates."""
        # Arrange
        mock_response = json.dumps({
            "results": [{
                "latitude": 51.5074,
                "longitude": -0.1278,
                "name": "London"
            }]
        }).encode()

        # Act
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value.read.return_value = mock_response
            mock_urlopen.return_value = mock_context_manager

            lat, lon, name = _geocode("London")

        # Assert
        assert lat == 51.5074
        assert lon == -0.1278
        assert name == "London"

    def test_geocode_city_not_found(self):
        """Test _geocode raises ValueError for non-existent city."""
        # Arrange
        mock_response = json.dumps({"results": []}).encode()

        # Act & Assert
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value.read.return_value = mock_response
            mock_urlopen.return_value = mock_context_manager

            with pytest.raises(ValueError, match="City not found"):
                _geocode("InvalidCity")

    def test_get_weather_code_description(self):
        """Test weather code to description mapping."""
        # Assert known codes
        assert _get_weather_code_description(0) == "clear sky"
        assert _get_weather_code_description(1) == "mainly clear"
        assert _get_weather_code_description(61) == "slight rain"
        assert _get_weather_code_description(95) == "thunderstorm"

        # Assert unknown code
        assert _get_weather_code_description(999) == "unknown"


# ============================================================================
# Calculator Tool Tests
# ============================================================================

class TestCalculatorTool:
    """Tests for calculate() function."""

    def test_calculate_addition(self):
        """Test calculate with addition."""
        # Act
        result = calculate("2 + 3")

        # Assert
        assert isinstance(result, ToolOutput)
        assert result.data["expression"] == "2 + 3"
        assert result.data["result"] == 5
        assert result.llm_format == "Result: 5"

    def test_calculate_subtraction(self):
        """Test calculate with subtraction."""
        # Act
        result = calculate("10 - 7")

        # Assert
        assert result.data["expression"] == "10 - 7"
        assert result.data["result"] == 3

    def test_calculate_multiplication(self):
        """Test calculate with multiplication."""
        # Act
        result = calculate("4 * 5")

        # Assert
        assert result.data["expression"] == "4 * 5"
        assert result.data["result"] == 20

    def test_calculate_division(self):
        """Test calculate with division."""
        # Act
        result = calculate("20 / 4")

        # Assert
        assert result.data["expression"] == "20 / 4"
        assert result.data["result"] == 5.0

    def test_calculate_complex_expression(self):
        """Test calculate with complex expression."""
        # Act
        result = calculate("(10 + 5) * 2 - 3")

        # Assert
        assert result.data["expression"] == "(10 + 5) * 2 - 3"
        assert result.data["result"] == 27

    def test_calculate_float_operations(self):
        """Test calculate with floating point numbers."""
        # Act
        result = calculate("3.14 * 2")

        # Assert
        assert result.data["result"] == pytest.approx(6.28)

    def test_calculate_division_by_zero(self):
        """Test calculate handles division by zero."""
        # Act
        result = calculate("10 / 0")

        # Assert
        assert result.data["expression"] == "10 / 0"
        assert "Error" in result.data["result"]

    def test_calculate_invalid_characters(self):
        """Test calculate rejects invalid characters (security)."""
        # Act
        result = calculate("__import__('os').system('ls')")

        # Assert
        assert result.data["result"] == "Invalid expression"

    def test_calculate_letters_rejected(self):
        """Test calculate rejects letters."""
        # Act
        result = calculate("2 + x")

        # Assert
        assert result.data["result"] == "Invalid expression"

    def test_calculate_empty_expression(self):
        """Test calculate with empty expression."""
        # Act
        result = calculate("")

        # Assert
        # Empty string is valid (evaluates but might error)
        assert result.data["expression"] == ""
        # Result will be either error or empty evaluation

    def test_calculate_only_whitespace(self):
        """Test calculate with only whitespace."""
        # Act
        result = calculate("   ")

        # Assert
        assert result.data["expression"] == "   "

    def test_calculate_parentheses_only(self):
        """Test calculate with empty parentheses evaluates to tuple."""
        # Act
        result = calculate("()")

        # Assert
        # Empty parentheses () evaluates to empty tuple in Python
        assert result.data["result"] == ()

    def test_calculate_registered_with_aliases(self):
        """Test calculate is registered with correct aliases."""
        # Assert
        assert "calculate" in registry
        assert "calc" in registry
        assert "c" in registry

        # Verify all resolve to same tool
        tool1 = registry.get("calculate")
        tool2 = registry.get("calc")
        tool3 = registry.get("c")
        assert tool1.name == tool2.name == tool3.name == "calculate"


# ============================================================================
# Time Tool Tests
# ============================================================================

class TestTimeTool:
    """Tests for get_time() function."""

    def test_get_time_format(self):
        """Test get_time returns correct format."""
        # Act
        result = get_time()

        # Assert
        assert isinstance(result, ToolOutput)
        assert "current_time" in result.data

        # Verify format: YYYY-MM-DD HH:MM:SS
        time_str = result.data["current_time"]
        try:
            parsed = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            assert parsed is not None
        except ValueError:
            pytest.fail(f"Time format incorrect: {time_str}")

    def test_get_time_reasonable_value(self):
        """Test get_time returns current time (not some random date)."""
        # Act
        result = get_time()
        time_str = result.data["current_time"]
        parsed = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        now = datetime.now()

        # Assert - time should be within 2 seconds of now
        diff = abs((now - parsed).total_seconds())
        assert diff < 2, f"Time difference too large: {diff} seconds"

    def test_get_time_llm_format(self):
        """Test get_time has correct llm_format."""
        # Act
        result = get_time()

        # Assert
        assert result.llm_format == result.data["current_time"]

    def test_get_time_registered_with_aliases(self):
        """Test get_time is registered with correct aliases."""
        # Assert
        assert "get_time" in registry
        assert "time" in registry
        assert "t" in registry

        # Verify all resolve to same tool
        tool1 = registry.get("get_time")
        tool2 = registry.get("time")
        tool3 = registry.get("t")
        assert tool1.name == tool2.name == tool3.name == "get_time"


# ============================================================================
# ToolOutput Wrapper Tests
# ============================================================================

class TestToolOutputWrapper:
    """Tests for ToolOutput wrapper behavior."""

    def test_tool_output_structure(self):
        """Test ToolOutput has correct structure."""
        # Act
        result = get_time()

        # Assert
        assert isinstance(result, ToolOutput)
        assert hasattr(result, "type_name")
        assert hasattr(result, "id")
        assert hasattr(result, "data")
        assert hasattr(result, "llm_format")
        assert hasattr(result, "gui_format")

    def test_tool_output_type_name(self):
        """Test ToolOutput type_name is correct."""
        # Act
        result = calculate("1 + 1")

        # Assert
        assert result.type_name == "dict"

    def test_tool_output_id_unique(self):
        """Test ToolOutput generates unique IDs."""
        # Act
        result1 = get_time()
        result2 = get_time()

        # Assert
        assert result1.id != result2.id
        assert result1.id.startswith("dict_")
        assert result2.id.startswith("dict_")

    def test_tool_output_string_format_template(self):
        """Test ToolOutput with string format template."""
        # Arrange
        mock_geocode_response = json.dumps({
            "results": [{
                "latitude": 48.8566,
                "longitude": 2.3522,
                "name": "Paris"
            }]
        }).encode()

        mock_weather_response = json.dumps({
            "current": {
                "temperature_2m": 20.0,
                "weather_code": 0
            }
        }).encode()

        # Act
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value.read.side_effect = [
                mock_geocode_response,
                mock_weather_response
            ]
            mock_urlopen.return_value = mock_context_manager

            result = get_weather("Paris")

        # Assert - format string "{city}: {temperature}°C, {condition}"
        expected = "Paris: 20°C, clear sky"
        assert result.llm_format == expected

    def test_tool_output_lambda_format(self):
        """Test ToolOutput with lambda format."""
        # Act
        result = calculate("5 * 5")

        # Assert - lambda format: lambda x: f"Result: {x['result']}"
        assert result.llm_format == "Result: 25"

    def test_tool_output_gui_format_none(self):
        """Test ToolOutput gui_format can be None."""
        # Act
        result = get_time()

        # Assert
        assert result.gui_format is None


# ============================================================================
# Registry Integration Tests
# ============================================================================

class TestRegistryIntegration:
    """Tests for tool registry integration."""

    def test_all_tools_registered(self):
        """Test all tools are registered."""
        # Assert
        tool_names = [entry.name for entry in registry]
        assert "get_weather" in tool_names
        assert "calculate" in tool_names
        assert "get_time" in tool_names

    def test_execute_via_registry(self):
        """Test tools can be executed via registry."""
        # Act
        result = registry.execute("calculate", {"expression": "10 + 5"})

        # Assert
        assert result.name == "calculate"
        assert result.arguments == {"expression": "10 + 5"}
        assert result.output.data["result"] == 15

    def test_execute_via_alias(self):
        """Test tools can be executed via alias."""
        # Act
        result = registry.execute("c", {"expression": "2 * 3"})

        # Assert
        assert result.name == "calculate"
        assert result.output.data["result"] == 6

    def test_execute_call_dict(self):
        """Test execute_call with tool_call dict."""
        # Arrange
        call = {
            "type": "tool_call",
            "name": "calculate",
            "arguments": {"expression": "7 + 3"}
        }

        # Act
        result = registry.execute_call(call)

        # Assert
        assert result.name == "calculate"
        assert result.output.data["result"] == 10

    def test_tool_to_openai_spec(self):
        """Test tools generate valid OpenAI specs."""
        # Act
        specs = registry.to_openai_tools()

        # Assert - we now have many more tools
        assert len(specs) >= 3
        spec_names = [s["function"]["name"] for s in specs]
        assert "get_weather" in spec_names
        assert "calculate" in spec_names
        assert "get_time" in spec_names

        # Check structure
        for spec in specs:
            assert spec["type"] == "function"
            assert "function" in spec
            assert "name" in spec["function"]
            assert "description" in spec["function"]
            assert "parameters" in spec["function"]


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_weather_with_special_characters_in_city(self):
        """Test weather with special characters in city name."""
        # Arrange
        mock_response = json.dumps({"results": []}).encode()

        # Act
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value.read.return_value = mock_response
            mock_urlopen.return_value = mock_context_manager

            result = get_weather("São Paulo")

        # Assert - should handle gracefully
        assert result.data["location"] == "São Paulo"

    def test_calculate_with_nested_parentheses(self):
        """Test calculate with deeply nested parentheses."""
        # Act
        result = calculate("((2 + 3) * (4 + 1))")

        # Assert
        assert result.data["result"] == 25

    def test_calculate_negative_numbers(self):
        """Test calculate with negative numbers."""
        # Act
        result = calculate("-5 + 3")

        # Assert
        assert result.data["result"] == -2

    def test_calculate_very_long_expression(self):
        """Test calculate with very long expression."""
        # Act
        long_expr = " + ".join(["1"] * 100)
        result = calculate(long_expr)

        # Assert
        assert result.data["result"] == 100

    def test_weather_http_error(self):
        """Test weather handles HTTP errors."""
        # Arrange
        http_error = HTTPError("url", 404, "Not Found", {}, None)

        # Act
        with patch('urllib.request.urlopen', side_effect=http_error):
            result = get_weather("Paris")

        # Assert
        assert result.data["temperature"] == "unknown"
        assert "error" in result.data["condition"]

    def test_time_multiple_calls_different(self):
        """Test multiple time calls return different values (time progresses)."""
        # Act
        result1 = get_time()
        import time
        time.sleep(1.1)  # Sleep for over a second
        result2 = get_time()

        # Assert
        time1 = result1.data["current_time"]
        time2 = result2.data["current_time"]
        assert time1 != time2  # Times should be different


# ============================================================================
# Information & Lookup Tools Tests
# ============================================================================

class TestWikipediaTool:
    """Tests for wikipedia() function."""

    def test_wikipedia_success(self):
        """Test wikipedia with valid topic."""
        mock_response = json.dumps({
            "title": "Python (programming language)",
            "extract": "Python is a programming language.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Python"}}
        }).encode()

        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value.read.return_value = mock_response
            mock_urlopen.return_value = mock_context_manager

            from l3m_backend.tools import wikipedia
            result = wikipedia("Python programming")

        assert result.data["title"] == "Python (programming language)"
        assert "programming language" in result.data["summary"]

    def test_wikipedia_not_found(self):
        """Test wikipedia with non-existent topic."""
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                "url", 404, "Not Found", {}, None
            )

            from l3m_backend.tools import wikipedia
            result = wikipedia("xyznonexistent123")

        assert "error" in result.data
        assert "found" in result.data["error"].lower()  # "No Wikipedia article found"

    def test_wikipedia_registered(self):
        """Test wikipedia is registered with alias."""
        assert "wikipedia" in registry
        assert "wiki" in registry


class TestDefineWordTool:
    """Tests for define_word() function."""

    def test_define_word_success(self):
        """Test define_word with valid word."""
        mock_response = json.dumps([{
            "word": "hello",
            "meanings": [{
                "partOfSpeech": "noun",
                "definitions": [{
                    "definition": "A greeting",
                    "example": "Hello, world!"
                }]
            }]
        }]).encode()

        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value.read.return_value = mock_response
            mock_urlopen.return_value = mock_context_manager

            from l3m_backend.tools import define_word
            result = define_word("hello")

        assert result.data["word"] == "hello"
        assert result.data["part_of_speech"] == "noun"
        assert "greeting" in result.data["definition"].lower()

    def test_define_word_not_found(self):
        """Test define_word with non-existent word."""
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                "url", 404, "Not Found", {}, None
            )

            from l3m_backend.tools import define_word
            result = define_word("xyznonexistent")

        assert "error" in result.data

    def test_define_word_registered(self):
        """Test define_word is registered with aliases."""
        assert "define_word" in registry
        assert "define" in registry
        assert "dict" in registry


class TestUnitConvertTool:
    """Tests for unit_convert() function."""

    def test_unit_convert_km_to_miles(self):
        """Test kilometers to miles conversion."""
        from l3m_backend.tools import unit_convert
        result = unit_convert(10, "km", "miles")

        assert result.data["value"] == 10
        assert result.data["from_unit"] == "km"
        assert result.data["to_unit"] == "miles"
        assert abs(result.data["result"] - 6.2137) < 0.01

    def test_unit_convert_celsius_to_fahrenheit(self):
        """Test Celsius to Fahrenheit conversion."""
        from l3m_backend.tools import unit_convert
        result = unit_convert(0, "c", "f")

        assert result.data["result"] == 32

    def test_unit_convert_kg_to_lb(self):
        """Test kilograms to pounds conversion."""
        from l3m_backend.tools import unit_convert
        result = unit_convert(1, "kg", "lb")

        assert abs(result.data["result"] - 2.2046) < 0.01

    def test_unit_convert_unknown(self):
        """Test unknown conversion returns error."""
        from l3m_backend.tools import unit_convert
        result = unit_convert(100, "foo", "bar")

        assert "error" in result.data
        assert "available" in result.data

    def test_unit_convert_registered(self):
        """Test unit_convert is registered with aliases."""
        assert "unit_convert" in registry
        assert "convert" in registry
        assert "unit" in registry


class TestCurrencyConvertTool:
    """Tests for currency_convert() function."""

    def test_currency_convert_success(self):
        """Test currency conversion."""
        mock_response = json.dumps({
            "rates": {"EUR": 0.85, "GBP": 0.73}
        }).encode()

        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value.read.return_value = mock_response
            mock_urlopen.return_value = mock_context_manager

            from l3m_backend.tools import currency_convert
            result = currency_convert(100, "USD", "EUR")

        assert result.data["amount"] == 100
        assert result.data["from_currency"] == "USD"
        assert result.data["to_currency"] == "EUR"
        assert result.data["result"] == 85.0

    def test_currency_convert_unknown_currency(self):
        """Test unknown target currency."""
        mock_response = json.dumps({"rates": {"EUR": 0.85}}).encode()

        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value.read.return_value = mock_response
            mock_urlopen.return_value = mock_context_manager

            from l3m_backend.tools import currency_convert
            result = currency_convert(100, "USD", "XYZ")

        assert "error" in result.data

    def test_currency_convert_registered(self):
        """Test currency_convert is registered with aliases."""
        assert "currency_convert" in registry
        assert "currency" in registry
        assert "fx" in registry


# ============================================================================
# Productivity Tools Tests
# ============================================================================

class TestNoteTool:
    """Tests for note() function."""

    def test_note_save_and_get(self, tmp_path):
        """Test saving and retrieving a note."""
        from l3m_backend.tools import note
        from l3m_backend.tools import _storage

        # Temporarily change storage location
        original_dir = _storage._STORAGE_DIR
        original_file = _storage._NOTES_FILE
        _storage._STORAGE_DIR = tmp_path
        _storage._NOTES_FILE = tmp_path / "notes.json"

        try:
            # Save a note
            result = note("save", "test_note", "Hello world")
            assert "saved" in result.data["message"].lower()

            # Get the note
            result = note("get", "test_note")
            assert result.data["content"] == "Hello world"

            # List notes
            result = note("list")
            assert "test_note" in result.data["notes"]

            # Delete the note
            result = note("delete", "test_note")
            assert "deleted" in result.data["message"].lower()
        finally:
            _storage._STORAGE_DIR = original_dir
            _storage._NOTES_FILE = original_file

    def test_note_get_not_found(self, tmp_path):
        """Test getting non-existent note."""
        from l3m_backend.tools import note
        from l3m_backend.tools import _storage

        original_dir = _storage._STORAGE_DIR
        original_file = _storage._NOTES_FILE
        _storage._STORAGE_DIR = tmp_path
        _storage._NOTES_FILE = tmp_path / "notes.json"

        try:
            result = note("get", "nonexistent")
            assert "error" in result.data
        finally:
            _storage._STORAGE_DIR = original_dir
            _storage._NOTES_FILE = original_file

    def test_note_registered(self):
        """Test note is registered with alias."""
        assert "note" in registry
        assert "notes" in registry


class TestTodoTool:
    """Tests for todo() function."""

    def test_todo_add_and_list(self, tmp_path):
        """Test adding and listing todos."""
        from l3m_backend.tools import todo
        from l3m_backend.tools import _storage

        original_dir = _storage._STORAGE_DIR
        original_file = _storage._TODOS_FILE
        _storage._STORAGE_DIR = tmp_path
        _storage._TODOS_FILE = tmp_path / "todos.json"

        try:
            # Add a todo
            result = todo("add", "Buy groceries")
            assert "Added" in result.data["message"]

            # List todos
            result = todo("list")
            assert len(result.data["items"]) == 1
            assert result.data["items"][0]["task"] == "Buy groceries"

            # Mark done
            result = todo("done", "1")
            assert "done" in result.data["message"].lower()

            # Clear
            result = todo("clear")
            assert "cleared" in result.data["message"].lower()
        finally:
            _storage._STORAGE_DIR = original_dir
            _storage._TODOS_FILE = original_file

    def test_todo_registered(self):
        """Test todo is registered with aliases."""
        assert "todo" in registry
        assert "todos" in registry
        assert "task" in registry


class TestTimerTool:
    """Tests for timer() function."""

    def test_timer_basic(self):
        """Test basic timer creation."""
        from l3m_backend.tools import timer
        result = timer(300, "Pomodoro")

        assert result.data["label"] == "Pomodoro"
        assert result.data["duration_seconds"] == 300
        assert "5m" in result.data["message"]

    def test_timer_registered(self):
        """Test timer is registered with alias."""
        assert "timer" in registry
        assert "countdown" in registry


# ============================================================================
# Development Tools Tests
# ============================================================================

class TestRunPythonTool:
    """Tests for run_python() function."""

    def test_run_python_success(self):
        """Test running valid Python code."""
        from l3m_backend.tools import run_python
        result = run_python("print('Hello, World!')")

        assert result.data["output"] == "Hello, World!"
        assert result.data["exit_code"] == 0

    def test_run_python_with_calculation(self):
        """Test running Python calculation."""
        from l3m_backend.tools import run_python
        result = run_python("print(2 + 2)")

        assert "4" in result.data["output"]

    def test_run_python_error(self):
        """Test running invalid Python code."""
        from l3m_backend.tools import run_python
        result = run_python("raise ValueError('test')")

        assert "ValueError" in result.data["output"]
        assert result.data["exit_code"] != 0

    def test_run_python_registered(self):
        """Test run_python is registered with aliases."""
        assert "run_python" in registry
        assert "python" in registry
        assert "py" in registry


class TestReadFileTool:
    """Tests for read_file() function."""

    def test_read_file_success(self, tmp_path):
        """Test reading a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello\nWorld\n")

        from l3m_backend.tools import read_file
        result = read_file(str(test_file))

        assert "Hello" in result.data["content"]
        assert result.data["lines"] == 2

    def test_read_file_not_found(self):
        """Test reading non-existent file."""
        from l3m_backend.tools import read_file
        result = read_file("/nonexistent/file.txt")

        assert "error" in result.data
        assert "not found" in result.data["error"].lower()

    def test_read_file_registered(self):
        """Test read_file is registered with aliases."""
        assert "read_file" in registry
        assert "read" in registry
        assert "cat" in registry


class TestWriteFileTool:
    """Tests for write_file() function."""

    def test_write_file_success(self, tmp_path):
        """Test writing a file."""
        test_file = tmp_path / "output.txt"

        from l3m_backend.tools import write_file
        result = write_file(str(test_file), "Hello, World!")

        assert "Wrote" in result.data["message"]
        assert test_file.read_text() == "Hello, World!"

    def test_write_file_append(self, tmp_path):
        """Test appending to a file."""
        test_file = tmp_path / "output.txt"
        test_file.write_text("Line 1\n")

        from l3m_backend.tools import write_file
        result = write_file(str(test_file), "Line 2\n", append=True)

        assert "Appended" in result.data["message"]
        assert test_file.read_text() == "Line 1\nLine 2\n"

    def test_write_file_registered(self):
        """Test write_file is registered with aliases."""
        assert "write_file" in registry
        assert "write" in registry


class TestShellCmdTool:
    """Tests for shell_cmd() function."""

    def test_shell_cmd_success(self):
        """Test running a shell command."""
        from l3m_backend.tools import shell_cmd
        result = shell_cmd("echo 'Hello'")

        assert "Hello" in result.data["output"]
        assert result.data["exit_code"] == 0

    def test_shell_cmd_blocked(self):
        """Test blocked command."""
        from l3m_backend.tools import shell_cmd
        result = shell_cmd("sudo ls")

        assert "error" in result.data
        assert "blocked" in result.data["error"].lower()

    def test_shell_cmd_registered(self):
        """Test shell_cmd is registered with aliases."""
        assert "shell_cmd" in registry
        assert "shell" in registry
        assert "sh" in registry


class TestHttpRequestTool:
    """Tests for http_request() function."""

    def test_http_request_get(self):
        """Test HTTP GET request."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.read.return_value = b'{"message": "success"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            from l3m_backend.tools import http_request
            result = http_request("https://example.com/api")

        assert result.data["status"] == 200
        assert "success" in result.data["body"]

    def test_http_request_registered(self):
        """Test http_request is registered with aliases."""
        assert "http_request" in registry
        assert "http" in registry
        assert "fetch" in registry
        assert "curl" in registry


# ============================================================================
# Utility Tools Tests
# ============================================================================

class TestRandomNumberTool:
    """Tests for random_number() function."""

    def test_random_number_single(self):
        """Test generating a single random number."""
        from l3m_backend.tools import random_number
        result = random_number(1, 10)

        assert 1 <= result.data["result"] <= 10

    def test_random_number_multiple(self):
        """Test generating multiple random numbers."""
        from l3m_backend.tools import random_number
        result = random_number(1, 100, count=5)

        assert len(result.data["results"]) == 5
        for num in result.data["results"]:
            assert 1 <= num <= 100

    def test_random_number_invalid_range(self):
        """Test invalid range."""
        from l3m_backend.tools import random_number
        result = random_number(100, 1)

        assert "error" in result.data

    def test_random_number_registered(self):
        """Test random_number is registered with aliases."""
        assert "random_number" in registry
        assert "rand" in registry
        assert "random" in registry


class TestUuidGenerateTool:
    """Tests for uuid_generate() function."""

    def test_uuid_generate_single(self):
        """Test generating a single UUID."""
        from l3m_backend.tools import uuid_generate
        result = uuid_generate()

        assert "uuid" in result.data
        assert len(result.data["uuid"]) == 36  # Standard UUID format

    def test_uuid_generate_hex_format(self):
        """Test generating UUID in hex format."""
        from l3m_backend.tools import uuid_generate
        result = uuid_generate(format="hex")

        assert len(result.data["uuid"]) == 32  # Hex format (no dashes)

    def test_uuid_generate_multiple(self):
        """Test generating multiple UUIDs."""
        from l3m_backend.tools import uuid_generate
        result = uuid_generate(count=3)

        assert len(result.data["uuids"]) == 3

    def test_uuid_generate_registered(self):
        """Test uuid_generate is registered with alias."""
        assert "uuid_generate" in registry
        assert "uuid" in registry


class TestHashTextTool:
    """Tests for hash_text() function."""

    def test_hash_text_sha256(self):
        """Test SHA256 hashing."""
        from l3m_backend.tools import hash_text
        result = hash_text("hello")

        assert result.data["algorithm"] == "sha256"
        assert len(result.data["hash"]) == 64  # SHA256 hex length
        # Known hash for "hello"
        assert result.data["hash"] == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_hash_text_md5(self):
        """Test MD5 hashing."""
        from l3m_backend.tools import hash_text
        result = hash_text("hello", algorithm="md5")

        assert result.data["algorithm"] == "md5"
        assert len(result.data["hash"]) == 32  # MD5 hex length

    def test_hash_text_registered(self):
        """Test hash_text is registered with alias."""
        assert "hash_text" in registry
        assert "hash" in registry


class TestBase64EncodeTool:
    """Tests for base64_encode() function."""

    def test_base64_encode(self):
        """Test Base64 encoding."""
        from l3m_backend.tools import base64_encode
        result = base64_encode("Hello, World!")

        assert result.data["result"] == "SGVsbG8sIFdvcmxkIQ=="
        assert result.data["action"] == "encoded"

    def test_base64_decode(self):
        """Test Base64 decoding."""
        from l3m_backend.tools import base64_encode
        result = base64_encode("SGVsbG8sIFdvcmxkIQ==", action="decode")

        assert result.data["result"] == "Hello, World!"
        assert result.data["action"] == "decoded"

    def test_base64_invalid_decode(self):
        """Test invalid Base64 decoding."""
        from l3m_backend.tools import base64_encode
        result = base64_encode("not-valid-base64!!!", action="decode")

        assert "error" in result.data

    def test_base64_registered(self):
        """Test base64_encode is registered with alias."""
        assert "base64_encode" in registry
        assert "b64" in registry


class TestJsonFormatTool:
    """Tests for json_format() function."""

    def test_json_format_prettify(self):
        """Test JSON prettification."""
        from l3m_backend.tools import json_format
        result = json_format('{"name":"John","age":30}')

        assert result.data["valid"] is True
        assert '"name": "John"' in result.data["formatted"]

    def test_json_format_minify(self):
        """Test JSON minification."""
        from l3m_backend.tools import json_format
        result = json_format('{"name": "John", "age": 30}', minify=True)

        assert result.data["valid"] is True
        assert result.data["formatted"] == '{"name":"John","age":30}'

    def test_json_format_invalid(self):
        """Test invalid JSON."""
        from l3m_backend.tools import json_format
        result = json_format('not valid json')

        assert result.data["valid"] is False
        assert "error" in result.data

    def test_json_format_registered(self):
        """Test json_format is registered with aliases."""
        assert "json_format" in registry
        assert "json" in registry


class TestWebSearchTool:
    """Tests for web_search() function."""

    def test_web_search_real_result(self):
        """Test web_search returns results from DuckDuckGo."""
        from l3m_backend.tools import web_search
        result = web_search("python programming")

        # Can return either instant answer or search results
        assert "type" in result.data or "results" in result.data
        if result.data.get("type") == "instant_answer":
            assert "answer" in result.data
        elif "results" in result.data:
            assert len(result.data["results"]) > 0

    def test_web_search_registered(self):
        """Test web_search is registered with aliases."""
        assert "web_search" in registry
        assert "search" in registry
        assert "google" in registry


class TestReminderTool:
    """Tests for reminder() function."""

    def test_reminder_add_and_list(self, tmp_path):
        """Test adding and listing reminders."""
        from l3m_backend.tools import reminder
        from l3m_backend.tools import _storage

        original_dir = _storage._STORAGE_DIR
        original_file = _storage._REMINDERS_FILE
        _storage._STORAGE_DIR = tmp_path
        _storage._REMINDERS_FILE = tmp_path / "reminders.json"

        try:
            # Add a reminder
            result = reminder("add", "Call Mom", "3pm")
            assert "Reminder set" in result.data["message"]

            # List reminders
            result = reminder("list")
            assert len(result.data["reminders"]) == 1
            assert result.data["reminders"][0]["text"] == "Call Mom"
        finally:
            _storage._STORAGE_DIR = original_dir
            _storage._REMINDERS_FILE = original_file

    def test_reminder_registered(self):
        """Test reminder is registered with aliases."""
        assert "reminder" in registry
        assert "remind" in registry


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
