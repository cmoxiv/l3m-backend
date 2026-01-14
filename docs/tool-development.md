# Tool Development Guide

A comprehensive guide to creating custom tools for l3m-backend.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Tool Structure](#tool-structure)
4. [Registration Options](#registration-options)
5. [Parameter Definitions](#parameter-definitions)
6. [Output Formatting](#output-formatting)
7. [Error Handling](#error-handling)
8. [Testing Tools](#testing-tools)
9. [Advanced Patterns](#advanced-patterns)
10. [Best Practices](#best-practices)

---

## Overview

Tools in l3m-backend are Python functions that can be called by the LLM during conversations. The system uses Pydantic for automatic validation and schema generation.

### Key Concepts

- **Tool**: A Python function registered with `@registry.register()`
- **Schema**: Auto-generated from function signature and type hints
- **Validation**: Arguments validated via Pydantic before execution
- **Output**: Can be wrapped with `@tool_output()` for formatting

---

## Quick Start

### Using the CLI

```bash
# Create a basic tool scaffold
l3m-tools create my_tool

# Create a wrapper for a shell command
l3m-tools create docker-ps --wrap "docker ps"
```

This creates `~/.l3m/tools/my_tool/__init__.py` with a basic template.

### Manual Creation

Create a directory in `~/.l3m/tools/` with an `__init__.py`:

```python
# ~/.l3m/tools/my_tool/__init__.py
from l3m_backend.tools import registry
from l3m_backend.core import tool_output

@registry.register(aliases=["mt"])
@tool_output(llm_format="{result}")
def my_tool(input: str) -> dict:
    """Process input and return a result.

    Args:
        input: The text to process
    """
    return {"result": input.upper()}
```

Tools are automatically loaded when l3m-chat starts.

---

## Tool Structure

### Directory Layout

```
~/.l3m/tools/
├── my_tool/
│   └── __init__.py    # Tool implementation
├── api_client/
│   ├── __init__.py    # Main tool
│   └── helpers.py     # Helper modules
└── data_processor/
    ├── __init__.py
    └── config.json    # Tool-specific config
```

### Basic Template

```python
"""My Tool - Brief description of what it does."""

from typing import Any
from l3m_backend.tools import registry
from l3m_backend.core import tool_output

@registry.register(aliases=["mt", "mytool"])
@tool_output(llm_format="{result}")
def my_tool(input: str) -> dict[str, Any]:
    """Process the input and return a result.

    This is a longer description that explains what the tool does
    in more detail. This text appears in the tool schema.

    Args:
        input: The text to process
    """
    # Your implementation here
    result = input.upper()
    return {"result": result, "input": input}
```

---

## Registration Options

### @registry.register()

```python
@registry.register(
    name: str | None = None,        # Override function name
    description: str | None = None,  # Override docstring
    aliases: list[str] | None = None # Alternative names
)
```

### Examples

```python
# Basic registration (uses function name)
@registry.register
def get_weather(location: str) -> dict:
    """Get weather for a location."""
    ...

# With aliases
@registry.register(aliases=["weather", "w"])
def get_weather(location: str) -> dict:
    """Get weather for a location."""
    ...

# Custom name and description
@registry.register(
    name="fetch_weather",
    description="Retrieve current weather conditions for any city",
    aliases=["weather"]
)
def get_weather(location: str) -> dict:
    ...

# Multiple tools from one module
@registry.register(aliases=["add"])
def calculator_add(a: float, b: float) -> dict:
    """Add two numbers."""
    return {"result": a + b}

@registry.register(aliases=["mult"])
def calculator_multiply(a: float, b: float) -> dict:
    """Multiply two numbers."""
    return {"result": a * b}
```

---

## Parameter Definitions

### Type Hints

Use standard Python type hints for parameters:

```python
from typing import Literal, Optional

@registry.register()
def my_tool(
    required_str: str,              # Required string
    required_int: int,              # Required integer
    optional_str: str = "default",  # Optional with default
    optional_int: int = 0,          # Optional with default
    choice: Literal["a", "b", "c"] = "a",  # Enum/choice
    maybe_value: Optional[str] = None,     # Optional nullable
) -> dict:
    ...
```

### Using Pydantic Field

For detailed parameter descriptions:

```python
from pydantic import Field

@registry.register()
def get_weather(
    location: str = Field(description="City name or coordinates"),
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit to use"
    ),
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    ),
) -> dict:
    """Get current weather for a location."""
    ...
```

### Generated Schema

The above generates this OpenAI-compatible schema:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a location.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "City name or coordinates"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "default": "celsius",
          "description": "Temperature unit to use"
        },
        "include_forecast": {
          "type": "boolean",
          "default": false,
          "description": "Include 5-day forecast"
        }
      },
      "required": ["location"],
      "additionalProperties": false
    }
  }
}
```

### Complex Types

```python
from typing import List, Dict

@registry.register()
def process_data(
    items: List[str],           # List of strings
    options: Dict[str, Any],    # Dictionary
    count: int = 10,            # Integer with default
) -> dict:
    ...
```

---

## Output Formatting

### @tool_output()

Controls how tool results are presented to the LLM.

```python
@tool_output(
    llm_format: str | Callable[[Any], str] | None = None,
    gui_format: str | Callable[[Any], str] | None = None,
)
```

### String Template

Uses Python's `str.format()` with dict keys:

```python
@tool_output(llm_format="{location}: {temperature}C, {condition}")
def get_weather(location: str) -> dict:
    return {
        "location": "Paris",
        "temperature": 22,
        "condition": "sunny"
    }
# LLM sees: "Paris: 22C, sunny"
```

### Callable Formatter

For complex formatting logic:

```python
@tool_output(llm_format=lambda x: f"Found {len(x['results'])} results:\n" +
             "\n".join(f"- {r}" for r in x['results']))
def search(query: str) -> dict:
    return {"results": ["Result 1", "Result 2", "Result 3"]}
# LLM sees:
# Found 3 results:
# - Result 1
# - Result 2
# - Result 3
```

### Dual Formatting

Different formats for LLM and GUI:

```python
@tool_output(
    llm_format="{summary}",
    gui_format="<div class='weather'><h3>{location}</h3><p>{temperature}C</p></div>"
)
def get_weather(location: str) -> dict:
    return {
        "location": "Paris",
        "temperature": 22,
        "summary": "Paris is 22C and sunny"
    }
```

### Without Formatting

If no `@tool_output`, raw dict is JSON-serialized:

```python
@registry.register()
def my_tool(x: int) -> dict:
    return {"value": x * 2}
# LLM sees: {"value": 4}
```

---

## Error Handling

### Return Errors in Output

```python
@registry.register()
@tool_output(llm_format=lambda x: x.get("error") or x.get("result"))
def safe_divide(a: float, b: float) -> dict:
    """Divide two numbers safely."""
    if b == 0:
        return {"error": "Cannot divide by zero"}
    return {"result": a / b}
```

### Raise Exceptions

Exceptions are caught and returned to the LLM:

```python
from l3m_backend.core import ToolError

@registry.register()
def risky_operation(data: str) -> dict:
    """Perform a risky operation."""
    if not data:
        raise ToolError("Data cannot be empty")
    return {"result": process(data)}
```

The LLM receives: `{"error": "Data cannot be empty"}`

### Validation Errors

Pydantic validation errors are automatically formatted:

```python
@registry.register()
def process(count: int) -> dict:
    """Process with count."""
    return {"processed": count}

# If LLM calls with count="abc", it gets:
# {"error": "1 validation error for...\ncount\n  value is not a valid integer"}
```

---

## Testing Tools

### Unit Testing

```python
# tests/test_my_tool.py
import pytest
from l3m_backend.tools import registry

def test_my_tool_basic():
    """Test basic functionality."""
    result = registry.execute("my_tool", {"input": "hello"})
    assert result.output.data["result"] == "HELLO"

def test_my_tool_validation():
    """Test argument validation."""
    from l3m_backend.core import ToolValidationError

    with pytest.raises(ToolValidationError):
        registry.execute("my_tool", {"input": 123})  # Wrong type

def test_my_tool_aliases():
    """Test that aliases work."""
    result1 = registry.execute("my_tool", {"input": "test"})
    result2 = registry.execute("mt", {"input": "test"})
    assert result1.output.data == result2.output.data
```

### Integration Testing

```python
def test_tool_with_engine():
    """Test tool works with ChatEngine."""
    from l3m_backend import ChatEngine
    from l3m_backend.tools import registry

    engine = ChatEngine("./model.gguf", registry)
    response = engine.chat("Use my_tool with input 'hello'")

    assert "HELLO" in response
```

### Manual Testing

```bash
# Test via REPL
l3m-chat

# Use magic command to invoke directly
%tool my_tool {"input": "hello"}

# Check schema
/schema my_tool

# List all tools
/tools
```

---

## Advanced Patterns

### Stateful Tools

```python
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self) -> int:
        self.count += 1
        return self.count

_counter = Counter()

@registry.register()
@tool_output(llm_format="Count: {count}")
def increment_counter() -> dict:
    """Increment and return the counter value."""
    return {"count": _counter.increment()}
```

### API Wrapper Tools

```python
import requests

@registry.register(aliases=["api"])
@tool_output(llm_format=lambda x: x.get("error") or str(x.get("data")))
def call_api(
    endpoint: str = Field(description="API endpoint path"),
    method: Literal["GET", "POST"] = "GET",
    data: dict = Field(default_factory=dict, description="Request body for POST"),
) -> dict:
    """Call an external API endpoint."""
    base_url = "https://api.example.com"

    try:
        if method == "GET":
            resp = requests.get(f"{base_url}/{endpoint}", timeout=10)
        else:
            resp = requests.post(f"{base_url}/{endpoint}", json=data, timeout=10)

        resp.raise_for_status()
        return {"data": resp.json()}
    except requests.RequestException as e:
        return {"error": f"API call failed: {str(e)}"}
```

### Tools with Configuration

```python
import json
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {"api_key": None, "base_url": "https://default.api.com"}

CONFIG = load_config()

@registry.register()
def configured_tool(query: str) -> dict:
    """Tool that uses external configuration."""
    if not CONFIG.get("api_key"):
        return {"error": "API key not configured in config.json"}

    # Use CONFIG["api_key"] and CONFIG["base_url"]
    ...
```

### Async Operations

Tools run synchronously, but can use threading for I/O:

```python
import concurrent.futures

@registry.register()
@tool_output(llm_format=lambda x: "\n".join(x["results"]))
def parallel_fetch(urls: list[str]) -> dict:
    """Fetch multiple URLs in parallel."""
    def fetch_one(url):
        import requests
        try:
            return requests.get(url, timeout=5).text[:100]
        except:
            return f"Error fetching {url}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch_one, urls))

    return {"results": results}
```

### Accessing Engine Context

Tools can access the current ChatEngine:

```python
from l3m_backend.engine.context import get_current_engine

@registry.register()
def context_aware_tool(query: str) -> dict:
    """Tool that accesses conversation context."""
    engine = get_current_engine()
    if engine is None:
        return {"error": "No engine context available"}

    # Access conversation history
    history_length = len(engine.history)

    # Use LLM for sub-task (without affecting main history)
    summary = engine.chat(
        f"Summarize this: {query}",
        ignore_history=True,
        temp_system="You are a summarizer."
    )

    return {
        "summary": summary,
        "context_messages": history_length
    }
```

---

## Best Practices

### 1. Clear Descriptions

Write clear, actionable descriptions:

```python
# Good
@registry.register()
def get_weather(location: str) -> dict:
    """Get current weather conditions including temperature, humidity, and forecast.

    Args:
        location: City name (e.g., 'Paris'), or coordinates (e.g., '48.8566,2.3522')
    """

# Bad
@registry.register()
def get_weather(location: str) -> dict:
    """Weather tool."""
```

### 2. Meaningful Return Values

Return structured data the LLM can use:

```python
# Good
@tool_output(llm_format="{city} is {temp}C with {condition}")
def get_weather(location: str) -> dict:
    return {
        "city": "Paris",
        "temp": 22,
        "condition": "partly cloudy",
        "humidity": 65,
        "wind_speed": 12
    }

# Bad
def get_weather(location: str) -> str:
    return "It's 22 degrees"
```

### 3. Handle Edge Cases

```python
@registry.register()
@tool_output(llm_format=lambda x: x.get("error") or f"Found: {x['result']}")
def search_database(query: str) -> dict:
    """Search the database."""
    if not query.strip():
        return {"error": "Query cannot be empty"}

    results = db.search(query)

    if not results:
        return {"error": f"No results found for '{query}'"}

    return {"result": results[0], "total": len(results)}
```

### 4. Use Appropriate Aliases

```python
# Good - short, memorable aliases
@registry.register(aliases=["w", "weather"])
def get_weather(location: str) -> dict: ...

@registry.register(aliases=["calc", "c"])
def calculate(expression: str) -> dict: ...

# Bad - confusing aliases
@registry.register(aliases=["gw", "wthr", "getw"])
def get_weather(location: str) -> dict: ...
```

### 5. Limit Side Effects

Tools should be predictable and safe:

```python
# Good - explicit about side effects
@registry.register()
def write_file(path: str, content: str, confirm: bool = False) -> dict:
    """Write content to a file.

    Args:
        path: File path to write to
        content: Content to write
        confirm: Must be True to actually write (safety check)
    """
    if not confirm:
        return {"error": "Set confirm=True to actually write the file"}

    Path(path).write_text(content)
    return {"success": True, "path": path}
```

### 6. Document Dependencies

```python
"""My API Tool - Requires requests library.

Install: pip install requests
"""

try:
    import requests
except ImportError:
    raise ImportError("This tool requires 'requests'. Install with: pip install requests")

@registry.register()
def api_call(url: str) -> dict:
    ...
```

### 7. Keep Tools Focused

One tool, one purpose:

```python
# Good - separate tools
@registry.register()
def search_web(query: str) -> dict:
    """Search the web."""
    ...

@registry.register()
def fetch_page(url: str) -> dict:
    """Fetch a web page."""
    ...

# Bad - overloaded tool
@registry.register()
def web_tool(action: str, query_or_url: str) -> dict:
    """Search or fetch depending on action."""
    if action == "search":
        ...
    elif action == "fetch":
        ...
```

---

## Troubleshooting

### Tool Not Loading

1. Check file location: `~/.l3m/tools/<name>/__init__.py`
2. Check for Python syntax errors
3. Run `l3m-tools list` to see registered tools
4. Check logs for import errors

### Tool Not Found by LLM

1. Verify tool is registered: `/tools`
2. Check tool schema: `/schema <tool_name>`
3. Use an alias if name is complex
4. Ensure description is clear

### Validation Errors

1. Check type hints match expected input
2. Use `Field()` for complex validation
3. Test with `%tool <name> {args}` in REPL

### Output Not Formatted

1. Ensure `@tool_output()` decorator is applied
2. Check format string keys match return dict keys
3. Test formatter function separately

---

## Example: Complete Tool

```python
"""Stock Price Tool - Get current stock prices.

Dependencies: requests
"""

from typing import Literal
from pydantic import Field
from l3m_backend.tools import registry
from l3m_backend.core import tool_output

try:
    import requests
except ImportError:
    raise ImportError("Install requests: pip install requests")


def _format_stock(data: dict) -> str:
    """Format stock data for LLM."""
    if "error" in data:
        return data["error"]

    change_emoji = "+" if data["change"] >= 0 else ""
    return (
        f"{data['symbol']}: ${data['price']:.2f} "
        f"({change_emoji}{data['change']:.2f}%, {data['volume']}M vol)"
    )


@registry.register(aliases=["stock", "price"])
@tool_output(llm_format=_format_stock)
def get_stock_price(
    symbol: str = Field(description="Stock ticker symbol (e.g., AAPL, GOOGL)"),
    include_history: bool = Field(
        default=False,
        description="Include 30-day price history"
    ),
) -> dict:
    """Get current stock price and trading information.

    Returns real-time stock data including price, change percentage,
    and trading volume. Optionally includes 30-day price history.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, GOOGL)
        include_history: Include 30-day price history
    """
    symbol = symbol.upper().strip()

    if not symbol.isalpha() or len(symbol) > 5:
        return {"error": f"Invalid stock symbol: {symbol}"}

    try:
        # Replace with actual API call
        resp = requests.get(
            f"https://api.example.com/stock/{symbol}",
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        result = {
            "symbol": symbol,
            "price": data["price"],
            "change": data["change_percent"],
            "volume": data["volume"] / 1_000_000,
        }

        if include_history:
            result["history"] = data.get("history_30d", [])

        return result

    except requests.Timeout:
        return {"error": f"Timeout fetching {symbol}"}
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return {"error": f"Stock symbol not found: {symbol}"}
        return {"error": f"API error: {e}"}
    except Exception as e:
        return {"error": f"Failed to fetch stock price: {e}"}
```
