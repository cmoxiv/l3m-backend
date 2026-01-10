# Architecture Documentation

This document provides a detailed explanation of how the LLM tool-calling system works, from registration to execution.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Tool Registration Flow](#tool-registration-flow)
4. [Schema Generation](#schema-generation)
5. [Tool Calling Flow](#tool-calling-flow)
6. [Contract-Based Approach](#contract-based-approach)
7. [Multi-Turn Conversation Loop](#multi-turn-conversation-loop)
8. [Model State Management](#model-state-management)

---

## System Overview

The system enables local LLMs to call functions/tools using a **contract-based approach**. Instead of relying on native function-calling capabilities (which many local models lack), it teaches the model a JSON-based protocol for requesting tool execution.

### Key Design Principles

1. **Model-Agnostic**: Works with any model that can generate valid JSON
2. **Contract-Based**: Uses explicit JSON format in system prompt rather than relying on model's built-in function calling
3. **Type-Safe**: Leverages Pydantic for automatic validation
4. **OpenAI-Compatible**: Generates schemas compatible with OpenAI's tool format
5. **Transparent**: Tool execution happens automatically within the conversation loop

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Input                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  l3m_backend/engine/                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              ChatEngine.chat()                       │  │
│  │  - Builds message context                            │  │
│  │  - Calls LLM via llama-cpp-python                    │  │
│  │  - Parses response (tool call or final)              │  │
│  │  - Executes tools via ToolRegistry                   │  │
│  │  - Loops until final response                        │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   l3m_backend/core/                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              ToolRegistry                            │  │
│  │  - Stores registered tools                           │  │
│  │  - Generates Pydantic models from signatures         │  │
│  │  - Creates OpenAI-compatible schemas                 │  │
│  │  - Validates and executes tool calls                 │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   l3m_backend/tools/                        │
│  - get_weather(location, unit)                              │
│  - get_time()                                               │
│  - calculate(expression)                                    │
│  - And many more...                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Tool Registration Flow

### Step 1: Define a Tool

```python
from l3m_backend import ToolRegistry, tool_output
from pydantic import Field
from typing import Literal

registry = ToolRegistry()

@registry.register(aliases=["weather", "w"])
@tool_output(llm_format="{location}: {temperature}°{unit}, {condition}")
def get_weather(
    location: str = Field(description="City name"),
    unit: Literal["celsius", "fahrenheit"] = "celsius",
) -> dict:
    """Get the current weather for a location."""
    # Implementation...
    return {
        "location": "Paris",
        "temperature": 15,
        "unit": "celsius",
        "condition": "partly cloudy"
    }
```

### Step 2: Registration Process

When `@registry.register()` is applied:

1. **Name Normalization**: Function name is normalized (special chars → underscores)
2. **ToolEntry Creation**: Creates a `ToolEntry` object storing:
   - Canonical name
   - Callable reference
   - Aliases
   - Description (from docstring)
3. **Registry Storage**: Stored in `_tools` dict by canonical name
4. **Alias Mapping**: Aliases stored in `_aliases` dict pointing to canonical name

### Step 3: Output Wrapping

When `@tool_output()` is applied:

1. Wraps the function to intercept return values
2. Converts return value to `ToolOutput` object
3. Applies format strings/callables:
   - `llm_format`: String representation for LLM (e.g., "Paris: 15°celsius")
   - `gui_format`: Optional format for GUI display

---

## Schema Generation

### Automatic Pydantic Model Creation

For each registered tool, a Pydantic model is automatically generated from the function signature:

```python
# Original function
def get_weather(
    location: str = Field(description="City name"),
    unit: Literal["celsius", "fahrenheit"] = "celsius"
) -> dict:
    ...

# Generated Pydantic model (conceptually)
class get_weather_Params(BaseModel):
    location: str = Field(description="City name")
    unit: Literal["celsius", "fahrenheit"] = "celsius"
```

This model is used for:
- **Validation**: Ensures arguments match expected types
- **Schema Generation**: Converts to JSON Schema for OpenAI format

### OpenAI-Compatible Schema

The `ToolEntry.to_openai_spec()` method generates:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "City name"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "default": "celsius"
        }
      },
      "required": ["location"],
      "additionalProperties": false
    }
  }
}
```

This schema is included in the system prompt to inform the model about available tools.

---

## Tool Calling Flow

### Complete Request-Response Cycle

```
┌──────────────────────────────────────────────────────────────┐
│  1. User: "What's the weather in Tokyo?"                     │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  2. ChatEngine builds messages:                              │
│     [system] (with tool contract)                            │
│     [user] "What's the weather in Tokyo?"                    │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  3. LLM generates response:                                  │
│     {"type": "tool_call", "name": "get_weather",             │
│      "arguments": {"location": "Tokyo"}}                     │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  4. ChatEngine parses JSON and calls registry.execute():    │
│     - Validates arguments via Pydantic model                 │
│     - Executes get_weather("Tokyo")                          │
│     - Returns: "Tokyo: 22°celsius, clear sky"                │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  5. Tool result added to messages:                           │
│     [user] 'Tool "get_weather" returned: Tokyo: 22°celsius,  │
│            clear sky. Now respond with type=final...'        │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  6. LLM generates final response:                            │
│     {"type": "final", "content": "The current weather in     │
│      Tokyo is 22°C with clear skies."}                       │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  7. Response returned to user:                               │
│     "The current weather in Tokyo is 22°C with clear skies." │
└──────────────────────────────────────────────────────────────┘
```

---

## Contract-Based Approach

### Why Contract-Based?

Traditional function calling relies on the model having native support (like GPT-4). Many local models don't have this, so we use a **contract**: a JSON protocol defined in the system prompt.

### The Tool Contract

```json
OUTPUT FORMAT - always respond with exactly one of these JSON formats:

1. To call a tool:
{"type": "tool_call", "name": "TOOL_NAME", "arguments": {"arg": "value"}}

2. To respond or ask a question:
{"type": "final", "content": "Your message here"}
```

### System Prompt Structure

```
You are a tool-calling assistant. Use the tools provided to get real information.

TOOLS:
{
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the current weather for a location.",
      "parameters": { ... },
      "aliases": ["weather", "w"]
    }
  ]
}

OUTPUT FORMAT - always respond with exactly one of these JSON formats:
...

IMPORTANT:
- If a tool requires arguments you don't have, ASK the user first using type=final
- Never guess or make up argument values
- type must always be exactly "tool_call" or "final"
- Output only valid JSON, no extra text
```

### Advantages

1. **Explicit**: Model knows exactly what format to use
2. **Debuggable**: Easy to see what the model attempted
3. **Flexible**: Works with any model that can generate JSON
4. **Fallback**: Model can still respond with plain text if JSON fails

---

## Multi-Turn Conversation Loop

### Loop Implementation

```python
def chat(self, user_input: str, max_tool_rounds: int = 5) -> str:
    user_msg = {"role": "user", "content": user_input}
    pending = [user_msg]

    for round_num in range(max_tool_rounds):
        messages = self._build_messages(pending)

        # Reset model state (for hybrid models)
        self.llm.reset()

        response = self.llm.create_chat_completion(messages=messages)
        assistant_msg = response["choices"][0]["message"]
        content = assistant_msg.get("content", "")

        parsed = self._parse_content_json(content)

        if parsed and parsed["type"] == "tool_call":
            # Execute tool and continue loop
            pending.append(assistant_msg)
            tool_result = self._execute_tool_by_name(
                parsed["name"],
                parsed["arguments"]
            )
            pending.append({
                "role": "user",
                "content": f'Tool "{parsed["name"]}" returned: {tool_result}...'
            })
            continue

        elif parsed and parsed["type"] == "final":
            # Final response - commit history and return
            pending.append(assistant_msg)
            self.history.extend(pending)
            return parsed["content"]

        else:
            # Plain text fallback
            pending.append(assistant_msg)
            self.history.extend(pending)
            return content

    # Max rounds reached
    self.history.extend(pending)
    return "[Max tool rounds reached]"
```

### Key Points

1. **Pending Messages**: Tool calls and results are kept in `pending` until final response
2. **History Commit**: Only committed to `self.history` when response is final
3. **Max Rounds**: Prevents infinite loops (default: 5)
4. **State Reset**: `llm.reset()` called before each generation (see next section)

### Example Multi-Turn Flow

```
Round 1:
  User: "What's the weather in Tokyo?"
  LLM: {"type": "tool_call", "name": "get_weather", "arguments": {"location": "Tokyo"}}
  Tool: "Tokyo: 22°celsius, clear sky"

Round 2:
  User: (tool result presented)
  LLM: {"type": "final", "content": "The weather in Tokyo is 22°C with clear skies."}

→ Returns to user: "The weather in Tokyo is 22°C with clear skies."
```

---

## Model State Management

### Why `llm.reset()` is Necessary

For **transformer models** (Llama, Mistral), the KV cache is typically managed automatically, but for **hybrid/Mamba models** (State Space Models), there's persistent internal state that must be cleared between generations.

### The Problem

```
WITHOUT reset():
  Generation 1 → SSM state builds up → Response 1 ✓
  Generation 2 → SSM state CORRUPTED (old + new) → llama_decode = -1 ✗

WITH reset():
  Generation 1 → Response 1 ✓
  reset() → Clear SSM state
  Generation 2 → Response 2 ✓
```

See [hybrid-model-reset.md](hybrid-model-reset.md) for a detailed explanation with diagrams.

### Implementation

```python
for round_num in range(max_tool_rounds):
    messages = self._build_messages(pending)

    # Clear model state (SSM layers for hybrid models, KV cache for transformers)
    self.llm.reset()

    response = self.llm.create_chat_completion(messages=messages)
    ...
```

This mimics what LM Studio does automatically in server mode.

---

## API Reference

### ToolRegistry

**Main Methods:**

| Method | Description |
|--------|-------------|
| `register(fn, name=None, description=None, aliases=None)` | Decorator to register a tool |
| `get(name_or_alias)` | Retrieve a ToolEntry by name or alias |
| `execute(name, arguments)` | Execute a tool with validation |
| `to_openai_tools()` | Get all tools as OpenAI-style specs |
| `to_registry_json()` | Get registry as JSON for system prompts |

### ChatEngine

**Main Methods:**

| Method | Description |
|--------|-------------|
| `__init__(model_path, registry, system_prompt=None, n_ctx=32768, n_gpu_layers=-1, verbose=False)` | Initialize engine with model and tools |
| `chat(user_input, max_tool_rounds=5)` | Send message and get response with automatic tool execution |
| `clear()` | Clear conversation history |

**Properties:**

| Property | Description |
|----------|-------------|
| `messages` | Full message list including system prompt and history |

### Tool Decorators

**@registry.register()**

```python
@registry.register(
    name: str | None = None,        # Override function name
    description: str | None = None, # Override docstring
    aliases: list[str] | None = None # Alternative names
)
```

**@tool_output()**

```python
@tool_output(
    llm_format: str | Callable[[Any], str] | None = None,  # Format for LLM
    gui_format: str | Callable[[Any], str] | None = None   # Format for GUI
)
```

Formats can be:
- String template: `"{location}: {temperature}°{unit}"`
- Callable: `lambda x: f"{x['temp']}°C"`

---

## Extending the System

### Adding a New Tool

1. **Define the function** in `l3m_backend/tools/`:

```python
@registry.register(aliases=["search", "s"])
@tool_output(llm_format=lambda x: "\n".join(x["results"]))
def web_search(
    query: str = Field(description="Search query"),
    limit: int = 5
) -> dict:
    """Search the web for information."""
    # Your implementation
    return {"query": query, "results": [...]}
```

2. **That's it!** The tool is automatically:
   - Registered in the registry
   - Schema generated from signature
   - Included in system prompt
   - Available for LLM to call

### Customizing the System Prompt

```python
engine = ChatEngine(
    model_path="./model.gguf",
    registry=registry,
    system_prompt="You are a specialized assistant for weather and calculations.",
    n_ctx=16384
)
```

### Using a Different Registry

```python
# Create a separate registry for specific tools
custom_registry = ToolRegistry()

@custom_registry.register()
def my_custom_tool(x: int) -> dict:
    """Custom tool."""
    return {"result": x * 2}

engine = ChatEngine(
    model_path="./model.gguf",
    registry=custom_registry
)
```

---

## Performance Considerations

### Context Size

- **Default**: 32K tokens
- **Why Large?**: JSON in tool contracts tokenizes inefficiently (byte-fallback for special characters)
- **Recommendation**: Use at least 16K for reliable tool calling with multiple tools

### GPU Layers

- **-1**: All layers on GPU (fastest, most VRAM)
- **0**: CPU only (slowest, no VRAM)
- **N**: Partial offload (balance between speed and VRAM)

### Model Selection

Best results with:
- Instruction-tuned models (Llama 3 Instruct, Mistral Instruct)
- Models trained on JSON/structured output
- Models with strong following capabilities

Avoid:
- Base models (not instruction-tuned)
- Very small models (<7B parameters)
- Models that struggle with JSON

---

## Troubleshooting

### Model Outputs Invalid JSON

**Symptoms:** Plain text instead of `{"type": ...}` format

**Solutions:**
1. Use a better instruction-tuned model
2. Increase temperature slightly (0.7-0.8)
3. Check that system prompt is being included

### Tool Not Found

**Symptoms:** `ToolNotFoundError: Unknown tool: xyz`

**Solutions:**
1. Use `/tools` command to list registered tools
2. Check tool name normalization (special chars → underscores)
3. Verify tool is imported in `l3m_backend/tools/`

### Model Hallucinates Tool Results

**Symptoms:** Model makes up tool responses instead of calling tools

**Solutions:**
1. Emphasize in system prompt: "NEVER make up tool results"
2. Use models better at following instructions
3. Provide examples in system prompt

### Out of Memory

**Symptoms:** CUDA out of memory or system OOM

**Solutions:**
1. Reduce context size: `--ctx 8192`
2. Reduce GPU layers: `--gpu 20`
3. Use smaller/more quantized model

---

## Summary

This system provides a **robust, flexible, contract-based tool-calling framework** for local LLMs:

1. **Tools are registered** via decorators with automatic schema generation
2. **System prompt** includes tool specs and JSON contract
3. **LLM responds** with tool calls or final answers in JSON format
4. **ChatEngine** automatically executes tools and loops until final response
5. **Model state** is reset between generations for compatibility with hybrid models

The architecture is designed to be **model-agnostic** and **transparent**, making it easy to add new tools and extend functionality.
