# Hermes Tool Calling Format

Hermes models (from NousResearch) use a specific format for tool calling based on ChatML with XML tags.

## Overview

Hermes tool calling uses:
- **ChatML tokens**: `<|im_start|>` and `<|im_end|>` for message boundaries
- **XML tags**: `<tools>`, `<tool_call>`, `<tool_response>` for tool interactions

## System Prompt Format

Tools are defined in the system prompt within `<tools>` tags:

```
<|im_start|>system
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.

<tools>
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City name"
          }
        },
        "required": ["location"]
      }
    }
  }
]
</tools>

For each function call, return a JSON object within <tool_call></tool_call> XML tags.
<|im_end|>
```

## Tool Call Format

When the model wants to call a tool, it outputs:

```
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>
```

### Variations

The model may also output:

```
<tool_call>
{'name': 'get_weather', 'arguments': {'location': 'Paris'}}
</tool_call>
```

Note: Single quotes are common - the `JsonCleaner` handles this.

## Tool Response Format

After executing the tool, provide the result in `<tool_response>` tags:

```
<|im_start|>user
<tool_response>
{"temperature": 18, "conditions": "partly cloudy"}
</tool_response>
<|im_end|>
```

Or as plain text:

```
<|im_start|>user
<tool_response>
Paris: 18°C, partly cloudy
</tool_response>
<|im_end|>
```

## Multiple Tool Calls

Hermes supports multiple tool calls in a single response:

```
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "London"}}
</tool_call>
```

## Complete Conversation Example

```
<|im_start|>system
You have access to tools. Use <tool_call> tags for function calls.

<tools>
[{"type": "function", "function": {"name": "get_weather", ...}}]
</tools>
<|im_end|>

<|im_start|>user
What's the weather in Paris?
<|im_end|>

<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>
<|im_end|>

<|im_start|>user
<tool_response>
Paris: 18°C, partly cloudy
</tool_response>
<|im_end|>

<|im_start|>assistant
The weather in Paris is currently 18°C with partly cloudy skies.
<|im_end|>
```

## Parsing in l3m-backend

The `ToolCallHandler` detects Hermes format by looking for `<tool_call>` tags:

```python
from l3m_backend.engine.toolcall import get_handler, ToolCallFormat

handler = get_handler()

content = '<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>'
tool_call = handler.parse(content)

print(tool_call.name)       # "get_weather"
print(tool_call.arguments)  # {"location": "Paris"}
print(tool_call.format)     # ToolCallFormat.HERMES
```

## Formatting Results

Use `format_result()` to create properly formatted tool responses:

```python
result = handler.format_result(
    "get_weather",
    {"temperature": 18, "conditions": "partly cloudy"},
    ToolCallFormat.HERMES
)
# Returns: "<tool_response>\n{\"temperature\": 18, \"conditions\": \"partly cloudy\"}\n</tool_response>"
```

## Recommended Models

| Model | Size | HuggingFace |
|-------|------|-------------|
| Hermes-3-Llama-3.1-8B | ~4.5GB (Q4) | NousResearch/Hermes-3-Llama-3.1-8B-GGUF |
| Hermes-2-Pro-Mistral-7B | ~4.4GB (Q4) | NousResearch/Hermes-2-Pro-Mistral-7B-GGUF |

## Download and Test

```bash
# Download model
l3m-download NousResearch/Hermes-3-Llama-3.1-8B-GGUF Hermes-3-Llama-3.1-8B.Q4_K_M.gguf

# Test with minimal contract
l3m-chat Hermes-3-Llama-3.1-8B.Q4_K_M.gguf --minimal-contract --debug
```

## E2E Testing

An automated E2E test suite validates Hermes with obfuscated test tools:

```bash
# Run E2E tests
source ~/.venvs/l3m/bin/activate
python scripts/e2e_test_hermes.py
```

### Test Cases

| Test | Tool | Description |
|------|------|-------------|
| flumbuster_basic | get_flumbuster | Basic tool call with location |
| flumbuster_with_uster | get_flumbuster | Tool call with optional arg |
| zorbix_add | calculate_zorbix | Math operation (default mode) |
| zorbix_multiply | calculate_zorbix | Math with explicit mode |
| no_tool_greeting | (none) | Negative test - no tool needed |

### Contract Performance

| Contract Type | Tests Passed | Time | Notes |
|--------------|--------------|------|-------|
| Full (2228 chars) | 5/5 | ~158s | More examples, slower |
| Minimal (634 chars) | 5/5 | ~67s | Faster, recommended |

The minimal contract uses balanced examples showing both "call tool" and "ask for clarification" behaviors.

## Common Issues

### 1. Single Quotes

Hermes often outputs single quotes instead of double quotes:
```
{'name': 'test'}  # Instead of {"name": "test"}
```

**Solution**: `JsonCleaner` automatically converts these.

### 2. Python Booleans

Hermes may output Python-style booleans:
```
{"enabled": True, "value": None}  # Instead of true/null
```

**Solution**: `JsonCleaner` converts `True`→`true`, `False`→`false`, `None`→`null`.

### 3. Surrounding Text

Hermes may include explanatory text around tool calls:
```
Let me check the weather for you.
<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>
I'll get that information now.
```

**Solution**: `FormatParser` extracts content between tags, ignoring surrounding text.

## References

- [Hermes Function Calling Repository](https://github.com/NousResearch/Hermes-Function-Calling)
- [Hermes-3 Model Card](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)
