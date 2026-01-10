# Plan 001: Tool Calling Module

**Status:** Completed
**Date:** 2025-01-08

## Problem

Tool call parsing was fragmented across `ChatEngine` with ad-hoc methods that failed across different LLM output formats. Different models use different JSON structures for tool calls:

- Contract format: `{"type": "tool_call", "name": "...", "arguments": {...}}`
- OpenAI/Llama: `{"type": "function", "name": "...", "parameters": {...}}`
- Hermes: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`

## Solution

Created a dedicated `toolcall` module with:

1. **Unified parsing** - Single interface for all formats
2. **JSON cleaning** - Fix malformed JSON from LLM outputs
3. **Format detection** - Auto-detect which format the model is using
4. **Singleton handler** - Global instance shared across engines

## Implementation

### Files Created

| File | Purpose |
|------|---------|
| `src/l3m_backend/engine/toolcall.py` | Main module (~400 lines) |
| `src/l3m_backend/tools/test_tools.py` | Obfuscated test tools |
| `tests/test_toolcall.py` | Unit tests (58 tests) |

### Files Modified

| File | Changes |
|------|---------|
| `src/l3m_backend/engine/chat.py` | Use ToolCallHandler, removed old parsing methods |
| `tests/test_frontend.py` | Updated method names |

### Classes

```
ToolCallHandler (singleton)
├── JsonCleaner
│   └── clean() - Fix malformed JSON
└── FormatParser
    ├── detect_format() - Identify format type
    └── parse() - Extract and normalize tool call
```

### Supported Formats

| Format | Detection | Status |
|--------|-----------|--------|
| CONTRACT | `{"type": "tool_call", ...}` | Implemented |
| OPENAI | `{"type": "function", ...}` | Implemented |
| HERMES | `<tool_call>...</tool_call>` | Implemented |
| LLAMA3 | `[func(arg=val)]` | Implemented |
| MISTRAL | `[TOOL_CALLS] [...]` | Implemented |
| NATIVE | llama-cpp-python native | Implemented |

### Test Tools

Obfuscated tools to avoid training data contamination:

- `get_flumbuster(flumb, uster)` - Wraps `get_weather`
- `calculate_zorbix(zorb, bix, mode)` - Math operations

## Testing

```bash
# Run toolcall unit tests
python -m pytest tests/test_toolcall.py -v

# Run all tests
python -m pytest tests/ -v
```

### Testing with Real Models

Use `--test-tools` flag to register obfuscated test tools (get_flumbuster, calculate_zorbix).

**Recommended: HERMES format** (works with our contract):
```bash
l3m-download NousResearch/Hermes-3-Llama-3.1-8B-GGUF Hermes-3-Llama-3.1-8B.Q4_K_M.gguf
l3m-chat Hermes-3-Llama-3.1-8B.Q4_K_M.gguf --test-tools --minimal-contract --debug
```

### Model Compatibility Notes

| Model | Contract Format | Native Format | Status |
|-------|-----------------|---------------|--------|
| Hermes 8B | ✅ Works | `<tool_call>` | Recommended |
| Llama 3.2 3B | ⚠️ Partial | `[func()]` | Outputs JSON but ignores instructions |
| Mistral 7B v0.3 | ❌ No | `[TOOL_CALLS]` | Ignores tools entirely |

**Key finding**: Smaller models (3B, 7B) don't follow contract instructions reliably. Hermes models are specifically fine-tuned for tool calling and work best with custom contracts.

## Usage

```python
from l3m_backend.engine.toolcall import get_handler, ToolCallFormat

handler = get_handler()

# Parse tool call from any format
tool_call = handler.parse(content)
if tool_call:
    print(f"Tool: {tool_call.name}")
    print(f"Args: {tool_call.arguments}")
    print(f"Format: {tool_call.format}")

# Parse native llama-cpp-python chunks
tool_calls = handler.parse_native_chunks(chunks)

# Format result for model
result_str = handler.format_result("tool_name", result, ToolCallFormat.HERMES)
```

## E2E Testing (Completed 2025-01-09)

Automated E2E test suite validates Hermes 8B with obfuscated test tools.

### Test Script

`scripts/e2e_test_hermes.py` - Runs 5 test cases:

| Test | Tool | Description |
|------|------|-------------|
| flumbuster_basic | get_flumbuster | Basic tool call with location |
| flumbuster_with_uster | get_flumbuster | Tool call with optional arg |
| zorbix_add | calculate_zorbix | Math operation (default mode) |
| zorbix_multiply | calculate_zorbix | Math with explicit mode |
| no_tool_greeting | (none) | Negative test - no tool needed |

### Results

| Contract Type | Tests Passed | Time |
|--------------|--------------|------|
| Full (2228 chars) | 5/5 | ~158s |
| Minimal (634 chars) | 5/5 | ~67s |

### Contract Improvement

The minimal contract was improved to use balanced examples:

**Before** (ask-first example):
```
EXAMPLE:
User: "Use tool X"
You: What value should I use for the required argument?
```

**After** (balanced examples):
```
EXAMPLES:
User: "Weather in Paris"
You: {"type": "tool_call", "name": "get_weather", "arguments": {"location": "Paris"}}

User: "What's the weather?"
You: Which city would you like the weather for?
```

This fixed the over-cautious behavior where the model asked for clarification even when arguments were provided.

## Future Work

- Investigate llama-cpp-python native tool calling mode for better model support
- Test with larger models (70B) for better instruction following
- CI integration for E2E tests (requires GPU)
