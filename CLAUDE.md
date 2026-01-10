# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important

DO NOT CHANGE THE BACKEND CORE LOGIC.

## Overview

LLM tool-calling system packaged as `l3m-backend`. Installable via `pip install git+...`.

Uses a contract-based approach to tool calling that works with any model capable of generating JSON.

## Environment

```bash
source ~/.venvs/l3m/bin/activate
pip install -e '.[dev,llm]'
```

## Running

```bash
# Download a model
l3m-download --preset llama3.2-1b

# List available models
l3m-chat --list

# Chat REPL (auto-selects model from ~/.l3m/models/)
l3m-chat

# Chat with specific model
l3m-chat Llama-3.2-1B-Instruct-Q4_K_M.gguf

# List available tools
l3m-tools list
l3m-tools schema
l3m-tools info get_weather

# Enable bash completion
source <(l3m-completion bash)

# Run tests
python -m pytest tests/
```

## Package Structure

```
src/l3m_backend/
├── core/           - Tool Registry (Pydantic schemas, OpenAI-compatible specs)
│   ├── registry.py    - ToolRegistry class
│   ├── datamodels.py  - ToolEntry, ToolOutput, ToolResult
│   ├── exceptions.py  - ToolError, ToolNotFoundError, etc.
│   ├── decorators.py  - @tool_output decorator
│   └── helpers.py     - Utility functions
├── engine/         - Chat Engine (llama-cpp-python integration)
│   ├── chat.py        - ChatEngine class
│   └── contract.py    - TOOL_CONTRACT_TEMPLATE
├── tools/          - Built-in tool definitions
│   ├── weather.py     - get_weather
│   ├── time.py        - get_time
│   ├── calculator.py  - calculate
│   ├── information.py - wikipedia, define_word, etc.
│   ├── productivity.py - note, todo, reminder, timer
│   ├── development.py - run_python, shell_cmd, etc.
│   └── utilities.py   - random, uuid, hash, base64, json
├── cli/            - CLI entry points
│   ├── chat_repl.py   - l3m-chat command
│   ├── tools_cli.py   - l3m-tools command
│   ├── download.py    - l3m-download command
│   ├── completion.py  - l3m-completion command
│   ├── _repl.py       - Feature-rich REPL (prompt_toolkit)
│   └── _simple_repl.py - Basic REPL (input())
└── scripts/        - Development/diagnostic scripts
    ├── download_gguf.py  - Model downloader
    ├── chat_repl.py      - Standalone chat REPL
    └── diagnose_*.py     - Diagnostic utilities
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `l3m-chat` | Interactive chat REPL with tool calling |
| `l3m-tools` | Tool inspection (list, schema, info) |
| `l3m-download` | Download GGUF models from Hugging Face |
| `l3m-completion` | Generate shell completion scripts |

## Default Paths

- **Models**: `~/.l3m/models/`
- **Chat history**: `~/.l3m/prompt_history`

## Key Implementation Details

**Contract-Based Tool Calling**: Tools respond with JSON:
- `{"type": "tool_call", "name": "...", "arguments": {...}}`
- `{"type": "final", "content": "..."}`

**Hybrid Model Reset**: Mamba/SSM models (like granite-hybrid) require `llm.reset()` before each generation call. See [docs/hybrid-model-reset.md](docs/hybrid-model-reset.md).

**Context Size**: Default 32K tokens. JSON in tool contracts tokenizes inefficiently (byte-fallback), so large context is needed even for small prompts.

**GPU Acceleration (Metal)**: On macOS, install llama-cpp-python with Metal support:
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```
Use `/model` command to verify backend. See [docs/installation.md](docs/installation.md) for details.
