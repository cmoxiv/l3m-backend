# l3m-backend User Manual

A comprehensive guide to using l3m-backend, a lightweight tool-calling system for local LLMs.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [CLI Commands](#cli-commands)
5. [REPL Interface](#repl-interface)
6. [Built-in Tools](#built-in-tools)
7. [Creating Custom Tools](#creating-custom-tools)
8. [Session Management](#session-management)
9. [Configuration](#configuration)
10. [Python API](#python-api)
11. [Troubleshooting](#troubleshooting)
12. [FAQ](#faq)

---

## Introduction

l3m-backend is a tool-calling system that enables local LLMs to execute functions and tools. Unlike cloud-based solutions, it runs entirely on your machine using llama-cpp-python.

### Key Features

- **Contract-based tool calling**: Works with any model capable of generating JSON
- **No native function-calling required**: Uses a learned protocol instead of model-specific features
- **Rich REPL interface**: Command history, tab completion, multi-line input
- **Session persistence**: Save and resume conversations
- **Extensible**: Create custom tools easily
- **Model-agnostic**: Works with Llama, Mistral, Qwen, Phi, and other GGUF models

### How It Works

The system uses a contract-based approach where the LLM responds with JSON:

```json
{"type": "tool_call", "name": "get_weather", "arguments": {"location": "Paris"}}
```

The system executes the tool and provides results back to the model, which then generates a final response:

```json
{"type": "final", "content": "The weather in Paris is 22C and sunny."}
```

---

## Installation

### Requirements

- Python 3.9 or higher
- pip package manager

### Installation Options

```bash
# Basic installation (no LLM support)
pip install git+https://github.com/mo/l3m-backend.git

# With LLM support (recommended)
pip install 'git+https://github.com/mo/l3m-backend.git[llm]'

# With enhanced REPL (recommended)
pip install 'git+https://github.com/mo/l3m-backend.git[repl]'

# Full installation (all features)
pip install 'git+https://github.com/mo/l3m-backend.git[llm,repl]'
```

### Verifying Installation

```bash
l3m-chat --help
l3m-tools --help
l3m-download --help
```

---

## Getting Started

### Step 1: Initialize Configuration

```bash
l3m-init
```

This creates:
- `~/.l3m/` - Main configuration directory
- `~/.l3m/config.json` - Configuration file
- `~/.l3m/models/` - Directory for model files
- `~/.l3m/sessions/` - Directory for saved sessions

### Step 2: Download a Model

```bash
# View available presets
l3m-download --list-presets

# Download a small model for testing
l3m-download --preset llama3.2-1b

# Or download a larger model for better quality
l3m-download --preset mistral-7b
```

### Step 3: Start Chatting

```bash
l3m-chat
```

### Example Conversation

```
You: What's the weather in Tokyo?
Assistant: The current weather in Tokyo is 18C with partly cloudy skies.

You: Calculate 15% tip on $47.50
Assistant: A 15% tip on $47.50 would be $7.13, for a total of $54.63.
```

---

## CLI Commands

### l3m-init

Initialize l3m-backend configuration and directories.

```bash
l3m-init              # Initialize (adds missing config keys, preserves existing)
l3m-init --force      # Overwrite existing config completely
l3m-init --quiet      # Suppress output
```

**Note:** Running `l3m-init` on an existing installation only adds new configuration keys while preserving your existing settings. Use `--force` to reset everything to defaults.

### l3m-chat

Interactive chat REPL with tool calling.

```bash
l3m-chat                      # Auto-detect model
l3m-chat model.gguf           # Use specific model
l3m-chat --list               # List available models
l3m-chat --ctx 8192           # Set context size
l3m-chat --gpu -1             # Use all GPU layers
l3m-chat -v                   # Verbose mode
```

### l3m-tools

Manage and inspect tools.

```bash
l3m-tools list                # List all tools
l3m-tools list -v             # List with descriptions
l3m-tools info get_weather    # Show tool details
l3m-tools schema              # Show all tool schemas (JSON)
l3m-tools schema get_weather  # Show specific tool schema
l3m-tools create my_tool      # Create new tool scaffold
l3m-tools create my_tool --wrapper  # Create wrapper tool
```

### l3m-download

Download GGUF models from Hugging Face.

```bash
l3m-download --list-presets           # Show available presets
l3m-download --preset llama3.2-1b     # Download using preset
l3m-download --repo org/model --file model.gguf  # Direct download
l3m-download --preset llama3.2-1b --output /path/to/dir  # Custom location
```

### l3m-completion

Generate shell completion scripts.

```bash
l3m-completion bash           # Generate bash completion
l3m-completion zsh            # Generate zsh completion
l3m-completion fish           # Generate fish completion
source <(l3m-completion bash) # Enable bash completion
```

---

## REPL Interface

The l3m-chat REPL provides a rich interactive environment with multiple command types.

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Tab` | Auto-complete commands |
| `Ctrl+R` | Search command history |
| `Ctrl+C` | Cancel current input |
| `Ctrl+D` (twice) | Exit REPL |
| Arrow keys | Navigate history |

### Slash Commands (/)

Pure commands that don't modify conversation history:

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation, transcript, and summaries |
| `/tools` | List available tools |
| `/system` | Show system prompt |
| `/schema` | Show tool schema (OpenAI format) |
| `/contract` | Show full system message with tools contract |
| `/history` | Show conversation history |
| `/undo` | Remove last user+assistant exchange |
| `/pop [n]` | Remove last n messages (default: 1) |
| `/context` | Show context usage estimate |
| `/model` | Show model info |
| `/config` | Show current configuration |
| `/help` | Show help message |
| `/quit` | Exit (also `/exit`, `/q`) |

### Shell Commands (!)

Run shell commands directly. Output is displayed but NOT added to conversation history:

```
!ls -la
!git status
!python --version
```

### Magic Commands (%)

Magic commands add their input and output to the conversation history, making them visible to the LLM:

| Command | Description |
|---------|-------------|
| `%!<cmd>` | Run shell command (adds to history) |
| `%tool <name> [json]` | Invoke tool directly |
| `%load <file>` | Load file content into conversation |
| `%time` | Report current time |
| `%save <file>` | Save conversation to file |
| `%edit-response` | Edit last assistant response in $VISUAL |

**Examples:**

```
# Run shell and add output to conversation
%!ls -la src/

# Invoke a tool directly
%tool get_weather {"location": "Paris"}
%tool calculate {"expression": "2+2"}

# Load a file for the LLM to see
%load README.md

# Save conversation
%save conversation.md
```

### Session Commands

Commands for managing persistent sessions:

| Command | Description |
|---------|-------------|
| `/session` | Show current session info |
| `/sessions [query]` | List or search sessions |
| `/session-save [title]` | Save session (auto-generates title) |
| `/session-load <id>` | Load session as context |
| `/session-title [title]` | Set or generate session title |
| `/session-tag <tag>` | Add tag to session |

---

## Built-in Tools

### Weather & Time

| Tool | Description | Example |
|------|-------------|---------|
| `get_weather` | Get weather for a location | "What's the weather in Paris?" |
| `get_time` | Get current date and time | "What time is it?" |

### Information

| Tool | Description | Example |
|------|-------------|---------|
| `wikipedia` | Fetch Wikipedia summaries | "Tell me about quantum computing" |
| `define_word` | Look up word definitions | "Define 'ephemeral'" |
| `unit_convert` | Convert between units | "Convert 100 km to miles" |
| `currency_convert` | Convert currencies | "Convert 100 USD to EUR" |
| `web_search` | Search the web | "Search for Python tutorials" |

### Development

| Tool | Description | Example |
|------|-------------|---------|
| `run_python` | Execute Python code | "Run this Python code: print(2+2)" |
| `shell_cmd` | Execute shell commands | "List files in /tmp" |
| `read_file` | Read file contents | "Read the contents of config.json" |
| `write_file` | Write to a file | "Create a file called test.txt" |
| `http_request` | Make HTTP requests | "Fetch https://api.example.com" |

### Productivity

| Tool | Description | Example |
|------|-------------|---------|
| `note` | Save/retrieve notes | "Save a note: meeting at 3pm" |
| `todo` | Manage todo items | "Add todo: review PR" |
| `reminder` | Set reminders | "Remind me in 30 minutes" |
| `timer` | Set timers | "Start a 5 minute timer" |

### Utilities

| Tool | Description | Example |
|------|-------------|---------|
| `calculate` | Evaluate math expressions | "Calculate 15% of 85" |
| `random_number` | Generate random numbers | "Give me a random number 1-100" |
| `uuid_generate` | Generate UUIDs | "Generate a UUID" |
| `hash_text` | Hash text (MD5, SHA256) | "Hash 'hello' with SHA256" |
| `base64_encode` | Base64 encode/decode | "Encode 'hello' in base64" |
| `json_format` | Format/minify JSON | "Pretty print this JSON" |

---

## Creating Custom Tools

### Quick Start

```bash
# Create a basic tool scaffold
l3m-tools create my_tool

# Create a wrapper for a shell command
l3m-tools create docker-ps --wrapper
```

### Tool Structure

Tools are Python modules in `~/.l3m/tools/<tool_name>/__init__.py`:

```python
# ~/.l3m/tools/my_tool/__init__.py
from typing import Any
from l3m_backend.tools import registry
from l3m_backend.core import tool_output

@registry.register(aliases=["mt"])
@tool_output(llm_format="{result}")
def my_tool(input: str) -> dict[str, Any]:
    """Process the input and return a result.
    
    Args:
        input: The text to process
    """
    result = input.upper()  # Your logic here
    return {"result": result, "input": input}
```

### Using Pydantic Field for Descriptions

```python
from pydantic import Field

@registry.register()
@tool_output(llm_format="{city}: {temp}C")
def get_temperature(
    city: str = Field(description="City name to check"),
    unit: str = Field(default="celsius", description="Temperature unit")
) -> dict:
    """Get the temperature for a city."""
    return {"city": city, "temp": 22, "unit": unit}
```

### Tool Output Formatting

The `@tool_output` decorator controls how results are presented to the LLM:

```python
# String template formatting
@tool_output(llm_format="{location}: {temperature}C, {condition}")

# Lambda/callable formatting
@tool_output(llm_format=lambda x: f"Result: {x['value']}")

# Separate formats for LLM and GUI
@tool_output(
    llm_format="{result}",
    gui_format="<div class='result'>{result}</div>"
)
```

---

## Session Management

Sessions allow you to save and resume conversations.

### Saving Sessions

```
# Auto-generate title from conversation
/session-save

# Set a custom title
/session-save "Debugging the login flow"
```

### Finding Sessions

```
# List all sessions
/sessions

# Search sessions
/sessions debugging
/sessions python api
```

### Loading Sessions

```
# Load a session by ID (first 8 chars is enough)
/session-load a1b2c3d4

# View current session
/session
```

### Resuming with Smaller Context

When resuming a session with a smaller context size than when it was saved, l3m-chat will:
1. Automatically trim older messages to fit the new context
2. Generate a summary of the trimmed content
3. Save the updated session

You'll see a notification like:
```
History trimmed to fit context (8192 tokens):
  Removed 15 messages, kept 8
  3500 -> 1800 tokens
  Generating summary of trimmed content...
  Session updated.
```

### Tagging Sessions

```
# Add a tag
/session-tag debugging
/session-tag important

# View tags (shown in /session output)
/session
```

---

## Configuration

### Configuration File

Located at `~/.l3m/config.json`:

```json
{
    "default_model": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "ctx": 8192,
    "gpu": -1,
    "verbose": false,
    "temperature": 0.1,
    "stop_tokens": ["<|end|>", "<|eot_id|>", "<|im_end|>"],
    "system_prompt": "You are a helpful assistant with access to tools."
}
```

### Configuration Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_model` | string | null | Default model file to use |
| `ctx` | int | 8192 | Context size in tokens |
| `gpu` | int | -1 | GPU layers (-1 = all) |
| `temperature` | float | 0.1 | Generation temperature |
| `stop_tokens` | list | [...] | Stop tokens for generation |
| `system_prompt` | string | "..." | Default system prompt |
| `verbose` | bool | false | Verbose output |
| `incognito` | bool | false | Don't save sessions |
| `auto_resume` | bool | false | Auto-resume last session in CWD |

### Viewing Configuration

```
/config
```

### Setting Configuration

```bash
l3m-chat --set-config default_model=my-model.gguf
l3m-chat --set-config context_size=16384
```

---

## Python API

### Basic Usage

```python
from l3m_backend import ToolRegistry, tool_output

# Create a registry
registry = ToolRegistry()

# Register a tool
@registry.register(aliases=["calc"])
@tool_output(llm_format="Result: {result}")
def calculate(expression: str) -> dict:
    """Evaluate a math expression."""
    return {"expression": expression, "result": eval(expression)}

# Use with ChatEngine
from l3m_backend import ChatEngine

engine = ChatEngine("./model.gguf", registry)
response = engine.chat("What is 2 + 2?")
print(response)
```

### Using the Built-in Registry

```python
from l3m_backend.tools import registry
from l3m_backend import ChatEngine

# registry already has all built-in tools loaded
engine = ChatEngine("./model.gguf", registry)
response = engine.chat("What's the weather in Tokyo?")
```

### ChatEngine API

```python
engine = ChatEngine(
    model_path="./model.gguf",
    registry=registry,
    system_prompt="You are a helpful assistant.",
    n_ctx=32768,        # Context size
    n_gpu_layers=-1,    # GPU layers (-1 = all)
    verbose=False,
    stop_tokens=["<|end|>", "<|eot_id|>"]  # Stop generation tokens
)

# Chat (blocking)
response = engine.chat("Hello!", max_tool_rounds=5)

# Chat with streaming (yields tokens as they're generated)
for token in engine.chat("Hello!", stream=True):
    print(token, end="", flush=True)

# Clear history (also clears partition context)
engine.clear()

# Access messages
print(engine.messages)
print(engine.history)
```

---

## Troubleshooting

### Model Not Found

```
Error: No models found in ~/.l3m/models/
```

**Solution:** Download a model:
```bash
l3m-download --preset llama3.2-1b
```

### Out of Memory

```
CUDA out of memory
```

**Solutions:**
1. Reduce context size: `l3m-chat --ctx 8192`
2. Use fewer GPU layers: `l3m-chat --gpu 20`
3. Use a smaller/more quantized model

### Model Outputs Invalid JSON

The model responds with plain text instead of JSON format.

**Solutions:**
1. Use a better instruction-tuned model (Llama 3 Instruct, Mistral Instruct)
2. Increase context size for more reliable JSON generation
3. Check `/system` to ensure the system prompt includes the tool contract

### Tool Not Found

```
ToolNotFoundError: Unknown tool: xyz
```

**Solutions:**
1. Check registered tools: `/tools`
2. Verify tool name (use `/schema` to see exact names)
3. Check that user tools in `~/.l3m/tools/` have valid `__init__.py`

### Hybrid Model Issues (llama_decode = -1)

Some models (like IBM Granite) have SSM/Mamba layers that require state reset.

**Solution:** This is handled automatically by l3m-backend. If you see this error, ensure you're using the latest version.

---

## FAQ

### What models work best?

Instruction-tuned models with good JSON generation capabilities work best:
- Llama 3.2 Instruct (1B, 3B)
- Mistral 7B Instruct
- Qwen 2.5 Instruct
- Phi 3.5 Mini

### How much VRAM do I need?

| Model Size | Quantization | VRAM Required |
|------------|--------------|---------------|
| 1B | Q4_K_M | ~1 GB |
| 3B | Q4_K_M | ~2.5 GB |
| 7B | Q4_K_M | ~5 GB |
| 8B | Q4_K_M | ~6 GB |

### Can I use this without a GPU?

Yes! Set `--gpu 0` to run on CPU only. It will be slower but works.

### How do I add my own API tools?

Create a custom tool that makes HTTP requests:

```python
@registry.register()
@tool_output(llm_format="{response}")
def my_api(query: str) -> dict:
    """Call my custom API."""
    import requests
    resp = requests.get(f"https://api.example.com?q={query}")
    return {"response": resp.json()}
```

### Can sessions be exported?

Yes! Use `%save <file>` to export the current conversation to a markdown file.

### How do I disable specific tools?

Create a custom registry with only the tools you want:

```python
from l3m_backend import ToolRegistry, ChatEngine
from l3m_backend.tools.weather import get_weather
from l3m_backend.tools.time import get_time

registry = ToolRegistry()
registry.register()(get_weather)
registry.register()(get_time)

engine = ChatEngine("./model.gguf", registry)
```

---

## Additional Resources

- [Architecture Documentation](architecture.md) - Deep dive into how the system works
- [Hybrid Model Reset](hybrid-model-reset.md) - Understanding SSM model state management
- [Code Review Report](code-review.md) - Security and code quality analysis
- [Test Report](test-report.md) - Test coverage and results

---

*Generated for l3m-backend v0.1.0*
