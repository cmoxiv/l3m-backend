# Configuration Reference

Complete reference for l3m-backend configuration options.

---

## Table of Contents

1. [Configuration File](#configuration-file)
2. [Configuration Options](#configuration-options)
3. [Environment Variables](#environment-variables)
4. [Command-Line Overrides](#command-line-overrides)
5. [MCP Server Configuration](#mcp-server-configuration)
6. [Tool Contract Configuration](#tool-contract-configuration)

---

## Configuration File

### Location

```
~/.l3m/config.json
```

### Creating Default Config

```bash
l3m-init
```

This creates the configuration file with default values. Running `l3m-init` on an existing installation only adds missing keys while preserving existing settings.

### Example Configuration

```json
{
  "_comment": "l3m-backend configuration file",
  "default_model": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
  "ctx": 8192,
  "gpu": -1,
  "verbose": false,
  "show_warnings": false,
  "simple": false,
  "no_native_tools": false,
  "no_flash_attn": false,
  "minimal_contract": false,
  "incognito": false,
  "auto_resume": false,
  "summary_ctx": 0,
  "transcript_ctx": 0,
  "temperature": 0.1,
  "system_prompt": "You are a helpful assistant with access to tools.",
  "chat_format": null,
  "max_tokens": null,
  "stop_tokens": ["<|end|>", "<|eot_id|>", "<|im_end|>"]
}
```

---

## Configuration Options

### Model Settings

#### default_model

Default model file to use when none is specified.

| Property | Value |
|----------|-------|
| Type | `string` or `null` |
| Default | `null` (auto-detect) |

```json
"default_model": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

When `null`, l3m-chat auto-selects from `~/.l3m/models/`.

#### ctx

Context size in tokens.

| Property | Value |
|----------|-------|
| Type | `integer` |
| Default | `8192` |
| Range | 512 - 131072 |

```json
"ctx": 16384
```

Larger contexts allow longer conversations but use more memory.

#### gpu

Number of model layers to offload to GPU.

| Property | Value |
|----------|-------|
| Type | `integer` |
| Default | `-1` |

| Value | Effect |
|-------|--------|
| `-1` | All layers on GPU (recommended) |
| `0` | CPU only |
| `N` | N layers on GPU |

```json
"gpu": -1
```

#### verbose

Enable verbose llama.cpp output.

| Property | Value |
|----------|-------|
| Type | `boolean` |
| Default | `false` |

```json
"verbose": true
```

Shows tokenization details, model loading info, etc.

#### show_warnings

Show Metal/GPU initialization warnings.

| Property | Value |
|----------|-------|
| Type | `boolean` |
| Default | `false` |

```json
"show_warnings": true
```

When `false`, stderr is suppressed during model loading.

### Chat Settings

#### temperature

Generation temperature (creativity/randomness).

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.1` |
| Range | 0.0 - 2.0 |

| Value | Effect |
|-------|--------|
| 0.0 | Deterministic (most focused) |
| 0.1 | Slightly varied (default) |
| 0.7 | Creative |
| 1.0+ | Very random |

```json
"temperature": 0.7
```

#### system_prompt

Custom system prompt for the assistant.

| Property | Value |
|----------|-------|
| Type | `string` |
| Default | See below |

Default system prompt:
```
You are a helpful assistant with access to tools. Answer concisely and follow instructions efficiently.

When you need information:
1. Use available tools to find answers
2. If a tool needs arguments you don't have, ask the user
3. If a tool fails, inform the user and ask for clarification
4. For questions beyond your knowledge cutoff, suggest enabling the web_search tool
```

Custom example:
```json
"system_prompt": "You are a Python expert assistant. Focus on code quality and best practices."
```

#### chat_format

Override the chat template format.

| Property | Value |
|----------|-------|
| Type | `string` or `null` |
| Default | `null` (auto-detect) |

Common formats:
- `chatml` - ChatML format
- `llama-2` - Llama 2 format
- `mistral` - Mistral format

```json
"chat_format": "chatml"
```

Usually not needed - llama.cpp auto-detects from model metadata.

#### max_tokens

Maximum tokens to generate per response.

| Property | Value |
|----------|-------|
| Type | `integer` or `null` |
| Default | `null` (unlimited) |

```json
"max_tokens": 2048
```

#### stop_tokens

Tokens that signal end of generation.

| Property | Value |
|----------|-------|
| Type | `array[string]` |
| Default | `["<|end|>", "<|eot_id|>", "<|im_end|>"]` |

```json
"stop_tokens": ["<|end|>", "<|eot_id|>", "<|im_end|>", "</s>"]
```

Model-specific; adjust if responses are cut off or don't stop.

### Tool Settings

#### no_native_tools

Disable native function calling, use contract-only.

| Property | Value |
|----------|-------|
| Type | `boolean` |
| Default | `false` |

```json
"no_native_tools": true
```

When `true`, tools only work via JSON contract in system prompt.

#### minimal_contract

Use minimal tool contract (smaller context footprint).

| Property | Value |
|----------|-------|
| Type | `boolean` |
| Default | `false` |

```json
"minimal_contract": true
```

Compact tool list format instead of full JSON schemas. Useful for smaller context windows or models that struggle with large system prompts.

#### no_flash_attn

Disable flash attention.

| Property | Value |
|----------|-------|
| Type | `boolean` |
| Default | `false` |

```json
"no_flash_attn": true
```

Flash attention is faster and more memory efficient. Only disable if you encounter issues.

### REPL Settings

#### simple

Use simple REPL (basic input()) instead of prompt_toolkit.

| Property | Value |
|----------|-------|
| Type | `boolean` |
| Default | `false` |

```json
"simple": true
```

Use when prompt_toolkit causes issues or in minimal environments.

### Session Settings

#### incognito

Default to incognito mode (sessions stored in /tmp).

| Property | Value |
|----------|-------|
| Type | `boolean` |
| Default | `false` |

```json
"incognito": true
```

Sessions in incognito mode are stored in `/tmp/l3m-sessions/` and cleared on reboot.

#### auto_resume

Automatically resume session from CWD without prompting.

| Property | Value |
|----------|-------|
| Type | `boolean` |
| Default | `false` |

```json
"auto_resume": true
```

When `true`, if a `.l3m-session` symlink exists in the working directory, the session is resumed automatically.

#### summary_ctx

Context tokens allocated for session summaries.

| Property | Value |
|----------|-------|
| Type | `integer` |
| Default | `0` |

| Value | Effect |
|-------|--------|
| `0` | Disabled |
| `> 0` | Fixed token budget |
| `-1` | Add to system default |

```json
"summary_ctx": 2048
```

#### transcript_ctx

Context tokens allocated for transcript excerpts.

| Property | Value |
|----------|-------|
| Type | `integer` |
| Default | `0` |

```json
"transcript_ctx": 1024
```

---

## Environment Variables

### LLAMA_CPP_LIB

Path to llama.cpp shared library.

```bash
export LLAMA_CPP_LIB=/path/to/libllama.so
```

### L3M_MODELS_DIR

Override default models directory.

```bash
export L3M_MODELS_DIR=/custom/path/to/models
```

Default: `~/.l3m/models/`

### L3M_CONFIG_DIR

Override configuration directory.

```bash
export L3M_CONFIG_DIR=/custom/path/to/config
```

Default: `~/.l3m/`

### VISUAL / EDITOR

Editor for `%edit-response` command.

```bash
export VISUAL=code   # VS Code
export VISUAL=vim    # Vim
export EDITOR=nano   # Nano (fallback)
```

---

## Command-Line Overrides

Command-line arguments override config file values.

### l3m-chat

```bash
l3m-chat [model] [options]
```

| Option | Config Key | Description |
|--------|------------|-------------|
| `--ctx N` | `ctx` | Context size |
| `--gpu N` | `gpu` | GPU layers |
| `-v, --verbose` | `verbose` | Verbose output |
| `--show-warnings` | `show_warnings` | Show GPU warnings |
| `--simple` | `simple` | Simple REPL |
| `--no-native-tools` | `no_native_tools` | Disable native tools |
| `--no-flash-attn` | `no_flash_attn` | Disable flash attention |
| `--minimal-contract` | `minimal_contract` | Minimal tool contract |
| `--incognito` | `incognito` | Incognito mode |
| `--temperature T` | `temperature` | Generation temperature |
| `--system PROMPT` | `system_prompt` | System prompt |
| `--chat-format FMT` | `chat_format` | Chat format |
| `--max-tokens N` | `max_tokens` | Max tokens |

### Setting Config via CLI

```bash
# Set a config value
l3m-chat --set-config ctx=16384
l3m-chat --set-config temperature=0.7
l3m-chat --set-config "system_prompt=You are a Python expert."

# Multiple values
l3m-chat --set-config ctx=16384 --set-config gpu=32
```

---

## MCP Server Configuration

### Location

```
~/.l3m/mcp-servers.json
```

### Format

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-filesystem", "/path/to/allowed/dir"],
      "auto_connect": true
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-api-key"
      },
      "auto_connect": false
    },
    "remote-server": {
      "url": "http://localhost:8080/mcp",
      "auto_connect": false
    }
  }
}
```

### Server Configuration Options

#### Local Server (stdio)

```json
{
  "command": "npx",
  "args": ["-y", "@anthropic/mcp-server-filesystem", "/allowed/path"],
  "env": {
    "ENV_VAR": "value"
  },
  "auto_connect": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `command` | `string` | Command to run |
| `args` | `array[string]` | Command arguments |
| `env` | `object` | Environment variables |
| `auto_connect` | `boolean` | Connect on startup |

#### Remote Server (HTTP/SSE)

```json
{
  "url": "http://localhost:8080/mcp",
  "auto_connect": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `url` | `string` | Server URL |
| `auto_connect` | `boolean` | Connect on startup |

### Managing MCP Servers

```bash
# In REPL
/mcp                    # Show MCP status
/mcp connect <name>     # Connect to server
/mcp disconnect <name>  # Disconnect from server
/mcp tools [name]       # List tools from server(s)
```

---

## Tool Contract Configuration

### Location

```
~/.l3m/tool-contract.txt
```

### Custom Contract Template

You can customize the tool calling contract. The template uses `{registry_json}` placeholder for tool definitions.

Default contract:
```
You have access to tools to help you with your response.

TOOL_REGISTRY_JSON:
{registry_json}

OUTPUT CONTRACT (STRICT):
- If you decide to call a tool, output ONLY one JSON object (no extra text):
  {"type": "tool_call", "name": "<tool name or alias>", "arguments": {...}}

- Otherwise output ONLY:
  {"type": "final", "content": "..."}

RULES:
- Use ONLY tool names/aliases from TOOL_REGISTRY_JSON.
- Arguments must satisfy the tool's JSON Schema.
- Never guess missing required arguments. Ask the user if needed.
```

### Custom Contract Example

```
You are an assistant with tools. Always use tools when they can help.

AVAILABLE TOOLS:
{registry_json}

TO CALL A TOOL:
{"type": "tool_call", "name": "tool_name", "arguments": {"arg": "value"}}

TO RESPOND:
{"type": "final", "content": "your response"}

IMPORTANT:
- Only output JSON, nothing else
- Ask for missing information, don't guess
```

---

## Directory Structure

```
~/.l3m/
├── config.json           # Main configuration
├── mcp-servers.json      # MCP server definitions
├── tool-contract.txt     # Custom tool contract (optional)
├── prompt_history        # Command history (auto-generated)
├── models/               # GGUF model files
│   └── *.gguf
├── sessions/             # Saved sessions
│   ├── *.json
│   └── anonymous/        # Sessions without titles
├── tools/                # User-defined tools
│   └── <tool_name>/
│       └── __init__.py
└── logs/                 # Log files (if enabled)
```

---

## Configuration Precedence

Settings are applied in this order (later overrides earlier):

1. **Built-in defaults** (`DEFAULTS` in config.py)
2. **Config file** (`~/.l3m/config.json`)
3. **Environment variables**
4. **Command-line arguments**

---

## Troubleshooting

### Config Not Loading

```bash
# Verify config file exists and is valid JSON
cat ~/.l3m/config.json | python -m json.tool

# Reinitialize (adds missing keys, keeps existing)
l3m-init

# Force reset to defaults
l3m-init --force
```

### View Current Config

```bash
# In REPL
/config

# From CLI
cat ~/.l3m/config.json
```

### Invalid Config Value

If a config value causes errors:

```bash
# Remove the problematic key
l3m-chat --set-config problematic_key=null

# Or edit directly
nano ~/.l3m/config.json
```
