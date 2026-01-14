# CLI Command Reference

Complete reference for all l3m-backend command-line tools and REPL commands.

---

## Table of Contents

1. [CLI Commands](#cli-commands)
   - [l3m-init](#l3m-init)
   - [l3m-chat](#l3m-chat)
   - [l3m-tools](#l3m-tools)
   - [l3m-download](#l3m-download)
   - [l3m-completion](#l3m-completion)
   - [l3m-mcp-server](#l3m-mcp-server)
2. [REPL Commands](#repl-commands)
   - [Slash Commands](#slash-commands)
   - [Shell Commands](#shell-commands)
   - [Magic Commands](#magic-commands)
   - [Session Commands](#session-commands)
   - [MCP Commands](#mcp-commands)
   - [Advanced Commands](#advanced-commands)
3. [Keyboard Shortcuts](#keyboard-shortcuts)

---

## CLI Commands

### l3m-init

Initialize l3m-backend configuration and directories.

```bash
l3m-init [options]
```

#### Options

| Option | Description |
|--------|-------------|
| `--force` | Overwrite existing config completely |
| `--quiet` | Suppress output |

#### Behavior

- Creates `~/.l3m/` directory structure
- Creates `config.json` with defaults
- Creates `models/`, `sessions/`, `tools/` directories
- **Non-destructive by default**: Only adds missing config keys

#### Examples

```bash
# Initialize (adds missing keys, preserves existing)
l3m-init

# Force reset to defaults
l3m-init --force
```

---

### l3m-chat

Interactive chat REPL with tool calling.

```bash
l3m-chat [model] [options]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `model` | Model filename or path (optional) |

#### Options

| Option | Description |
|--------|-------------|
| `--list`, `-l` | List available models |
| `--ctx N` | Context size in tokens |
| `--gpu N` | GPU layers (-1=all, 0=CPU) |
| `-v`, `--verbose` | Verbose llama.cpp output |
| `--show-warnings` | Show GPU initialization warnings |
| `--simple` | Use simple REPL (no prompt_toolkit) |
| `--no-native-tools` | Use contract-only tool calling |
| `--no-flash-attn` | Disable flash attention |
| `--minimal-contract` | Use minimal tool contract |
| `--incognito` | Don't persist session |
| `--temperature T` | Generation temperature |
| `--system PROMPT` | Custom system prompt |
| `--chat-format FMT` | Override chat format |
| `--max-tokens N` | Max tokens per response |
| `--debug` | Enable debug output |
| `--set-config K=V` | Set config value |
| `--mcp SERVER` | Auto-connect to MCP server |

#### Examples

```bash
# Auto-detect model
l3m-chat

# Specific model
l3m-chat Llama-3.2-3B-Instruct-Q4_K_M.gguf

# List available models
l3m-chat --list

# Custom settings
l3m-chat --ctx 16384 --gpu -1 --temperature 0.7

# Incognito mode
l3m-chat --incognito

# With MCP server
l3m-chat --mcp filesystem

# Set config
l3m-chat --set-config ctx=16384
```

---

### l3m-tools

Manage and inspect tools.

```bash
l3m-tools <command> [options]
```

#### Commands

##### list

List all registered tools.

```bash
l3m-tools list [-v]
```

| Option | Description |
|--------|-------------|
| `-v`, `--verbose` | Show descriptions |

##### info

Show detailed information about a tool.

```bash
l3m-tools info <tool_name>
```

##### schema

Output OpenAI-style tool schemas.

```bash
l3m-tools schema [tool_name]
```

Without tool name, outputs all schemas.

##### create

Create a new tool scaffold.

```bash
l3m-tools create <name> [options]
```

| Option | Description |
|--------|-------------|
| `--wrap CMD` | Create wrapper for shell command |
| `--description DESC` | Tool description |

#### Examples

```bash
# List tools
l3m-tools list
l3m-tools list -v

# Tool info
l3m-tools info get_weather

# Show schema
l3m-tools schema
l3m-tools schema calculate

# Create new tool
l3m-tools create my_tool
l3m-tools create docker-ps --wrap "docker ps"
```

---

### l3m-download

Download GGUF models from Hugging Face.

```bash
l3m-download [options]
```

#### Options

| Option | Description |
|--------|-------------|
| `--preset NAME` | Download using preset |
| `--list-presets` | Show available presets |
| `--repo REPO` | Hugging Face repo (org/model) |
| `--file FILE` | Specific file to download |
| `--output DIR` | Output directory |

#### Presets

| Preset | Model | Size |
|--------|-------|------|
| `llama3.2-1b` | Llama 3.2 1B Instruct | ~0.8GB |
| `llama3.2-3b` | Llama 3.2 3B Instruct | ~2GB |
| `mistral-7b` | Mistral 7B Instruct v0.3 | ~4.4GB |
| `granite-8b` | IBM Granite 3.1 8B Instruct | ~5GB |
| `qwen2.5-3b` | Qwen 2.5 3B Instruct | ~2GB |
| `phi3-mini` | Microsoft Phi 3.5 Mini | ~2.4GB |

#### Examples

```bash
# List presets
l3m-download --list-presets

# Download using preset
l3m-download --preset llama3.2-3b

# Custom download
l3m-download --repo TheBloke/Mistral-7B-Instruct-v0.2-GGUF --file mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Custom output directory
l3m-download --preset llama3.2-1b --output /custom/path
```

---

### l3m-completion

Generate shell completion scripts.

```bash
l3m-completion <shell>
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `shell` | Shell type: `bash`, `zsh`, `fish` |

#### Examples

```bash
# Generate and source bash completion
source <(l3m-completion bash)

# Add to shell config
echo 'source <(l3m-completion bash)' >> ~/.bashrc
echo 'source <(l3m-completion zsh)' >> ~/.zshrc

# Fish shell
l3m-completion fish > ~/.config/fish/completions/l3m.fish
```

---

### l3m-mcp-server

Run l3m tools as an MCP server.

```bash
l3m-mcp-server [options]
```

#### Options

| Option | Description |
|--------|-------------|
| `--port PORT` | HTTP port (default: 8080) |
| `--host HOST` | Host to bind (default: localhost) |
| `--stdio` | Use stdio transport |

#### Examples

```bash
# Run HTTP server
l3m-mcp-server --port 8080

# Run stdio server (for local integration)
l3m-mcp-server --stdio
```

---

## REPL Commands

### Slash Commands

Pure commands that don't modify conversation history.

#### General

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/quit`, `/exit`, `/q` | Exit the REPL |
| `/clear` | Clear conversation history |

#### Information

| Command | Description |
|---------|-------------|
| `/tools` | List available tools |
| `/schema [name]` | Show tool schema(s) |
| `/system` | Show system prompt |
| `/contract` | Show full system message with tool contract |
| `/model` | Show model info and GPU status |
| `/context` | Show context usage estimate |
| `/config` | Show current configuration |

#### History Management

| Command | Description |
|---------|-------------|
| `/history` | Show conversation history |
| `/undo` | Remove last user+assistant exchange |
| `/pop [n]` | Remove last n messages (default: 1) |

---

### Shell Commands

Run shell commands directly. Output is displayed but NOT added to conversation.

```
!<command>
```

#### Examples

```
!ls -la
!git status
!python --version
!pwd
!cat /etc/hosts
```

---

### Magic Commands

Magic commands add their input and output to conversation history, making them visible to the LLM.

| Command | Description |
|---------|-------------|
| `%!<cmd>` | Run shell command (adds to history) |
| `%tool <name> [json]` | Invoke tool directly |
| `%load <file>` | Load file content into conversation |
| `%time` | Report current time |
| `%save <file>` | Save conversation to file |
| `%edit-response` | Edit last assistant response in $VISUAL |

#### Examples

```
# Run shell and share output with LLM
%!ls -la src/

# Invoke tool directly
%tool get_weather {"location": "Paris"}
%tool calculate {"expression": "2 + 2"}

# Load file for LLM to see
%load README.md
%load src/main.py

# Report time
%time

# Save conversation
%save conversation.md
%save ~/Documents/chat.txt

# Edit last response
%edit-response
```

---

### Session Commands

Commands for managing persistent sessions.

| Command | Description |
|---------|-------------|
| `/session` | Show current session info |
| `/sessions [query]` | List or search sessions |
| `/session-save [title]` | Save session (auto-generates title) |
| `/session-load <id>` | Load session as context |
| `/session-title [title]` | Set or generate session title |
| `/session-tag <tag>` | Add tag to session |
| `/session-delete <id>` | Delete a session |

#### Examples

```
# View current session
/session

# List all sessions
/sessions

# Search sessions
/sessions python
/sessions debugging api

# Save with auto-generated title
/session-save

# Save with custom title
/session-save "Debugging the login flow"

# Load session by ID (first 8 chars)
/session-load a1b2c3d4

# Set title
/session-title "My chat session"

# Add tag
/session-tag coding
/session-tag important

# Delete session
/session-delete a1b2c3d4
```

---

### MCP Commands

Commands for Model Context Protocol integration.

| Command | Description |
|---------|-------------|
| `/mcp` | Show MCP status and connected servers |
| `/mcp connect <name>` | Connect to MCP server |
| `/mcp disconnect <name>` | Disconnect from server |
| `/mcp tools [name]` | List tools from server(s) |
| `/mcp reconnect <name>` | Reconnect to server |

#### Examples

```
# Show status
/mcp

# Connect to configured server
/mcp connect filesystem
/mcp connect brave-search

# List tools
/mcp tools              # All connected servers
/mcp tools filesystem   # Specific server

# Disconnect
/mcp disconnect filesystem

# Reconnect after server restart
/mcp reconnect filesystem
```

---

### Advanced Commands

#### Knowledge Graph

| Command | Description |
|---------|-------------|
| `/kgraph` | Show knowledge graph status |
| `/entities [type]` | List extracted entities |
| `/related <entity>` | Find related entities |

#### Similarity Graph

| Command | Description |
|---------|-------------|
| `/sgraph` | Show similarity graph status |
| `/cluster` | Show message clusters |
| `/similar <text>` | Find similar messages |

#### Model Management

| Command | Description |
|---------|-------------|
| `/switch <model>` | Switch to different model |
| `/reload` | Reload current model |

#### Debugging

| Command | Description |
|---------|-------------|
| `/debug [on\|off]` | Toggle debug mode |
| `/tokens <text>` | Count tokens in text |
| `/raw` | Show raw last response |

---

## Keyboard Shortcuts

### prompt_toolkit REPL

| Shortcut | Action |
|----------|--------|
| `Tab` | Auto-complete commands |
| `Ctrl+R` | Search command history |
| `Ctrl+C` | Cancel current input |
| `Ctrl+D` | Exit (press twice) |
| `Ctrl+L` | Clear screen |
| `Up/Down` | Navigate history |
| `Ctrl+A` | Move to line start |
| `Ctrl+E` | Move to line end |
| `Ctrl+K` | Delete to end of line |
| `Ctrl+U` | Delete to start of line |
| `Ctrl+W` | Delete previous word |
| `Alt+B` | Move back one word |
| `Alt+F` | Move forward one word |

### Multi-line Input

Press `Enter` to submit. For multi-line input:

| Method | Description |
|--------|-------------|
| `\\` at line end | Continue on next line |
| Triple backticks | Start/end code block |

```
You: ```python
def hello():
    print("Hello!")
```
```

---

## Command Completion

Tab completion is available for:

- Slash commands (`/he<tab>` → `/help`)
- Tool names (`%tool get_<tab>` → `%tool get_weather`)
- Session IDs (`/session-load a1<tab>`)
- File paths (`%load ~/Doc<tab>`)
- MCP server names (`/mcp connect file<tab>`)

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Normal exit |
| `1` | Error (config, model loading, etc.) |
| `130` | Interrupted (Ctrl+C) |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VISUAL` | Editor for `%edit-response` |
| `EDITOR` | Fallback editor |
| `L3M_MODELS_DIR` | Override models directory |
| `L3M_CONFIG_DIR` | Override config directory |
