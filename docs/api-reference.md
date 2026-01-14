# API Reference

Complete API documentation for l3m-backend.

---

## Table of Contents

1. [Core Module](#core-module)
   - [ToolRegistry](#toolregistry)
   - [ToolEntry](#toolentry)
   - [ToolOutput](#tooloutput)
   - [ToolResult](#toolresult)
   - [Decorators](#decorators)
   - [Exceptions](#exceptions)
2. [Engine Module](#engine-module)
   - [ChatEngine](#chatengine)
   - [ContextPartition](#contextpartition)
3. [Config Module](#config-module)
   - [Config](#config)
   - [ConfigManager](#configmanager)
4. [Session Module](#session-module)
   - [Session](#session)
   - [SessionManager](#sessionmanager)
   - [SessionMessage](#sessionmessage)
   - [SessionMetadata](#sessionmetadata)
5. [MCP Module](#mcp-module)
   - [MCPClient](#mcpclient)
   - [MCPConnection](#mcpconnection)
6. [Tools Module](#tools-module)

---

## Core Module

The core module provides the foundation for tool registration, validation, and execution.

```python
from l3m_backend import ToolRegistry, ToolEntry, ToolOutput, ToolResult, tool_output
from l3m_backend.core import ToolError, ToolNotFoundError, ToolValidationError, ToolParseError
```

### ToolRegistry

Registry for managing and executing tools.

```python
class ToolRegistry:
    """Registry for managing tools."""
```

#### Constructor

```python
ToolRegistry()
```

Creates an empty registry.

#### Methods

##### register()

```python
def register(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    aliases: list[str] | None = None,
) -> Callable
```

Register a tool. Can be used as a decorator with or without arguments.

**Parameters:**
- `fn`: Function to register (when used without parentheses)
- `name`: Override function name (optional)
- `description`: Override docstring description (optional)
- `aliases`: Alternative names for the tool (optional)

**Usage:**

```python
# Without arguments
@registry.register
def my_tool(x: int) -> dict:
    """My tool description."""
    return {"result": x}

# With arguments
@registry.register(name="custom_name", aliases=["mt", "mytool"])
def my_tool(x: int) -> dict:
    """My tool description."""
    return {"result": x}
```

##### get()

```python
def get(name_or_alias: str) -> ToolEntry
```

Resolve a tool by name or alias.

**Parameters:**
- `name_or_alias`: Tool name or any of its aliases

**Returns:** `ToolEntry` object

**Raises:** `ToolNotFoundError` if tool doesn't exist

##### execute()

```python
def execute(name: str, arguments: dict[str, Any]) -> ToolResult
```

Execute a tool with the given arguments.

**Parameters:**
- `name`: Tool name or alias
- `arguments`: Dictionary of arguments

**Returns:** `ToolResult` with name, arguments, and output

**Raises:**
- `ToolNotFoundError`: If tool doesn't exist
- `ToolValidationError`: If arguments fail validation

##### execute_call()

```python
def execute_call(call: dict[str, Any]) -> ToolResult
```

Execute a tool call from a dictionary.

**Parameters:**
- `call`: Dict with format `{"type": "tool_call", "name": "...", "arguments": {...}}`

**Returns:** `ToolResult`

**Raises:** `ToolParseError` if format is invalid

##### to_openai_tools()

```python
def to_openai_tools() -> list[dict[str, Any]]
```

Get all tools as OpenAI-style tool specifications.

**Returns:** List of OpenAI function tool specs

**Example output:**

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get weather for a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
      }
    }
  }
]
```

##### to_registry_json()

```python
def to_registry_json() -> str
```

Get registry as JSON string for system prompts.

**Returns:** JSON string with tool definitions

##### to_minimal_list()

```python
def to_minimal_list() -> str
```

Generate compact tool list for smaller context windows.

**Returns:** String with format `- tool_name(param1, param2): Brief description`

##### build_system_prompt()

```python
def build_system_prompt() -> str
```

Build a complete system prompt with tool contract.

**Returns:** System prompt string with embedded tool definitions

#### Magic Methods

```python
def __contains__(name: str) -> bool  # Check if tool exists
def __iter__() -> Iterator[ToolEntry]  # Iterate over tools
def __len__() -> int  # Number of registered tools
```

---

### ToolEntry

Registry entry for a single tool.

```python
class ToolEntry(BaseModel):
    """Registry entry for a single tool."""

    name: str
    callable_fn: Callable
    aliases: list[str]
    description: str | None
```

#### Methods

##### get_params_model()

```python
def get_params_model() -> type[BaseModel]
```

Get or create the Pydantic model for this tool's parameters.

**Returns:** Pydantic model class generated from function signature

##### get_description()

```python
def get_description() -> str
```

Get description from override or docstring.

**Returns:** Description string

##### to_openai_spec()

```python
def to_openai_spec() -> dict[str, Any]
```

Convert to OpenAI-style tool specification.

**Returns:** OpenAI function tool spec dictionary

---

### ToolOutput

Structured tool output with formatting options.

```python
class ToolOutput(BaseModel):
    """Structured tool output."""

    type_name: str       # Type of the output data
    id: str              # Unique ID for this output
    data: Any            # The actual output data
    llm_format: str | None  # LLM-friendly string representation
    gui_format: str | None  # GUI-friendly string representation
```

#### Class Methods

##### create()

```python
@classmethod
def create(
    cls,
    output: Any,
    llm_format: str | None = None,
    gui_format: str | None = None,
) -> ToolOutput
```

Create a ToolOutput from raw output data.

**Parameters:**
- `output`: The raw output data
- `llm_format`: Pre-formatted string for LLM consumption
- `gui_format`: Pre-formatted string for GUI display

**Returns:** `ToolOutput` instance

---

### ToolResult

Result of a tool execution.

```python
class ToolResult(BaseModel):
    """Result of a tool execution."""

    name: str                  # Tool name that was executed
    arguments: dict[str, Any]  # Arguments passed to the tool
    output: Any                # Tool output (ToolOutput or raw)
```

---

### Decorators

#### @tool_output

```python
def tool_output(
    llm_format: str | Callable[[Any], str] | None = None,
    gui_format: str | Callable[[Any], str] | None = None,
) -> Callable
```

Decorator that wraps function return value in `ToolOutput`.

**Parameters:**
- `llm_format`: Format string template or callable for LLM output
- `gui_format`: Format string template or callable for GUI output

**Format options:**
- String template: `"{location}: {temperature}C"` - uses dict keys
- Callable: `lambda x: f"Temp: {x['temperature']}C"` - custom formatting

**Usage:**

```python
# String template
@tool_output(llm_format="{location}: {temperature}C, {condition}")
def get_weather(location: str) -> dict:
    return {"location": location, "temperature": 22, "condition": "sunny"}

# Callable
@tool_output(llm_format=lambda x: f"Result: {x['value']}")
def calculate(expression: str) -> dict:
    return {"value": eval(expression)}

# Both formats
@tool_output(
    llm_format="{result}",
    gui_format="<div class='result'>{result}</div>"
)
def my_tool(x: int) -> dict:
    return {"result": x * 2}
```

---

### Exceptions

```python
class ToolError(Exception):
    """Base exception for tool-related errors."""

class ToolNotFoundError(ToolError):
    """Tool not found in registry."""

class ToolValidationError(ToolError):
    """Tool argument validation failed."""

class ToolParseError(ToolError):
    """Failed to parse model output as tool call."""
```

---

## Engine Module

The engine module provides the chat engine for LLM interaction with tool calling.

```python
from l3m_backend import ChatEngine
from l3m_backend.engine import TOOL_CONTRACT_TEMPLATE
```

### ChatEngine

Chat engine that manages LLM interaction with contract-based tool calling.

```python
class ChatEngine:
    """Chat engine that manages LLM interaction with contract-based tool calling."""
```

#### Constructor

```python
def __init__(
    self,
    model_path: str,
    registry: ToolRegistry,
    system_prompt: str | None = None,
    n_ctx: int = 32768,
    n_gpu_layers: int = -1,
    verbose: bool = False,
    show_warnings: bool = False,
    use_native_tools: bool = True,
    chat_format: str | None = None,
    temperature: float = 0.0,
    flash_attn: bool = True,
    debug: bool = False,
    minimal_contract: bool = False,
    context_partition: ContextPartition | None = None,
    stop_tokens: list[str] | None = None,
)
```

**Parameters:**
- `model_path`: Path to the GGUF model file
- `registry`: ToolRegistry with registered tools
- `system_prompt`: Custom system prompt (default: helpful assistant)
- `n_ctx`: Context size in tokens (default: 32768)
- `n_gpu_layers`: GPU layers (-1=all, 0=CPU only, N=partial)
- `verbose`: Enable verbose llama.cpp output
- `show_warnings`: Show Metal/GPU initialization warnings
- `use_native_tools`: Use native function calling if available
- `chat_format`: Override chat format (e.g., "chatml")
- `temperature`: Generation temperature (default: 0.0)
- `flash_attn`: Enable flash attention (default: True)
- `debug`: Enable debug output (default: False)
- `minimal_contract`: Use minimal tool contract (default: False)
- `context_partition`: Pre-loaded context partition
- `stop_tokens`: Stop tokens for generation

#### Methods

##### chat()

```python
def chat(
    user_input: str,
    max_tool_rounds: int = 3,
    stream: bool = False,
    ignore_history: bool = False,
    temp_system: str | None = None,
) -> str | Iterator[str]
```

Send a message and get a response with automatic tool execution.

**Parameters:**
- `user_input`: The user's message
- `max_tool_rounds`: Max LLM generation rounds (prevents infinite loops)
- `stream`: If True, returns a generator yielding tokens
- `ignore_history`: Bypass history for one-shot completion
- `temp_system`: Temporary system message (with ignore_history)

**Returns:**
- `str`: Complete response (if stream=False)
- `Iterator[str]`: Token generator (if stream=True)

**Example:**

```python
# Non-streaming
response = engine.chat("What's the weather in Tokyo?")
print(response)

# Streaming
for token in engine.chat("Tell me a story", stream=True):
    print(token, end="", flush=True)
```

##### clear()

```python
def clear() -> None
```

Clear conversation history. System message is NOT affected.

##### switch_model()

```python
def switch_model(
    model_path: str,
    preserve_history: bool = True,
    n_ctx: int | None = None,
    n_gpu_layers: int | None = None,
) -> None
```

Switch to a different model, optionally preserving history.

**Parameters:**
- `model_path`: Path to the new model file
- `preserve_history`: Keep conversation history (default: True)
- `n_ctx`: New context size (default: keep current)
- `n_gpu_layers`: New GPU layers (default: keep current)

##### warmup()

```python
def warmup(
    transcript: list[dict[str, Any]] | None = None,
    summaries: list[str] | None = None,
    max_transcript_messages: int = 20,
    build_similarity_graph: bool = False,
    build_knowledge_graph: bool = False,
    kg_progress_callback: Callable[[int, int], None] | None = None,
    sg_progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]
```

Run warmup generation to prime KV cache. Useful when loading sessions.

**Parameters:**
- `transcript`: Historical messages to process
- `summaries`: Summary strings to include
- `max_transcript_messages`: Max transcript messages (default: 20)
- `build_similarity_graph`: Build similarity graph from content
- `build_knowledge_graph`: Build knowledge graph from content
- `kg_progress_callback`: Progress callback (current, total)
- `sg_progress_callback`: Progress callback (current, total)

**Returns:** Dict with `tokens`, `time_s`, `graph_built`, `kg_built`

##### check_and_trim_history()

```python
def check_and_trim_history(reserve_tokens: int = 2048) -> dict[str, Any]
```

Check if history exceeds context and trim if needed.

**Returns:** Dict with `trimmed`, `removed_count`, `remaining_count`, etc.

##### execute_suggested_tool()

```python
def execute_suggested_tool(tool_call: dict) -> str
```

Execute a tool that was suggested in a mixed text+JSON response.

**Parameters:**
- `tool_call`: Dict with "name" and "arguments" keys

**Returns:** Formatted tool result string

#### Properties

```python
@property
def messages(self) -> list[dict[str, Any]]
    """Return full message list for display."""

@property
def history(self) -> list[dict[str, Any]]
    """Conversation history (user/assistant/tool messages)."""

@property
def has_knowledge_graph(self) -> bool
    """Check if knowledge graph is available."""

@property
def has_similarity_graph(self) -> bool
    """Check if similarity graph is available."""
```

---

### ContextPartition

Manages loaded context from previous sessions.

```python
class ContextPartition:
    """Manages loaded summaries and transcript excerpts."""

    loaded_summaries: list[str]
    loaded_transcript: list[dict[str, Any]]
```

#### Methods

##### has_partitions()

```python
def has_partitions() -> bool
```

Check if any partitioned context is loaded.

---

## Config Module

Configuration management for l3m-backend.

```python
from l3m_backend.config import Config, ConfigManager, get_config_manager, get_config, DEFAULTS
```

### DEFAULTS

Default configuration values:

```python
DEFAULTS = {
    "ctx": 8192,
    "gpu": -1,
    "verbose": False,
    "show_warnings": False,
    "simple": False,
    "no_native_tools": False,
    "no_flash_attn": False,
    "minimal_contract": False,
    "incognito": False,
    "auto_resume": False,
    "temperature": 0.1,
    "system_prompt": "You are a helpful assistant...",
    "summary_ctx": 0,
    "transcript_ctx": 0,
    "stop_tokens": ["<|end|>", "<|eot_id|>", "<|im_end|>"],
}
```

### Config

Configuration settings model.

```python
class Config(BaseModel):
    """Configuration settings for l3m-backend."""

    default_model: str | None
    ctx: int | None
    gpu: int | None
    verbose: bool | None
    show_warnings: bool | None
    no_native_tools: bool | None
    no_flash_attn: bool | None
    minimal_contract: bool | None
    simple: bool | None
    incognito: bool | None
    auto_resume: bool | None
    summary_ctx: int | None
    transcript_ctx: int | None
    system_prompt: str | None
    temperature: float | None
    chat_format: str | None
    max_tokens: int | None
    stop_tokens: list[str] | None
```

#### Methods

##### get()

```python
def get(key: str, default: Any = None) -> Any
```

Get a config value with fallback to DEFAULTS, then to provided default.

### ConfigManager

Manages loading and saving configuration.

```python
class ConfigManager:
    CONFIG_DIR = Path.home() / ".l3m"
    CONFIG_FILE = CONFIG_DIR / "config.json"
```

#### Methods

```python
def load(create_if_missing: bool = True) -> Config
def save(config: Config | None = None) -> Path
def set(key: str, value: Any) -> None
def unset(key: str) -> None
def get(key: str, default: Any = None) -> Any
def list_settings() -> dict[str, Any]  # User-customized settings
def reset() -> None  # Reset to defaults
def ensure_config_complete() -> list[str]  # Add missing keys
```

### Helper Functions

```python
def get_config_manager() -> ConfigManager
    """Get the singleton ConfigManager instance."""

def get_config() -> Config
    """Get the current configuration."""
```

---

## Session Module

Session persistence and management.

```python
from l3m_backend.session import Session, SessionManager, SessionMessage, SessionMetadata
```

### Session

Full session with transcript and metadata.

```python
class Session(BaseModel):
    metadata: SessionMetadata
    transcript: list[SessionMessage]
    history: list[dict[str, Any]]  # Engine-compatible format
```

#### Class Methods

```python
@classmethod
def create(
    model_path: str,
    working_directory: str,
    n_ctx: int = 32768,
    is_incognito: bool = False,
    tag: str | None = None,
    loaded_model_path: str | None = None,
) -> Session
```

### SessionManager

Manages session persistence and operations.

```python
class SessionManager:
    SESSIONS_DIR = Path.home() / ".l3m" / "sessions"
    ANONYMOUS_DIR = SESSIONS_DIR / "anonymous"
    INCOGNITO_DIR = Path("/tmp/l3m-sessions")
    SYMLINK_NAME = ".l3m-session"
```

#### Constructor

```python
def __init__(session: Session | None = None)
```

#### Core Methods

```python
def create(
    model_path: str,
    working_directory: str,
    n_ctx: int = 32768,
    incognito: bool = False,
    tag: str | None = None,
    loaded_model_path: str | None = None,
) -> Session

def save(cwd: Path | None = None) -> Path
def load(session_id: str) -> Session
def delete(session_id: str) -> bool
```

#### History Management

```python
def add_message(
    role: str,
    content: str,
    to_history: bool = True,
    to_transcript: bool = True,
    working_directory: str | None = None,
    **kwargs,
) -> SessionMessage | None

def sync_from_engine(engine_history: list[dict[str, Any]]) -> None
def get_engine_history() -> list[dict[str, Any]]
```

#### Title/Tag Generation

```python
def generate_title(engine: ChatEngine, max_messages: int | None = None) -> str | None
def generate_tag(engine: ChatEngine) -> str | None
def generate_summary(engine: ChatEngine, start_idx: int, end_idx: int | None) -> SessionSummary | None
```

#### Symlink Management

```python
def create_symlink(cwd: Path) -> Path | None
def update_symlink(old_cwd: Path, new_cwd: Path) -> Path | None
def find_session_in_cwd(cwd: Path) -> str | None  # Returns session ID
```

#### Listing and Search

```python
def list_sessions(
    tag: str | None = None,
    cwd: str | None = None,
    include_incognito: bool = False,
    include_anonymous: bool = True,
) -> list[SessionMetadata]

def search(
    query: str,
    include_incognito: bool = False,
    include_anonymous: bool = True,
) -> list[SessionMetadata]
```

### SessionMessage

Single message in a session.

```python
class SessionMessage(BaseModel):
    role: str  # "user", "assistant", "tool", "system"
    content: str
    timestamp: datetime
    working_directory: str | None
    # Additional fields for tool messages
    tool_call_id: str | None
    tool_name: str | None
```

### SessionMetadata

Session metadata.

```python
class SessionMetadata(BaseModel):
    id: str  # UUID
    title: str | None
    tag: str | None
    model_path: str
    loaded_model_path: str | None
    working_directory: str
    n_ctx: int
    is_incognito: bool
    initial_datetime: str  # YYMMDD-HHMM
    last_save_datetime: str | None
    created_at: datetime
    updated_at: datetime
    summaries: list[SessionSummary]
```

---

## MCP Module

Model Context Protocol client for connecting to MCP servers.

```python
from l3m_backend.mcp import MCPClient, MCPConnection
from l3m_backend.mcp.exceptions import MCPError, MCPConnectionError, MCPToolError
```

### MCPClient

Client for connecting to and managing MCP servers.

```python
class MCPClient:
    """Client for connecting to and managing MCP servers."""
```

#### Constructor

```python
MCPClient()
```

#### Properties

```python
@property
def connected_servers(self) -> list[str]
    """List of currently connected server names."""
```

#### Methods

##### connect()

```python
async def connect(server_name: str) -> MCPConnection
```

Connect to an MCP server by name (from configuration).

**Parameters:**
- `server_name`: Name of the server from `~/.l3m/mcp-servers.json`

**Returns:** `MCPConnection` object

**Raises:**
- `MCPServerNotFoundError`: If server not in configuration
- `MCPConnectionError`: If connection fails

##### connect_with_config()

```python
async def connect_with_config(name: str, config: Any) -> MCPConnection
```

Connect to an MCP server with explicit configuration.

##### disconnect()

```python
async def disconnect(server_name: str) -> bool
```

Disconnect from an MCP server.

**Returns:** True if disconnected, False if wasn't connected

##### disconnect_all()

```python
async def disconnect_all() -> None
```

Disconnect from all connected servers.

##### list_tools()

```python
async def list_tools(server_name: str) -> list[Tool]
```

List tools from a specific server.

##### list_all_tools()

```python
async def list_all_tools() -> dict[str, list[Tool]]
```

List tools from all connected servers.

**Returns:** Dict mapping server name to list of tools

##### call_tool()

```python
async def call_tool(
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any
```

Call a tool on a specific server.

**Parameters:**
- `server_name`: Name of the server
- `tool_name`: Name of the tool
- `arguments`: Tool arguments

**Returns:** Tool result

**Raises:**
- `MCPConnectionError`: If not connected to server
- `MCPToolError`: If tool call fails

##### is_connected()

```python
def is_connected(server_name: str) -> bool
```

Check if a server is currently connected.

##### is_healthy()

```python
def is_healthy(server_name: str) -> bool
```

Check if a connection is still healthy.

##### reconnect()

```python
async def reconnect(server_name: str) -> MCPConnection | None
```

Attempt to reconnect to a disconnected server.

##### connect_auto_servers()

```python
async def connect_auto_servers() -> list[str]
```

Connect to all servers marked for auto-connect in configuration.

#### Async Context Manager

```python
async with MCPClient() as client:
    await client.connect("filesystem")
    # ... use client
# Automatically disconnects all servers on exit
```

### MCPConnection

Represents an active connection to an MCP server.

```python
@dataclass
class MCPConnection:
    name: str
    session: ClientSession
    tools: list[Tool]
    resources: list[Any]
    resource_templates: list[Any]
    prompts: list[Any]
```

#### Methods

```python
async def refresh_capabilities() -> None
    """Refresh the list of available tools, resources, and prompts."""
```

### Helper Functions

```python
def run_async(coro) -> Any
    """Run an async coroutine in a sync context.

    Schedules the coroutine on the dedicated MCP event loop.
    """
```

**Usage:**

```python
from l3m_backend.mcp.client import MCPClient, run_async

client = MCPClient()
connection = run_async(client.connect("filesystem"))
tools = run_async(client.list_tools("filesystem"))
result = run_async(client.call_tool("filesystem", "read_file", {"path": "/etc/hosts"}))
```

---

## Tools Module

Built-in tools and the shared registry.

```python
from l3m_backend.tools import registry  # Shared registry with all built-in tools
```

### Built-in Tool Categories

| Category | Tools |
|----------|-------|
| **Calculator** | `calculate` |
| **Time** | `get_time` |
| **Weather** | `get_weather` |
| **Information** | `wikipedia`, `define_word` |
| **Conversion** | `unit_convert`, `currency_convert` |
| **Web** | `web_search` |
| **Development** | `run_python`, `shell_cmd`, `read_file`, `write_file`, `http_request` |
| **Productivity** | `note`, `todo`, `reminder`, `timer` |
| **Utilities** | `random_number`, `uuid_generate`, `hash_text`, `base64_encode`, `json_format` |

### Using the Shared Registry

```python
from l3m_backend.tools import registry
from l3m_backend import ChatEngine

# Registry is pre-loaded with all built-in tools
engine = ChatEngine("./model.gguf", registry)

# List available tools
for tool in registry:
    print(f"{tool.name}: {tool.get_description()}")
```

### Adding Custom Tools

```python
from l3m_backend.tools import registry
from l3m_backend.core import tool_output

@registry.register(aliases=["mt"])
@tool_output(llm_format="{result}")
def my_tool(input: str) -> dict:
    """Process input and return result."""
    return {"result": input.upper()}
```

---

## Quick Reference

### Minimal Setup

```python
from l3m_backend import ToolRegistry, ChatEngine, tool_output

# Create registry
registry = ToolRegistry()

# Register a tool
@registry.register()
@tool_output(llm_format="{result}")
def greet(name: str) -> dict:
    """Greet someone by name."""
    return {"result": f"Hello, {name}!"}

# Create engine and chat
engine = ChatEngine("./model.gguf", registry)
response = engine.chat("Greet Alice")
print(response)
```

### Using Built-in Tools

```python
from l3m_backend import ChatEngine
from l3m_backend.tools import registry

engine = ChatEngine("./model.gguf", registry)
response = engine.chat("What's the weather in Paris?")
print(response)
```

### Streaming Responses

```python
for token in engine.chat("Tell me a story", stream=True):
    print(token, end="", flush=True)
print()
```

### Configuration

```python
from l3m_backend.config import get_config_manager

config = get_config_manager()
config.set("ctx", 16384)
config.set("temperature", 0.7)
```

### Session Management

```python
from l3m_backend.session import SessionManager

manager = SessionManager()
session = manager.create(
    model_path="model.gguf",
    working_directory="/path/to/project",
)
manager.add_message("user", "Hello!")
manager.save()
```
