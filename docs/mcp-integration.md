# MCP Integration Guide

Guide to using Model Context Protocol (MCP) with l3m-backend.

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Using MCP Servers](#using-mcp-servers)
4. [Running l3m as MCP Server](#running-l3m-as-mcp-server)
5. [Python API](#python-api)
6. [Common MCP Servers](#common-mcp-servers)
7. [Troubleshooting](#troubleshooting)

---

## Overview

MCP (Model Context Protocol) is a standard for connecting AI assistants to external tools and data sources. l3m-backend supports MCP in two ways:

1. **As MCP Client**: Connect to external MCP servers to access their tools
2. **As MCP Server**: Expose l3m tools to other MCP clients

### Benefits

- Access file systems, databases, APIs through standardized tools
- Use community-built MCP servers
- Share l3m tools with other applications

---

## Configuration

### MCP Server Configuration File

Create `~/.l3m/mcp-servers.json`:

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-filesystem", "/home/user/documents"],
      "auto_connect": true
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-api-key-here"
      },
      "auto_connect": false
    }
  }
}
```

### Server Types

#### Local Server (stdio)

Runs as a subprocess, communicates via stdin/stdout.

```json
{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@anthropic/mcp-server-filesystem", "/allowed/path"],
    "env": {
      "OPTIONAL_VAR": "value"
    },
    "auto_connect": true
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `command` | Yes | Command to run |
| `args` | Yes | Command arguments |
| `env` | No | Environment variables |
| `auto_connect` | No | Connect on startup (default: false) |

#### Remote Server (HTTP/SSE)

Connects to a remote MCP server over HTTP.

```json
{
  "remote-server": {
    "url": "http://localhost:8080/mcp",
    "auto_connect": false
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `url` | Yes | Server URL |
| `auto_connect` | No | Connect on startup (default: false) |

---

## Using MCP Servers

### Auto-Connect on Startup

Servers with `"auto_connect": true` connect automatically when l3m-chat starts.

Or specify on command line:

```bash
l3m-chat --mcp filesystem --mcp brave-search
```

### REPL Commands

```
# Show MCP status
/mcp

# Connect to a configured server
/mcp connect filesystem

# List tools from connected servers
/mcp tools

# List tools from specific server
/mcp tools filesystem

# Disconnect
/mcp disconnect filesystem

# Reconnect after server restart
/mcp reconnect filesystem
```

### Using MCP Tools

Once connected, MCP tools are available to the LLM just like built-in tools.

```
You: Read the file /etc/hosts
Assistant: [Uses MCP filesystem tool to read file]
The hosts file contains:
127.0.0.1 localhost
...

You: Search for Python tutorials
Assistant: [Uses MCP brave-search tool]
Here are some Python tutorials:
...
```

### Tool Naming

MCP tools are prefixed with server name to avoid conflicts:

- `filesystem.read_file`
- `filesystem.write_file`
- `brave-search.search`

The LLM can use either the full name or try the base name if unique.

---

## Running l3m as MCP Server

Expose l3m's built-in tools as an MCP server for other applications.

### Stdio Mode

For local integration (e.g., Claude Desktop):

```bash
l3m-mcp-server --stdio
```

Add to Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "l3m": {
      "command": "l3m-mcp-server",
      "args": ["--stdio"]
    }
  }
}
```

### HTTP Mode

For network access:

```bash
l3m-mcp-server --port 8080 --host 0.0.0.0
```

Connect from other l3m instances:

```json
{
  "servers": {
    "remote-l3m": {
      "url": "http://192.168.1.100:8080/mcp",
      "auto_connect": false
    }
  }
}
```

### Available Tools

All registered l3m tools are exposed:

- `calculate` - Math calculations
- `get_weather` - Weather lookup
- `get_time` - Current time
- `wikipedia` - Wikipedia search
- User-defined tools from `~/.l3m/tools/`

---

## Python API

### MCPClient

```python
from l3m_backend.mcp import MCPClient
from l3m_backend.mcp.client import run_async

# Create client
client = MCPClient()

# Connect to server
connection = run_async(client.connect("filesystem"))

# List tools
tools = run_async(client.list_tools("filesystem"))
for tool in tools:
    print(f"{tool.name}: {tool.description}")

# Call a tool
result = run_async(client.call_tool(
    "filesystem",
    "read_file",
    {"path": "/etc/hosts"}
))
print(result)

# Disconnect
run_async(client.disconnect("filesystem"))
```

### Async Usage

```python
import asyncio
from l3m_backend.mcp import MCPClient

async def main():
    async with MCPClient() as client:
        # Connect
        await client.connect("filesystem")

        # List all tools from all servers
        all_tools = await client.list_all_tools()
        for server, tools in all_tools.items():
            print(f"{server}: {len(tools)} tools")

        # Call tool
        result = await client.call_tool(
            "filesystem",
            "read_file",
            {"path": "/etc/hosts"}
        )
        print(result)

asyncio.run(main())
```

### With ChatEngine

```python
from l3m_backend import ChatEngine
from l3m_backend.tools import registry
from l3m_backend.mcp import MCPClient
from l3m_backend.mcp.client import run_async
from l3m_backend.mcp.client.registry_adapter import MCPRegistryAdapter

# Create MCP client and connect
client = MCPClient()
run_async(client.connect("filesystem"))

# Create adapter to bridge MCP tools to registry
adapter = MCPRegistryAdapter(client, registry)
run_async(adapter.register_all_tools())

# Now ChatEngine has access to MCP tools
engine = ChatEngine("./model.gguf", registry)
response = engine.chat("Read the file /etc/hosts")
print(response)
```

### MCPConnection Properties

```python
connection = run_async(client.connect("filesystem"))

# Available tools
print(connection.tools)

# Available resources
print(connection.resources)

# Available prompts
print(connection.prompts)

# Refresh capabilities
run_async(connection.refresh_capabilities())
```

### Error Handling

```python
from l3m_backend.mcp.exceptions import (
    MCPError,
    MCPConnectionError,
    MCPServerNotFoundError,
    MCPToolError,
)

try:
    connection = run_async(client.connect("unknown-server"))
except MCPServerNotFoundError:
    print("Server not configured")
except MCPConnectionError as e:
    print(f"Connection failed: {e}")

try:
    result = run_async(client.call_tool("filesystem", "read_file", {"path": "/root"}))
except MCPToolError as e:
    print(f"Tool error: {e}")
```

---

## Common MCP Servers

### Anthropic Official Servers

#### Filesystem

Access local files within allowed directories.

```json
{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@anthropic/mcp-server-filesystem", "/path/to/allowed/dir"],
    "auto_connect": true
  }
}
```

Tools: `read_file`, `write_file`, `list_directory`, `create_directory`

#### Brave Search

Web search using Brave Search API.

```json
{
  "brave-search": {
    "command": "npx",
    "args": ["-y", "@anthropic/mcp-server-brave-search"],
    "env": {
      "BRAVE_API_KEY": "your-api-key"
    }
  }
}
```

Tools: `search`

Get API key: https://brave.com/search/api/

#### GitHub

Access GitHub repositories.

```json
{
  "github": {
    "command": "npx",
    "args": ["-y", "@anthropic/mcp-server-github"],
    "env": {
      "GITHUB_TOKEN": "your-token"
    }
  }
}
```

Tools: `search_repos`, `get_file_contents`, `create_issue`, etc.

### Community Servers

Many community MCP servers are available. Check:
- https://github.com/topics/mcp-server
- https://mcp.so/

---

## Troubleshooting

### Server Not Connecting

1. Check server is configured in `~/.l3m/mcp-servers.json`
2. Verify command and args are correct
3. Check if npx/node is installed: `which npx`
4. Run command manually to see errors:
   ```bash
   npx -y @anthropic/mcp-server-filesystem /tmp
   ```

### "Server not found" Error

```
MCPServerNotFoundError: Server 'xyz' not configured
```

Add the server to `~/.l3m/mcp-servers.json`.

### Connection Timeout

```
MCPConnectionError: Connection to xyz timed out
```

- Server may be slow to start
- Check network connectivity for remote servers
- Increase timeout in code if using Python API

### Tool Call Failed

```
MCPToolError: Error calling read_file on filesystem: Permission denied
```

- Check permissions for the requested resource
- Verify allowed paths in server configuration
- Check server logs for details

### Server Crashed

```
/mcp
filesystem: disconnected (server died)
```

Reconnect:
```
/mcp reconnect filesystem
```

If persistent, check server logs and restart:
```
/mcp disconnect filesystem
/mcp connect filesystem
```

### Debug Mode

Enable debug output to see MCP communication:

```bash
l3m-chat --debug
```

Or in Python:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Checking Server Health

```python
from l3m_backend.mcp import MCPClient
from l3m_backend.mcp.client import run_async

client = MCPClient()
run_async(client.connect("filesystem"))

# Check if healthy
if client.is_healthy("filesystem"):
    print("Server is healthy")
else:
    print("Server may be down, attempting reconnect...")
    run_async(client.reconnect("filesystem"))
```

---

## Best Practices

### Security

1. **Limit allowed paths** in filesystem server
2. **Use read-only** configurations when possible
3. **Don't expose** sensitive environment variables
4. **Audit** MCP server code before using

### Performance

1. **Auto-connect** only frequently used servers
2. **Disconnect** servers when not needed
3. **Use local** servers when possible (lower latency)

### Reliability

1. **Handle reconnection** gracefully
2. **Check health** before critical operations
3. **Log errors** for debugging

```python
# Example: Robust tool call with retry
async def call_tool_robust(client, server, tool, args, max_retries=2):
    for attempt in range(max_retries + 1):
        if not client.is_healthy(server):
            await client.reconnect(server)

        try:
            return await client.call_tool(server, tool, args)
        except MCPToolError:
            if attempt < max_retries:
                await asyncio.sleep(1)
            else:
                raise
```
