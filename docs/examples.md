# Examples and Tutorials

Practical examples for using l3m-backend.

---

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Python API Examples](#python-api-examples)
3. [Tool Examples](#tool-examples)
4. [Advanced Examples](#advanced-examples)
5. [Integration Examples](#integration-examples)

---

## Quick Start Examples

### Basic Chat Session

```bash
# Initialize and download a model
l3m-init
l3m-download --preset llama3.2-3b

# Start chatting
l3m-chat
```

```
You: Hello!
Assistant: Hello! How can I help you today?

You: What's 15% of 85?
Assistant: 15% of 85 is 12.75.

You: What's the weather in Paris?
Assistant: The current weather in Paris is 18C with partly cloudy skies.

You: /quit
```

### Using Session Management

```
You: Help me debug a Python function
Assistant: I'd be happy to help! Please share the function code.

You: %load my_function.py
[File loaded: my_function.py (25 lines)]

You: Find the bug in this code
Assistant: I see the issue - on line 12, you're using = instead of ==...

You: /session-save "Debug factorial function"
Session saved: debug-factorial-function-a1b2c3d4

You: /quit
```

Resume later:
```bash
cd /path/to/project
l3m-chat
# Prompts: "Resume session? [y/N]"
```

### Direct Tool Invocation

```
You: %tool calculate {"expression": "sqrt(144) + 2^10"}
Result: 1036.0

You: %tool get_time {}
Current time: 2024-01-15 14:30:00 UTC

You: %tool wikipedia {"query": "Python programming language"}
Python is a high-level programming language...
```

### Shell Integration

```
You: !git status
On branch main
nothing to commit, working tree clean

You: %!cat pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
...
[The file content is now visible to the LLM]

You: Explain this pyproject.toml structure
Assistant: This is a Python project configuration file...
```

---

## Python API Examples

### Minimal Example

```python
from l3m_backend import ToolRegistry, ChatEngine, tool_output

# Create registry and register a tool
registry = ToolRegistry()

@registry.register()
@tool_output(llm_format="{greeting}")
def greet(name: str) -> dict:
    """Greet someone by name."""
    return {"greeting": f"Hello, {name}!"}

# Create engine and chat
engine = ChatEngine("~/.l3m/models/model.gguf", registry)
response = engine.chat("Greet Alice")
print(response)  # "Hello, Alice!"
```

### Using Built-in Tools

```python
from l3m_backend import ChatEngine
from l3m_backend.tools import registry

# Registry comes pre-loaded with built-in tools
engine = ChatEngine("~/.l3m/models/model.gguf", registry)

# The LLM can use any built-in tool
response = engine.chat("What's 15% tip on $47.50?")
print(response)  # "A 15% tip on $47.50 is $7.13"

response = engine.chat("What time is it?")
print(response)  # "The current time is..."
```

### Streaming Responses

```python
from l3m_backend import ChatEngine
from l3m_backend.tools import registry

engine = ChatEngine("~/.l3m/models/model.gguf", registry)

# Stream tokens as they're generated
print("Assistant: ", end="")
for token in engine.chat("Write a haiku about coding", stream=True):
    print(token, end="", flush=True)
print()
```

### Multi-Turn Conversation

```python
from l3m_backend import ChatEngine
from l3m_backend.tools import registry

engine = ChatEngine("~/.l3m/models/model.gguf", registry)

# Conversation maintains context
engine.chat("My name is Alice")
engine.chat("I'm working on a Python project")
response = engine.chat("What's my name and what am I working on?")
print(response)  # "Your name is Alice and you're working on a Python project"

# Clear history to start fresh
engine.clear()
```

### Custom System Prompt

```python
from l3m_backend import ChatEngine
from l3m_backend.tools import registry

engine = ChatEngine(
    "~/.l3m/models/model.gguf",
    registry,
    system_prompt="""You are a Python expert assistant.
    Always provide code examples when explaining concepts.
    Focus on best practices and clean code.""",
    temperature=0.7,
)

response = engine.chat("Explain decorators")
print(response)
```

### Error Handling

```python
from l3m_backend import ChatEngine, ToolRegistry
from l3m_backend.core import ToolError, ToolNotFoundError, ToolValidationError

registry = ToolRegistry()

@registry.register()
def divide(a: float, b: float) -> dict:
    """Divide two numbers."""
    if b == 0:
        raise ToolError("Cannot divide by zero")
    return {"result": a / b}

engine = ChatEngine("~/.l3m/models/model.gguf", registry)

try:
    response = engine.chat("Divide 10 by 0")
    print(response)  # LLM receives error and responds appropriately
except Exception as e:
    print(f"Error: {e}")
```

---

## Tool Examples

### Weather API Tool

```python
# ~/.l3m/tools/weather_api/__init__.py
import requests
from pydantic import Field
from l3m_backend.tools import registry
from l3m_backend.core import tool_output

API_KEY = "your-api-key"  # Get from openweathermap.org

@registry.register(aliases=["weather", "w"])
@tool_output(llm_format="{city}: {temp}C, {description}")
def get_real_weather(
    city: str = Field(description="City name"),
    units: str = Field(default="metric", description="Units: metric or imperial"),
) -> dict:
    """Get real weather data from OpenWeatherMap API."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": API_KEY, "units": units}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        return {
            "city": data["name"],
            "temp": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
        }
    except requests.RequestException as e:
        return {"error": f"API error: {e}"}
```

### Database Query Tool

```python
# ~/.l3m/tools/database/__init__.py
import sqlite3
from pydantic import Field
from l3m_backend.tools import registry
from l3m_backend.core import tool_output

DB_PATH = "~/.l3m/data/app.db"

@registry.register(aliases=["sql", "query"])
@tool_output(llm_format=lambda x: x.get("error") or f"Found {len(x['rows'])} rows")
def query_database(
    sql: str = Field(description="SQL SELECT query to execute"),
) -> dict:
    """Execute a read-only SQL query on the database."""
    # Safety check - only allow SELECT
    if not sql.strip().upper().startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed"}

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return {"rows": rows, "count": len(rows)}
    except sqlite3.Error as e:
        return {"error": f"Database error: {e}"}
```

### File Processor Tool

```python
# ~/.l3m/tools/file_processor/__init__.py
from pathlib import Path
from pydantic import Field
from typing import Literal
from l3m_backend.tools import registry
from l3m_backend.core import tool_output

@registry.register(aliases=["file"])
@tool_output(llm_format=lambda x: x.get("error") or x.get("content", x.get("message")))
def process_file(
    path: str = Field(description="File path"),
    action: Literal["read", "info", "count"] = Field(
        default="read",
        description="Action: read content, get info, or count lines"
    ),
) -> dict:
    """Process a file with various actions."""
    file_path = Path(path).expanduser()

    if not file_path.exists():
        return {"error": f"File not found: {path}"}

    if not file_path.is_file():
        return {"error": f"Not a file: {path}"}

    if action == "info":
        stat = file_path.stat()
        return {
            "name": file_path.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "extension": file_path.suffix,
        }

    try:
        content = file_path.read_text()
    except Exception as e:
        return {"error": f"Cannot read file: {e}"}

    if action == "count":
        lines = len(content.splitlines())
        words = len(content.split())
        chars = len(content)
        return {
            "message": f"{lines} lines, {words} words, {chars} characters",
            "lines": lines,
            "words": words,
            "chars": chars,
        }

    # Default: read
    if len(content) > 10000:
        content = content[:10000] + "\n... (truncated)"
    return {"content": content}
```

### Command Wrapper Tool

```bash
# Create using CLI
l3m-tools create docker-status --wrap "docker ps --format 'table {{.Names}}\t{{.Status}}'"
```

Generated code:
```python
# ~/.l3m/tools/docker_status/__init__.py
import subprocess
from l3m_backend.tools import registry
from l3m_backend.core import tool_output

@registry.register(aliases=["docker"])
@tool_output(llm_format=lambda x: x.get("error") or x["output"])
def docker_status() -> dict:
    """Show running Docker containers."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {"error": result.stderr}
        return {"output": result.stdout}
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out"}
    except Exception as e:
        return {"error": str(e)}
```

---

## Advanced Examples

### Custom Registry with Selected Tools

```python
from l3m_backend import ToolRegistry, ChatEngine, tool_output
from l3m_backend.tools.calculator import calculate
from l3m_backend.tools.time import get_time

# Create custom registry with only specific tools
registry = ToolRegistry()

# Re-register built-in tools
registry.register()(calculate)
registry.register()(get_time)

# Add custom tool
@registry.register()
@tool_output(llm_format="{result}")
def custom_tool(x: int) -> dict:
    return {"result": x * 2}

engine = ChatEngine("~/.l3m/models/model.gguf", registry)
# Only calculate, get_time, and custom_tool are available
```

### Session Management in Code

```python
from pathlib import Path
from l3m_backend import ChatEngine
from l3m_backend.tools import registry
from l3m_backend.session import SessionManager

# Create session manager
manager = SessionManager()
session = manager.create(
    model_path="model.gguf",
    working_directory=str(Path.cwd()),
    tag="coding",
)

# Create engine
engine = ChatEngine("~/.l3m/models/model.gguf", registry)

# Chat and track messages
user_input = "Help me with Python"
response = engine.chat(user_input)

# Sync to session
manager.sync_from_engine(engine.history)

# Save session
path = manager.save()
print(f"Session saved to: {path}")

# Generate title
title = manager.generate_title(engine)
print(f"Generated title: {title}")
```

### Loading Previous Session

```python
from l3m_backend import ChatEngine
from l3m_backend.tools import registry
from l3m_backend.session import SessionManager

manager = SessionManager()

# Load session by ID
session = manager.load("a1b2c3d4")
print(f"Loaded: {session.metadata.title}")

# Create engine and restore history
engine = ChatEngine("~/.l3m/models/model.gguf", registry)
engine.history = session.get_engine_history()

# Continue conversation
response = engine.chat("Where were we?")
```

### Batch Processing

```python
from l3m_backend import ChatEngine
from l3m_backend.tools import registry

engine = ChatEngine("~/.l3m/models/model.gguf", registry)

questions = [
    "What is Python?",
    "What is machine learning?",
    "What is an API?",
]

for question in questions:
    response = engine.chat(question, max_tool_rounds=1)
    print(f"Q: {question}")
    print(f"A: {response[:200]}...")
    print()
    engine.clear()  # Clear history between questions
```

### One-Shot Processing

```python
from l3m_backend import ChatEngine
from l3m_backend.tools import registry

engine = ChatEngine("~/.l3m/models/model.gguf", registry)

# Process without affecting conversation history
summary = engine.chat(
    "Summarize: Python is a programming language...",
    ignore_history=True,
    temp_system="You are a summarizer. Be concise.",
)

# Main conversation is unaffected
response = engine.chat("Hello!")  # Fresh conversation
```

---

## Integration Examples

### Flask Web API

```python
from flask import Flask, request, jsonify
from l3m_backend import ChatEngine
from l3m_backend.tools import registry

app = Flask(__name__)
engine = ChatEngine("~/.l3m/models/model.gguf", registry)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    response = engine.chat(message)
    return jsonify({"response": response})

@app.route("/clear", methods=["POST"])
def clear():
    engine.clear()
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(port=5000)
```

### Discord Bot

```python
import discord
from l3m_backend import ChatEngine
from l3m_backend.tools import registry

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# One engine per channel for conversation context
engines = {}

def get_engine(channel_id):
    if channel_id not in engines:
        engines[channel_id] = ChatEngine(
            "~/.l3m/models/model.gguf",
            registry
        )
    return engines[channel_id]

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!ai "):
        query = message.content[4:]
        engine = get_engine(message.channel.id)
        response = engine.chat(query)
        await message.channel.send(response[:2000])

    elif message.content == "!clear":
        engine = get_engine(message.channel.id)
        engine.clear()
        await message.channel.send("Conversation cleared!")

client.run("YOUR_BOT_TOKEN")
```

### Jupyter Notebook

```python
# Cell 1: Setup
from l3m_backend import ChatEngine
from l3m_backend.tools import registry

engine = ChatEngine("~/.l3m/models/model.gguf", registry)

def chat(message):
    """Helper function for cleaner notebook usage."""
    response = engine.chat(message)
    print(f"Assistant: {response}")
    return response

# Cell 2: Use
chat("What's 25 * 4?")

# Cell 3: Continue conversation
chat("Now add 50 to that")

# Cell 4: Stream output
from IPython.display import clear_output

for token in engine.chat("Write a poem", stream=True):
    print(token, end="", flush=True)
```

### CLI Application

```python
#!/usr/bin/env python3
"""Simple CLI chat application."""
import argparse
import sys
from l3m_backend import ChatEngine
from l3m_backend.tools import registry

def main():
    parser = argparse.ArgumentParser(description="Chat with LLM")
    parser.add_argument("--model", default="~/.l3m/models/model.gguf")
    parser.add_argument("--query", "-q", help="Single query (non-interactive)")
    args = parser.parse_args()

    engine = ChatEngine(args.model, registry)

    if args.query:
        # Single query mode
        print(engine.chat(args.query))
        return

    # Interactive mode
    print("Chat with LLM (type 'quit' to exit)")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            print("\nAssistant: ", end="")
            for token in engine.chat(user_input, stream=True):
                print(token, end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
```