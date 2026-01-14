# l3m-backend Documentation

Welcome to the l3m-backend documentation. This guide covers everything you need to use and extend the tool-calling system.

---

## Getting Started

| Document | Description |
|----------|-------------|
| [Installation Guide](installation.md) | How to install l3m-backend with GPU support |
| [User Manual](user-manual.md) | Complete guide for using l3m-chat |
| [Architecture](architecture.md) | How the system works internally |

---

## Reference Documentation

| Document | Description |
|----------|-------------|
| [API Reference](api-reference.md) | Complete Python API documentation |
| [CLI Reference](cli-reference.md) | All CLI commands and REPL commands |
| [Configuration Reference](configuration-reference.md) | All configuration options |

---

## Development Guides

| Document | Description |
|----------|-------------|
| [Tool Development Guide](tool-development.md) | Creating custom tools |
| [MCP Integration Guide](mcp-integration.md) | Model Context Protocol client and server |
| [Examples and Tutorials](examples.md) | Practical examples and code samples |

---

## Technical Documentation

| Document | Description |
|----------|-------------|
| [Hybrid Model Reset](hybrid-model-reset.md) | SSM/Mamba model state management |
| [Hermes Tool Calling](hermes-toolcalling.md) | Hermes model-specific details |

---

## Quick Reference (TLDR)

| Document | Description |
|----------|-------------|
| [l3m-chat](tldr/l3m-chat.md) | Quick reference for chat REPL |
| [l3m-tools](tldr/l3m-tools.md) | Quick reference for tool management |
| [l3m-download](tldr/l3m-download.md) | Quick reference for model download |
| [l3m-init](tldr/l3m-init.md) | Quick reference for initialization |
| [l3m-completion](tldr/l3m-completion.md) | Quick reference for shell completion |

---

## Reports

| Document | Description |
|----------|-------------|
| [Test Report](test-report.md) | Test coverage and results |
| [Code Review](code-review.md) | Security and code quality analysis |

---

## Quick Links

### Installation

```bash
pip install 'git+https://github.com/mo/l3m-backend.git[llm,repl]'
l3m-init
l3m-download --preset llama3.2-3b
l3m-chat
```

### Basic Usage

```python
from l3m_backend import ChatEngine
from l3m_backend.tools import registry

engine = ChatEngine("~/.l3m/models/model.gguf", registry)
response = engine.chat("What's the weather in Paris?")
```

### Creating a Tool

```python
from l3m_backend.tools import registry
from l3m_backend.core import tool_output

@registry.register(aliases=["mt"])
@tool_output(llm_format="{result}")
def my_tool(input: str) -> dict:
    """Process input."""
    return {"result": input.upper()}
```

---

## Documentation Version

This documentation is for **l3m-backend v0.1.0**.

---

## Contributing

Found an issue or want to improve the docs? Please open an issue or PR on GitHub.

---

## License

MIT License - See LICENSE file for details.
