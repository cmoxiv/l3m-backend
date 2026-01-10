"""
l3m_backend - LLM Tool-Calling System

A lightweight tool-calling system for local LLMs using llama-cpp-python.
Provides a contract-based approach that works with any model capable of
generating JSON, without requiring native function-calling support.

Example usage:
    from l3m_backend import ToolRegistry, tool_output

    registry = ToolRegistry()

    @registry.register(aliases=["weather", "w"])
    @tool_output(llm_format="{location}: {temperature}°C")
    def get_weather(location: str) -> dict:
        return {"location": location, "temperature": 22}

    # Use with ChatEngine (requires llama-cpp-python)
    from l3m_backend import ChatEngine
    from l3m_backend.tools import registry  # Pre-loaded with built-in tools

    engine = ChatEngine("./model.gguf", registry)
    response = engine.chat("What's the weather in Paris?")
"""

__version__ = "0.1.0"

# Core exports
from l3m_backend.core import (
    ToolEntry,
    ToolError,
    ToolNotFoundError,
    ToolOutput,
    ToolParseError,
    ToolRegistry,
    ToolResult,
    ToolValidationError,
    parse_model_response,
    tool_output,
)

# Lazy import for ChatEngine (requires llama-cpp-python)
def __getattr__(name):
    if name == "ChatEngine":
        from l3m_backend.engine import ChatEngine
        return ChatEngine
    if name == "TOOL_CONTRACT_TEMPLATE":
        from l3m_backend.engine import TOOL_CONTRACT_TEMPLATE
        return TOOL_CONTRACT_TEMPLATE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Tools registry (lazy import to avoid circular imports)
def __getattr__(name):
    if name == "registry":
        from l3m_backend.tools import registry
        return registry
    if name == "ChatEngine":
        from l3m_backend.engine import ChatEngine
        return ChatEngine
    if name == "TOOL_CONTRACT_TEMPLATE":
        from l3m_backend.engine import TOOL_CONTRACT_TEMPLATE
        return TOOL_CONTRACT_TEMPLATE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Version
    "__version__",
    # Core
    "ToolRegistry",
    "ToolEntry",
    "ToolOutput",
    "ToolResult",
    "ToolError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolParseError",
    "tool_output",
    "parse_model_response",
    # Engine (lazy loaded)
    "ChatEngine",
    "TOOL_CONTRACT_TEMPLATE",
    # Tools (lazy loaded)
    "registry",
]
