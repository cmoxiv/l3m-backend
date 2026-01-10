"""
Built-in tools and shared registry for the l3m_backend package.

Importing this module registers all built-in tools with the shared registry.
"""

# Import registry first
from l3m_backend.tools._registry import registry

# Import all tool modules to register them with the registry
from l3m_backend.tools.calculator import calculate
from l3m_backend.tools.development import (
    http_request,
    read_file,
    run_python,
    shell_cmd,
    write_file,
)
from l3m_backend.tools.information import (
    currency_convert,
    define_word,
    unit_convert,
    web_search,
    wikipedia,
)
from l3m_backend.tools.productivity import note, reminder, timer, todo
from l3m_backend.tools.time import get_time
from l3m_backend.tools.utilities import (
    base64_encode,
    hash_text,
    json_format,
    random_number,
    uuid_generate,
)
from l3m_backend.tools.weather import get_weather

# Load user tools from ~/.l3m/tools/ after built-in tools
from l3m_backend.tools.loader import USER_TOOLS_DIR, load_user_tools

load_user_tools()

__all__ = [
    # Registry
    "registry",
    # User tools loader
    "load_user_tools",
    "USER_TOOLS_DIR",
    # Weather
    "get_weather",
    # Time
    "get_time",
    # Calculator
    "calculate",
    # Information
    "wikipedia",
    "define_word",
    "unit_convert",
    "currency_convert",
    "web_search",
    # Productivity
    "note",
    "todo",
    "reminder",
    "timer",
    # Development
    "run_python",
    "read_file",
    "write_file",
    "shell_cmd",
    "http_request",
    # Utilities
    "random_number",
    "uuid_generate",
    "hash_text",
    "base64_encode",
    "json_format",
]
