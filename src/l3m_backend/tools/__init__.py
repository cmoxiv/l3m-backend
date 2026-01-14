"""
Tools and shared registry for the l3m_backend package.

Tools are loaded ONLY from ~/.l3m/tools/. Use `l3m-init` to create
symlinks to package builtins. Users can:
- Remove symlinks to disable built-in tools
- Replace symlinks with custom implementations
"""

# Import registry first
from l3m_backend.tools._registry import registry

# Load tools from ~/.l3m/tools/ (symlinks to builtins or custom)
from l3m_backend.tools.loader import (
    USER_TOOLS_DIR,
    PACKAGE_TOOLS_DIR,
    load_all_tools,
    load_user_tools,  # Backwards compatibility
)

load_all_tools()

__all__ = [
    # Registry
    "registry",
    # Tools loader
    "load_all_tools",
    "load_user_tools",  # Backwards compatibility
    "USER_TOOLS_DIR",
    "PACKAGE_TOOLS_DIR",
]
