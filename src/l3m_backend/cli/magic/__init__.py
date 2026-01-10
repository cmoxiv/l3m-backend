"""
Magic command system for l3m-chat REPL.

Magic commands are % prefixed and typically add to conversation history.
Commands are loaded from:
1. ~/.l3m/magic/ (user-hackable)
2. Package builtins (fallback)
"""

from __future__ import annotations

from l3m_backend.cli.magic.registry import MagicRegistry, magic_registry
from l3m_backend.cli.magic.loader import load_all_magic, load_user_magic

__all__ = ["MagicRegistry", "magic_registry", "load_all_magic", "load_user_magic"]
