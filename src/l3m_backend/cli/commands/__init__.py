"""
Command system for l3m-chat REPL.

Commands are / prefixed actions that don't go to the LLM.
Commands are loaded from:
1. ~/.l3m/commands/ (user-hackable)
2. Package builtins (fallback)
"""

from __future__ import annotations

from l3m_backend.cli.commands.registry import CommandRegistry, command_registry
from l3m_backend.cli.commands.loader import load_all_commands, load_user_commands

__all__ = ["CommandRegistry", "command_registry", "load_all_commands", "load_user_commands"]
