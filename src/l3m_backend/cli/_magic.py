"""Magic command handlers for the REPL.

Magic commands use the % prefix:
- Most magic commands add the command as user message and output as assistant response
- All commands are loaded from ~/.l3m/magic/ with package builtins as fallback
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from l3m_backend.cli.magic import magic_registry, load_all_magic

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager

# Load all magic commands (user + builtins)
load_all_magic()

# Magic commands for completion
MAGIC_COMMANDS = {
    "%!": "Run shell command (adds to history)",
    "%tool": "Invoke tool (adds to history)",
    "%load": "Load file content (adds to history)",
    "%time": "Report current time (adds to history)",
    "%save": "Save conversation (adds to history)",
    "%edit-response": "Edit last assistant response in $VISUAL",
}

# Add loaded magic commands to completion
MAGIC_COMMANDS.update(magic_registry.get_completions())


class MagicCommands:
    """Handler for % magic commands.

    Most magic commands add the command as a user message and the output
    as an assistant response. All commands are dispatched through the registry.
    """

    def __init__(self, engine: "ChatEngine", session_mgr: Optional["SessionManager"] = None):
        self.engine = engine
        self.session_mgr = session_mgr

    def _add_to_history(self, user_msg: str, assistant_msg: str) -> None:
        """Add a user/assistant exchange to history.

        The assistant message is wrapped in a final JSON structure in engine.history
        for consistency with the tool-calling contract. The session transcript
        stores the plain text for display purposes.
        """
        # Wrap assistant response in final JSON format for LLM context
        final_response = json.dumps({"type": "final", "content": assistant_msg})

        # Add to engine history (JSON wrapped for LLM)
        self.engine.history.append({"role": "user", "content": user_msg})
        self.engine.history.append({"role": "assistant", "content": final_response})

        # Sync with session if available
        if self.session_mgr:
            # Transcript stores plain text for display
            self.session_mgr.add_message("user", user_msg, to_history=False)
            self.session_mgr.add_message("assistant", assistant_msg, to_history=False)
            # History syncs from engine (has JSON wrapped responses)
            self.session_mgr.sync_from_engine(self.engine.history)
            self.session_mgr.save()

    def execute(self, command: str) -> bool:
        """Execute a magic command.

        Args:
            command: The command string without the leading %.

        Returns:
            True if command was handled, False otherwise.
        """
        if not command:
            self._show_help()
            return True

        # Handle %! specially - maps to "!" magic command
        if command.startswith("!"):
            cmd = "!"
            args = command[1:]
        else:
            # Parse command and args
            parts = command.split(None, 1)
            cmd = parts[0].lower().replace("-", "_")  # Handle hyphens in command names
            args = parts[1] if len(parts) > 1 else ""

        # Dispatch through registry (includes both user and builtin commands)
        entry = magic_registry.get(cmd)
        if entry:
            try:
                return entry.handler(self.engine, self.session_mgr, args, self._add_to_history)
            except Exception as e:
                print(f"Magic command error: {e}")
                return True

        print(f"Unknown magic command: %{command.split()[0] if command else ''}")
        print("Type %help for available commands.")
        return True

    def _show_help(self) -> None:
        """Show magic command help."""
        print("\nMagic Commands (% prefix):")
        for entry in magic_registry.all_commands():
            name = entry.name if entry.name != "!" else "!<cmd>"
            print(f"  %{name:<16} {entry.description}")
        print("\nFor session commands and other pure commands, use /help")
        print()
