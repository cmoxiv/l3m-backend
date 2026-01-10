"""
Command registry for l3m-chat REPL.

Commands are registered with a name, handler function, and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Any

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@dataclass
class CommandEntry:
    """Entry for a registered command."""

    name: str
    handler: Callable[..., bool | None]
    description: str
    usage: str | None = None
    has_args: bool = False
    aliases: list[str] = field(default_factory=list)


class CommandRegistry:
    """Registry for REPL commands."""

    def __init__(self):
        self._commands: dict[str, CommandEntry] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        description: str,
        usage: str | None = None,
        has_args: bool = False,
        aliases: list[str] | None = None,
    ) -> Callable:
        """Decorator to register a command.

        Args:
            name: Command name without the / prefix (e.g., "clear")
            description: Short description for /help
            usage: Usage string (e.g., "/pop [n]")
            has_args: True if command accepts arguments (uses startswith matching)
            aliases: Alternative names for the command

        Returns:
            Decorator function

        Example:
            @command_registry.register("clear", "Clear conversation history")
            def cmd_clear(engine, session_mgr, args):
                engine.clear()
                print("Conversation cleared.\\n")
        """
        def decorator(func: Callable) -> Callable:
            entry = CommandEntry(
                name=name,
                handler=func,
                description=description,
                usage=usage or f"/{name}",
                has_args=has_args,
                aliases=aliases or [],
            )
            self._commands[name] = entry

            # Register aliases
            for alias in entry.aliases:
                self._aliases[alias] = name

            return func
        return decorator

    def get(self, name: str) -> CommandEntry | None:
        """Get a command by name or alias."""
        # Check direct name
        if name in self._commands:
            return self._commands[name]
        # Check aliases
        if name in self._aliases:
            return self._commands[self._aliases[name]]
        return None

    def match(self, cmd: str) -> tuple[CommandEntry | None, str]:
        """Match a command string to a command entry.

        Args:
            cmd: The full command string (e.g., "/pop 3" or "/clear")

        Returns:
            Tuple of (CommandEntry or None, remaining args string)
        """
        if not cmd.startswith("/"):
            return None, ""

        # Extract command name (first word after /)
        parts = cmd[1:].split(maxsplit=1)
        cmd_name = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        # Try exact match first
        entry = self.get(cmd_name)
        if entry:
            return entry, args

        # Try prefix match for commands with has_args=True
        for name, entry in self._commands.items():
            if entry.has_args and cmd_name.startswith(name):
                # The "args" includes any suffix after the command name
                return entry, cmd[1 + len(name):].strip()

        return None, ""

    def all_commands(self) -> list[CommandEntry]:
        """Get all registered commands sorted by name."""
        return sorted(self._commands.values(), key=lambda e: e.name)

    def get_completions(self) -> dict[str, str]:
        """Get command names and descriptions for completion."""
        result = {}
        for entry in self._commands.values():
            result[f"/{entry.name}"] = entry.description
            for alias in entry.aliases:
                result[f"/{alias}"] = entry.description
        return result


# Global command registry
command_registry = CommandRegistry()
