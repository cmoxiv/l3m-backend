"""
Magic command registry for l3m-chat REPL.

Magic commands are registered with a name, handler function, and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@dataclass
class MagicEntry:
    """Entry for a registered magic command."""

    name: str
    handler: Callable[..., bool]
    description: str
    usage: str | None = None
    aliases: list[str] = field(default_factory=list)


class MagicRegistry:
    """Registry for magic commands."""

    def __init__(self):
        self._commands: dict[str, MagicEntry] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        description: str,
        usage: str | None = None,
        aliases: list[str] | None = None,
    ) -> Callable:
        """Decorator to register a magic command.

        Args:
            name: Command name without the % prefix (e.g., "hello")
            description: Short description for help
            usage: Usage string (e.g., "%hello [name]")
            aliases: Alternative names for the command

        Returns:
            Decorator function

        Example:
            @magic_registry.register("hello", "Say hello")
            def magic_hello(engine, session_mgr, args, add_to_history):
                output = f"Hello, {args or 'world'}!"
                print(output)
                add_to_history(f"%hello {args}", output)
                return True
        """
        def decorator(func: Callable) -> Callable:
            # Normalize name (replace hyphens with underscores) for consistent lookup
            normalized_name = name.replace("-", "_")
            entry = MagicEntry(
                name=name,  # Keep original for display
                handler=func,
                description=description,
                usage=usage or f"%{name}",
                aliases=aliases or [],
            )
            self._commands[normalized_name] = entry

            # Register aliases (normalized)
            for alias in entry.aliases:
                self._aliases[alias.replace("-", "_")] = normalized_name

            return func
        return decorator

    def get(self, name: str) -> MagicEntry | None:
        """Get a magic command by name or alias."""
        # Normalize name (replace hyphens with underscores)
        name = name.replace("-", "_")
        # Check direct name
        if name in self._commands:
            return self._commands[name]
        # Check aliases
        if name in self._aliases:
            return self._commands[self._aliases[name]]
        return None

    def all_commands(self) -> list[MagicEntry]:
        """Get all registered magic commands sorted by name."""
        return sorted(self._commands.values(), key=lambda e: e.name)

    def get_completions(self) -> dict[str, str]:
        """Get command names and descriptions for completion."""
        result = {}
        for entry in self._commands.values():
            result[f"%{entry.name}"] = entry.description
            for alias in entry.aliases:
                result[f"%{alias}"] = entry.description
        return result


# Global magic registry
magic_registry = MagicRegistry()
