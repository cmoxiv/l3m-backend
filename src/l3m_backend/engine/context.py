"""
Context management for chat engine.

Provides:
- Context partitioning for session loading (splitting context window)
- Engine context for tool execution (allowing tools to access the engine)
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from l3m_backend.engine.chat import ChatEngine

# Context variable to hold the current engine during tool execution
_current_engine: ContextVar["ChatEngine | None"] = ContextVar("current_engine", default=None)


def get_current_engine() -> "ChatEngine | None":
    """Get the current ChatEngine from execution context.

    Returns the ChatEngine instance that is currently executing a tool,
    or None if called outside of tool execution context.

    Returns:
        The current ChatEngine instance, or None.
    """
    return _current_engine.get()


def set_current_engine(engine: "ChatEngine | None") -> Any:
    """Set the current ChatEngine in execution context.

    Args:
        engine: The ChatEngine to set as current.

    Returns:
        Token that can be used to reset the context.
    """
    return _current_engine.set(engine)


def reset_current_engine(token: Any) -> None:
    """Reset the engine context to its previous value.

    Args:
        token: Token returned from set_current_engine.
    """
    _current_engine.reset(token)


@dataclass
class ContextPartition:
    """Configuration for partitioned context loading.

    The total context window is divided into sections:
    - system_tokens: Fixed space for system prompt + tools contract
    - summary_tokens: Space for loaded session summaries (--summary-ctx)
    - transcript_tokens: Space for loaded transcript excerpts (--transcript-ctx)
    - reserve_tokens: Reserved for generation output

    The remaining space is used for the live conversation history.

    Attributes:
        system_tokens: Tokens reserved for system prompt + contract (default 2048)
        summary_tokens: Tokens for summaries (0=disabled, >0=fixed, -1=add 4096)
        transcript_tokens: Tokens for transcripts (0=disabled, >0=fixed, -1=add 4096)
        reserve_tokens: Tokens reserved for generation (default 2048)
    """

    system_tokens: int = 2048
    summary_tokens: int = 0
    transcript_tokens: int = 0
    reserve_tokens: int = 2048

    # Content loaded into each partition
    loaded_summaries: list[str] = field(default_factory=list)
    loaded_transcript: list[dict[str, Any]] = field(default_factory=list)

    def resolve_size(self, value: int, default_add: int = 4096) -> int:
        """Resolve a partition size value.

        Args:
            value: The user-specified value:
                   0 = disabled
                   >0 = fixed size
                   -1 = add default_add to base
            default_add: Amount to add when value is -1 (default 4096)

        Returns:
            Resolved token count (0 if disabled)
        """
        if value == 0:
            return 0
        elif value == -1:
            return default_add
        else:
            return value

    def live_tokens(self, total_ctx: int) -> int:
        """Calculate available tokens for live conversation.

        Args:
            total_ctx: Total context window size

        Returns:
            Tokens available for live conversation history
        """
        used = (
            self.system_tokens
            + self.resolve_size(self.summary_tokens)
            + self.resolve_size(self.transcript_tokens)
            + self.reserve_tokens
        )
        return max(0, total_ctx - used)

    def has_partitions(self) -> bool:
        """Check if any partitions are enabled."""
        return self.summary_tokens != 0 or self.transcript_tokens != 0

    def clear(self) -> None:
        """Clear loaded content."""
        self.loaded_summaries = []
        self.loaded_transcript = []
