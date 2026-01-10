"""
Context partitioning for session loading.

Allows splitting the context window into dedicated sections for:
- System prompt + tools contract
- Previous session summaries
- Transcript excerpts
- Live conversation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
