"""
Session data models for l3m-chat.

Pydantic models for session persistence with chat history and metadata.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionMessage(BaseModel):
    """A single message in a session."""

    role: str  # "user", "assistant", "tool", "system"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    working_directory: Optional[str] = None  # CWD when message was added
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None

    @classmethod
    def from_engine_message(cls, msg: dict[str, Any]) -> "SessionMessage":
        """Create SessionMessage from ChatEngine message format."""
        return cls(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            tool_call_id=msg.get("tool_call_id"),
            tool_calls=msg.get("tool_calls"),
        )

    def to_engine_message(self) -> dict[str, Any]:
        """Convert to ChatEngine message format."""
        msg: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class SessionSummary(BaseModel):
    """A summary of a portion of the conversation."""

    summary: str
    start_idx: int  # Start message index (inclusive)
    end_idx: int  # End message index (exclusive)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class SessionMetadata(BaseModel):
    """Metadata for a session."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    tag: Optional[str] = None  # Single tag (was tags: list[str])
    model_path: str  # Original user-provided model path
    loaded_model_path: Optional[str] = None  # Actual resolved path (e.g., /var/...)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    initial_datetime: str = Field(
        default_factory=lambda: datetime.now().strftime("%y%m%d-%H%M")
    )  # YYMMDD-HHMM format for filename
    last_save_datetime: Optional[str] = None  # YYMMDD-HHMM format
    working_directory: str
    n_ctx: int = 32768
    is_incognito: bool = False
    summaries: list[SessionSummary] = Field(default_factory=list)

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()

    def update_save_datetime(self) -> None:
        """Update the last_save_datetime to current time."""
        self.last_save_datetime = datetime.now().strftime("%y%m%d-%H%M")


class Session(BaseModel):
    """A complete session with metadata and message history."""

    metadata: SessionMetadata
    history: list[SessionMessage] = Field(default_factory=list)
    transcript: list[SessionMessage] = Field(default_factory=list)

    @classmethod
    def create(
        cls,
        model_path: str,
        working_directory: str,
        n_ctx: int = 32768,
        is_incognito: bool = False,
        tag: Optional[str] = None,
        loaded_model_path: Optional[str] = None,
    ) -> "Session":
        """Create a new session with the given parameters."""
        metadata = SessionMetadata(
            model_path=model_path,
            loaded_model_path=loaded_model_path,
            working_directory=working_directory,
            n_ctx=n_ctx,
            is_incognito=is_incognito,
            tag=tag,
        )
        return cls(metadata=metadata)

    def add_message(
        self,
        role: str,
        content: str,
        to_history: bool = True,
        to_transcript: bool = True,
        **kwargs: Any,
    ) -> SessionMessage:
        """Add a message to the session.

        Args:
            role: Message role (user, assistant, tool, system)
            content: Message content
            to_history: Add to LLM context history
            to_transcript: Add to full display transcript
            **kwargs: Additional fields (tool_call_id, tool_calls)

        Returns:
            The created SessionMessage
        """
        msg = SessionMessage(role=role, content=content, **kwargs)
        if to_history:
            self.history.append(msg)
        if to_transcript:
            self.transcript.append(msg)
        self.metadata.touch()
        return msg

    def sync_history_from_engine(self, engine_history: list[dict[str, Any]]) -> None:
        """Sync session history from ChatEngine history.

        Replaces session.history with converted engine messages.
        Does NOT affect transcript.
        """
        self.history = [
            SessionMessage.from_engine_message(msg) for msg in engine_history
        ]
        self.metadata.touch()

    def get_engine_history(self) -> list[dict[str, Any]]:
        """Convert session history to ChatEngine format."""
        return [msg.to_engine_message() for msg in self.history]

    def get_display_title(self) -> str:
        """Get a display title for the session."""
        if self.metadata.title:
            return self.metadata.title
        # Fallback: use first user message preview or ID
        for msg in self.transcript:
            if msg.role == "user":
                preview = msg.content[:40]
                if len(msg.content) > 40:
                    preview += "..."
                return preview
        return f"Session {self.metadata.id[:8]}"
