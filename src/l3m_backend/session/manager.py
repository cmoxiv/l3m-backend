"""
Session manager for l3m-chat.

Handles session CRUD operations, symlinks, and persistence.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from l3m_backend.session.data_models import (
    Session,
    SessionMessage,
    SessionMetadata,
    SessionSummary,
)

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine

# Commands that should not be saved to transcript
QUIT_COMMANDS = {"/quit", "/exit", "/q", "quit", "exit"}


class SessionManager:
    """Manages session persistence and operations."""

    SESSIONS_DIR = Path.home() / ".l3m" / "sessions"
    ANONYMOUS_DIR = Path.home() / ".l3m" / "sessions" / "anonymous"
    INCOGNITO_DIR = Path("/tmp/l3m-sessions")
    SYMLINK_NAME = ".l3m-session"

    def __init__(self, session: Optional[Session] = None):
        """Initialize the session manager.

        Args:
            session: Optional existing session to manage
        """
        self.session = session
        self._current_path: Optional[Path] = None  # Track current file path for rename
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure session directories exist."""
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.ANONYMOUS_DIR.mkdir(parents=True, exist_ok=True)
        self.INCOGNITO_DIR.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename_part(self, text: str, max_len: int = 20) -> str:
        """Sanitize text for use in filename."""
        # Remove non-alphanumeric except spaces, convert to lowercase
        sanitized = re.sub(r"[^a-zA-Z0-9\s]", "", text).lower()
        # Replace spaces with hyphens, collapse multiple
        sanitized = re.sub(r"\s+", "-", sanitized.strip())
        # Truncate
        if len(sanitized) > max_len:
            sanitized = sanitized[:max_len].rstrip("-")
        return sanitized or "untitled"

    def _generate_session_filename(self, session: Session) -> str:
        """Generate descriptive filename for session.

        Format: {initial_datetime}-{save_datetime}-{tag|anonymous}-{title}-{id_short}.json
        Example: 250107-1423-250107-1530-coding-help-me-write-a-func-a1b2c3d4.json
        """
        meta = session.metadata

        # Use initial_datetime (YYMMDD-HHMM format)
        initial_dt = meta.initial_datetime

        # Use last_save_datetime or initial_datetime
        save_dt = meta.last_save_datetime or initial_dt

        # Get tag (single string, not list)
        tag = self._sanitize_filename_part(meta.tag or "anonymous", 15)

        # Get title or first user message preview
        if meta.title:
            title_part = self._sanitize_filename_part(meta.title, 25)
        else:
            title_part = "untitled"
            for msg in session.transcript:
                if msg.role == "user":
                    title_part = self._sanitize_filename_part(msg.content, 25)
                    break

        # Short ID (first 8 chars)
        short_id = meta.id[:8]

        return f"{initial_dt}-{save_dt}-{tag}-{title_part}-{short_id}.json"

    def _is_anonymous_session(self, session: Session) -> bool:
        """Check if a session is anonymous (no title set)."""
        return not session.metadata.title

    def _get_session_path(self, session: Session) -> Path:
        """Get the file path for a session."""
        filename = self._generate_session_filename(session)
        if session.metadata.is_incognito:
            return self.INCOGNITO_DIR / filename
        if self._is_anonymous_session(session):
            return self.ANONYMOUS_DIR / filename
        return self.SESSIONS_DIR / filename

    def _get_session_path_by_id(self, session_id: str) -> Optional[Path]:
        """Find session file by ID or path (checks all directories).

        Supports:
        - Full file path (e.g., /path/to/session.json)
        - Filename (e.g., 260110-...-59e1624e.json)
        - Title-ID format (e.g., my-chat-title-59e1624e)
        - Full UUID
        - Short ID (first 8 chars)
        """
        # Handle full path input
        if "/" in session_id or session_id.endswith(".json"):
            path = Path(session_id).expanduser()
            if path.exists():
                return path
            # If path doesn't exist, try extracting ID from filename
            # e.g., "260110-0033-...-59e1624e.json" -> "59e1624e"
            if "-" in path.stem:
                session_id = path.stem.rsplit("-", 1)[-1]
        # Handle title-id format (e.g., "my-chat-title-59e1624e")
        # But NOT full UUIDs (36 chars with specific format)
        elif "-" in session_id and len(session_id) > 9 and len(session_id) != 36:
            # Extract short ID from end (last 8 chars after last hyphen)
            session_id = session_id.rsplit("-", 1)[-1]

        short_id = session_id[:8] if len(session_id) > 8 else session_id

        # Search in normal sessions (excluding anonymous subdirectory)
        for path in self.SESSIONS_DIR.glob("*.json"):
            # Check if filename ends with -{short_id}.json
            if path.stem.endswith(f"-{short_id}"):
                return path

        # Search in anonymous sessions
        for path in self.ANONYMOUS_DIR.glob("*.json"):
            if path.stem.endswith(f"-{short_id}"):
                return path

        # Search in incognito sessions
        for path in self.INCOGNITO_DIR.glob("*.json"):
            if path.stem.endswith(f"-{short_id}"):
                return path

        return None

    # =========================================================================
    # Core Operations
    # =========================================================================

    def create(
        self,
        model_path: str,
        working_directory: str,
        n_ctx: int = 32768,
        incognito: bool = False,
        tag: Optional[str] = None,
        loaded_model_path: Optional[str] = None,
    ) -> Session:
        """Create a new session.

        Args:
            model_path: Path to the model being used (user-provided)
            working_directory: Current working directory
            n_ctx: Context size
            incognito: If True, store in /tmp
            tag: Optional tag (single string)
            loaded_model_path: Actual resolved model path

        Returns:
            The created Session
        """
        self.session = Session.create(
            model_path=model_path,
            loaded_model_path=loaded_model_path,
            working_directory=working_directory,
            n_ctx=n_ctx,
            is_incognito=incognito,
            tag=tag,
        )
        self._current_path = None  # Reset current path for new session
        return self.session

    def save(self, cwd: Optional[Path] = None) -> Path:
        """Save the current session to disk.

        If filename changed (due to title/tag update), the file is renamed
        and symlink updated.

        Args:
            cwd: Working directory for symlink update (optional)

        Returns:
            Path to the saved session file

        Raises:
            ValueError: If no session is loaded
        """
        if not self.session:
            raise ValueError("No session to save")

        # Update timestamps
        self.session.metadata.touch()
        self.session.metadata.update_save_datetime()

        new_path = self._get_session_path(self.session)

        # Handle file rename/move scenarios
        old_path = self._current_path or self._get_session_path_by_id(self.session.metadata.id)

        if old_path and old_path != new_path and old_path.exists():
            # File needs to be renamed/moved (title/tag changed, or moved from anonymous)
            old_path.unlink()

        # Write to new path
        new_path.write_text(self.session.model_dump_json(indent=2))
        self._current_path = new_path

        # Update symlink AFTER file is written (so target exists)
        if old_path and old_path != new_path and cwd:
            self.create_symlink(cwd)

        return new_path

    def load(self, session_id: str) -> Session:
        """Load a session by ID.

        Args:
            session_id: The session UUID

        Returns:
            The loaded Session

        Raises:
            FileNotFoundError: If session doesn't exist
        """
        path = self._get_session_path_by_id(session_id)
        if not path:
            raise FileNotFoundError(f"Session not found: {session_id}")

        data = json.loads(path.read_text())
        self.session = Session.model_validate(data)
        self._current_path = path  # Track loaded path for rename handling
        return self.session

    def delete(self, session_id: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id: The session UUID

        Returns:
            True if deleted, False if not found
        """
        path = self._get_session_path_by_id(session_id)
        if path and path.exists():
            path.unlink()
            return True
        return False

    # =========================================================================
    # Symlink Management
    # =========================================================================

    def create_symlink(self, cwd: Path) -> Optional[Path]:
        """Create a symlink in the working directory pointing to session file.

        Args:
            cwd: Current working directory

        Returns:
            Path to the symlink, or None if session not saved yet
        """
        if not self.session:
            return None

        session_path = self._get_session_path(self.session)
        symlink_path = cwd / self.SYMLINK_NAME

        # Remove existing symlink if it exists
        if symlink_path.is_symlink() or symlink_path.exists():
            symlink_path.unlink()

        # Create symlink only if session file exists
        if session_path.exists():
            symlink_path.symlink_to(session_path)
            return symlink_path

        return None

    def update_symlink(self, old_cwd: Path, new_cwd: Path) -> Optional[Path]:
        """Update symlink when working directory changes.

        Args:
            old_cwd: Previous working directory
            new_cwd: New working directory

        Returns:
            Path to the new symlink
        """
        # Remove old symlink
        old_symlink = old_cwd / self.SYMLINK_NAME
        if old_symlink.is_symlink():
            old_symlink.unlink()

        # Update session's working directory
        if self.session:
            self.session.metadata.working_directory = str(new_cwd)

        # Create new symlink
        return self.create_symlink(new_cwd)

    def find_session_in_cwd(self, cwd: Path) -> Optional[str]:
        """Check if a session symlink exists in the given directory.

        Args:
            cwd: Directory to check

        Returns:
            Session ID (short, 8 chars) if found, None otherwise
        """
        symlink_path = cwd / self.SYMLINK_NAME
        if symlink_path.is_symlink():
            target = symlink_path.resolve()
            if target.exists() and target.suffix == ".json":
                # Extract session ID from filename (last part after final hyphen)
                # Format: {YYMMDDHHMM}-{tag}-{prompt}-{id_short}.json
                parts = target.stem.rsplit("-", 1)
                if len(parts) == 2:
                    return parts[1]  # Return the short ID
                return target.stem  # Fallback for old format
        return None

    # =========================================================================
    # History Management
    # =========================================================================

    def add_message(
        self,
        role: str,
        content: str,
        to_history: bool = True,
        to_transcript: bool = True,
        working_directory: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[SessionMessage]:
        """Add a message to the current session.

        Args:
            role: Message role
            content: Message content
            to_history: Add to LLM history
            to_transcript: Add to display transcript
            working_directory: CWD when message was added
            **kwargs: Additional fields

        Returns:
            The created message, or None if no session or quit command
        """
        if not self.session:
            return None

        # Skip quit commands
        if content.strip().lower() in QUIT_COMMANDS:
            return None

        return self.session.add_message(
            role=role,
            content=content,
            to_history=to_history,
            to_transcript=to_transcript,
            working_directory=working_directory or self.session.metadata.working_directory,
            **kwargs,
        )

    def sync_from_engine(self, engine_history: list[dict[str, Any]]) -> None:
        """Sync session history from ChatEngine.

        Args:
            engine_history: ChatEngine.history list
        """
        if self.session:
            self.session.sync_history_from_engine(engine_history)

    def get_engine_history(self) -> list[dict[str, Any]]:
        """Get history in ChatEngine format.

        Returns:
            List of message dicts for ChatEngine
        """
        if not self.session:
            return []
        return self.session.get_engine_history()

    # =========================================================================
    # Title Generation
    # =========================================================================

    def generate_title(
        self,
        engine: "ChatEngine",
        max_messages: Optional[int] = None,
    ) -> Optional[str]:
        """Generate a title for the session using the model.

        Uses the full conversation context. For long conversations (>8 messages),
        generates a summary first, then creates a title from the summary.

        Args:
            engine: ChatEngine instance for generation
            max_messages: Optional limit on messages to consider (default: all)

        Returns:
            Generated title, or None if generation fails
        """
        if not self.session or not self.session.transcript:
            return None

        transcript = self.session.transcript
        if max_messages:
            transcript = transcript[:max_messages]

        # For long conversations, summarize first then generate title
        if len(transcript) > 8:
            # Create a condensed summary for title generation
            summary = self._summarize_for_title(transcript, engine)
            if summary:
                prompt = f"""Based on this conversation summary, generate a brief title (3-7 words, no quotes):

{summary}

Title:"""
            else:
                # Fallback: use beginning and end of conversation
                context_text = self._format_transcript_for_title(transcript)
                prompt = f"""Based on this conversation, generate a brief title (3-7 words, no quotes):

{context_text}

Title:"""
        else:
            # Short conversation: use all messages directly
            context_text = "\n".join(
                f"{msg.role}: {msg.content[:200]}" for msg in transcript
            )
            prompt = f"""Based on this conversation, generate a brief title (3-7 words, no quotes):

{context_text}

Title:"""

        try:
            # Generate without affecting session history
            engine.llm.reset()
            response = engine.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
            )
            title = response["choices"][0]["message"]["content"].strip()
            # Clean up title (remove quotes, extra punctuation)
            title = title.strip('"\'').strip()
            if title:
                self.session.metadata.title = title
                return title
        except Exception:
            pass
        return None

    def _summarize_for_title(
        self,
        transcript: list[SessionMessage],
        engine: "ChatEngine",
    ) -> Optional[str]:
        """Generate a condensed summary for title generation.

        Args:
            transcript: Messages to summarize
            engine: ChatEngine instance

        Returns:
            Summary text, or None if generation fails
        """
        # Use beginning, middle, and end of conversation for context
        context_parts = []

        # First 3 messages
        for msg in transcript[:3]:
            context_parts.append(f"{msg.role}: {msg.content[:150]}")

        # Middle sample (if long enough)
        if len(transcript) > 10:
            mid_idx = len(transcript) // 2
            for msg in transcript[mid_idx:mid_idx + 2]:
                context_parts.append(f"{msg.role}: {msg.content[:150]}")

        # Last 3 messages
        for msg in transcript[-3:]:
            context_parts.append(f"{msg.role}: {msg.content[:150]}")

        context_text = "\n".join(context_parts)

        prompt = f"""Summarize this conversation in 1-2 sentences (focus on the main topic/purpose):

{context_text}

Summary:"""

        try:
            engine.llm.reset()
            response = engine.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    def _format_transcript_for_title(
        self,
        transcript: list[SessionMessage],
    ) -> str:
        """Format transcript for title generation, including beginning and end.

        Args:
            transcript: Messages to format

        Returns:
            Formatted context text
        """
        parts = []

        # First 3 messages
        parts.append("--- Start of conversation ---")
        for msg in transcript[:3]:
            parts.append(f"{msg.role}: {msg.content[:200]}")

        # Last 3 messages (if different from first 3)
        if len(transcript) > 6:
            parts.append(f"\n--- End of conversation ({len(transcript)} messages total) ---")
            for msg in transcript[-3:]:
                parts.append(f"{msg.role}: {msg.content[:200]}")

        return "\n".join(parts)

    def generate_tag(self, engine: "ChatEngine") -> Optional[str]:
        """Generate a single-word tag for the session using the model.

        Args:
            engine: ChatEngine instance for generation

        Returns:
            Generated tag (single word), or None if generation fails
        """
        if not self.session or not self.session.transcript:
            return None

        # Use first few exchanges for context
        context_messages = self.session.transcript[:4]
        context_text = "\n".join(
            f"{msg.role}: {msg.content[:200]}" for msg in context_messages
        )

        prompt = f"""Summarize this conversation topic in ONE word (lowercase, no punctuation):

{context_text}

Tag:"""

        try:
            engine.llm.reset()
            response = engine.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
            )
            tag = response["choices"][0]["message"]["content"].strip()
            # Clean up: lowercase, remove punctuation, take first word
            tag = re.sub(r"[^a-zA-Z]", "", tag).lower()
            if tag:
                self.session.metadata.tag = tag
                return tag
        except Exception:
            pass
        return None

    def generate_summary(
        self,
        engine: "ChatEngine",
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> Optional[SessionSummary]:
        """Generate a summary of conversation messages.

        Args:
            engine: ChatEngine instance for generation
            start_idx: Start message index (inclusive)
            end_idx: End message index (exclusive), defaults to transcript length

        Returns:
            Generated SessionSummary, or None if generation fails
        """
        if not self.session or not self.session.transcript:
            return None

        if end_idx is None:
            end_idx = len(self.session.transcript)

        if start_idx >= end_idx:
            return None

        # Get messages to summarize
        messages = self.session.transcript[start_idx:end_idx]
        if not messages:
            return None

        # Build transcript text (limit content length)
        transcript_text = "\n".join(
            f"{msg.role}: {msg.content[:300]}" for msg in messages[:20]
        )

        prompt = f"""Summarize this conversation in 2-3 sentences:

{transcript_text}

Summary:"""

        try:
            engine.llm.reset()
            response = engine.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            summary_text = response["choices"][0]["message"]["content"].strip()
            if summary_text:
                summary = SessionSummary(
                    summary=summary_text,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
                self.session.metadata.summaries.append(summary)
                return summary
        except Exception:
            pass
        return None

    def should_regenerate(self, msg_count: int) -> bool:
        """Check if title/tag should be regenerated at this message count.

        Uses exponential timing: 1, 2, 4, 8, 16, 32...

        Args:
            msg_count: Current transcript message count

        Returns:
            True if regeneration should occur
        """
        if msg_count <= 0:
            return False
        # Power of 2 check: n & (n-1) == 0 for powers of 2
        return (msg_count & (msg_count - 1)) == 0

    def get_last_summary_end_idx(self) -> int:
        """Get the end index of the last summary.

        Returns:
            End index of last summary, or 0 if no summaries
        """
        if not self.session or not self.session.metadata.summaries:
            return 0
        return self.session.metadata.summaries[-1].end_idx

    # =========================================================================
    # Session Listing and Search
    # =========================================================================

    def list_sessions(
        self,
        tag: Optional[str] = None,
        cwd: Optional[str] = None,
        include_incognito: bool = False,
        include_anonymous: bool = True,
    ) -> list[SessionMetadata]:
        """List available sessions.

        Args:
            tag: Filter by tag
            cwd: Filter by working directory
            include_incognito: Include incognito sessions
            include_anonymous: Include anonymous sessions (default True)

        Returns:
            List of session metadata, sorted by updated_at descending
        """
        results: list[SessionMetadata] = []

        # Collect from normal sessions (excludes anonymous subdirectory)
        for path in self.SESSIONS_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                session = Session.model_validate(data)
                metadata = session.metadata

                # Apply filters
                if tag and metadata.tag != tag:
                    continue
                if cwd and metadata.working_directory != cwd:
                    continue

                results.append(metadata)
            except Exception:
                continue

        # Collect from anonymous sessions if requested
        if include_anonymous:
            for path in self.ANONYMOUS_DIR.glob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    session = Session.model_validate(data)
                    metadata = session.metadata

                    if tag and metadata.tag != tag:
                        continue
                    if cwd and metadata.working_directory != cwd:
                        continue

                    results.append(metadata)
                except Exception:
                    continue

        # Collect from incognito sessions if requested
        if include_incognito:
            for path in self.INCOGNITO_DIR.glob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    session = Session.model_validate(data)
                    metadata = session.metadata

                    if tag and metadata.tag != tag:
                        continue
                    if cwd and metadata.working_directory != cwd:
                        continue

                    results.append(metadata)
                except Exception:
                    continue

        # Sort by updated_at descending
        return sorted(results, key=lambda m: m.updated_at, reverse=True)

    def search(
        self,
        query: str,
        include_incognito: bool = False,
        include_anonymous: bool = True,
    ) -> list[SessionMetadata]:
        """Search sessions by title, tag, or content.

        Args:
            query: Search query
            include_incognito: Include incognito sessions
            include_anonymous: Include anonymous sessions (default True)

        Returns:
            List of matching session metadata
        """
        query_lower = query.lower()
        results: list[SessionMetadata] = []

        dirs = [self.SESSIONS_DIR]
        if include_anonymous:
            dirs.append(self.ANONYMOUS_DIR)
        if include_incognito:
            dirs.append(self.INCOGNITO_DIR)

        for sessions_dir in dirs:
            for path in sessions_dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    session = Session.model_validate(data)

                    # Search in title
                    if session.metadata.title and query_lower in session.metadata.title.lower():
                        results.append(session.metadata)
                        continue

                    # Search in tag
                    if session.metadata.tag and query_lower in session.metadata.tag.lower():
                        results.append(session.metadata)
                        continue

                    # Search in transcript content
                    for msg in session.transcript:
                        if query_lower in msg.content.lower():
                            results.append(session.metadata)
                            break
                except Exception:
                    continue

        return sorted(results, key=lambda m: m.updated_at, reverse=True)

    # =========================================================================
    # Multi-Session Loading
    # =========================================================================

    def summarize_session(self, session_id: str, engine: "ChatEngine") -> Optional[str]:
        """Generate a summary of a session using the model.

        Args:
            session_id: ID of session to summarize
            engine: ChatEngine for generation

        Returns:
            Summary text, or None if failed
        """
        path = self._get_session_path_by_id(session_id)
        if not path:
            return None

        try:
            data = json.loads(path.read_text())
            session = Session.model_validate(data)

            # Build transcript text (limit to prevent overflow)
            transcript_text = "\n".join(
                f"{msg.role}: {msg.content[:300]}"
                for msg in session.transcript[:20]  # Limit messages
            )

            prompt = f"""Summarize this conversation in 2-3 sentences:

{transcript_text}

Summary:"""

            engine.llm.reset()
            response = engine.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    def load_as_context(
        self,
        session_ids: list[str],
        engine: "ChatEngine",
    ) -> list[str]:
        """Load multiple sessions as summaries into current context.

        Args:
            session_ids: List of session IDs to load
            engine: ChatEngine instance

        Returns:
            List of loaded session titles/IDs
        """
        loaded: list[str] = []

        for sid in session_ids:
            path = self._get_session_path_by_id(sid)
            if not path:
                continue

            try:
                data = json.loads(path.read_text())
                session = Session.model_validate(data)
                title = session.get_display_title()

                # Generate summary
                summary = self.summarize_session(sid, engine)
                if not summary:
                    summary = f"(Previous session: {title})"

                # Add to current session transcript
                context_msg = f"[Loaded session: {title}]\n{summary}"
                self.add_message(
                    role="user",
                    content=context_msg,
                    to_history=True,
                    to_transcript=True,
                )

                # Also add to engine history
                engine.history.append({
                    "role": "user",
                    "content": f"[Context from previous session: {title}]\n{summary}",
                })

                loaded.append(title)
            except Exception:
                continue

        return loaded
