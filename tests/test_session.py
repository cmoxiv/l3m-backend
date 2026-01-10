#!/usr/bin/env python3
"""
Tests for session management module.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from l3m_backend.session import Session, SessionManager, SessionMessage, SessionMetadata


# ============================================================================
# SessionMessage Tests
# ============================================================================

class TestSessionMessage:
    """Tests for SessionMessage model."""

    def test_create_message(self):
        """Test creating a basic message."""
        msg = SessionMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp  # Should have default timestamp

    def test_message_with_tool_call(self):
        """Test message with tool call fields."""
        msg = SessionMessage(
            role="assistant",
            content="",
            tool_calls=[{"name": "get_time", "arguments": {}}]
        )
        assert msg.tool_calls is not None

    def test_from_engine_message(self):
        """Test converting from ChatEngine message format."""
        engine_msg = {
            "role": "user",
            "content": "Test message",
        }
        msg = SessionMessage.from_engine_message(engine_msg)
        assert msg.role == "user"
        assert msg.content == "Test message"

    def test_to_engine_message(self):
        """Test converting to ChatEngine message format."""
        msg = SessionMessage(role="assistant", content="Response")
        engine_msg = msg.to_engine_message()
        assert engine_msg["role"] == "assistant"
        assert engine_msg["content"] == "Response"


# ============================================================================
# SessionMetadata Tests
# ============================================================================

class TestSessionMetadata:
    """Tests for SessionMetadata model."""

    def test_create_metadata(self):
        """Test creating metadata with defaults."""
        meta = SessionMetadata(
            model_path="/path/to/model.gguf",
            working_directory="/home/user/project"
        )
        assert meta.id  # UUID should be generated
        assert meta.model_path == "/path/to/model.gguf"
        assert meta.working_directory == "/home/user/project"
        assert meta.tag is None
        assert meta.n_ctx == 32768
        assert not meta.is_incognito

    def test_metadata_with_tag(self):
        """Test metadata with custom tag."""
        meta = SessionMetadata(
            model_path="/path/to/model.gguf",
            working_directory="/home/user",
            tag="project-x"
        )
        assert meta.tag == "project-x"

    def test_touch_updates_timestamp(self):
        """Test that touch updates updated_at."""
        meta = SessionMetadata(
            model_path="/path/to/model.gguf",
            working_directory="/home/user"
        )
        original = meta.updated_at
        import time
        time.sleep(0.01)
        meta.touch()
        assert meta.updated_at != original


# ============================================================================
# Session Tests
# ============================================================================

class TestSession:
    """Tests for Session model."""

    def test_create_session(self):
        """Test creating a session with factory method."""
        session = Session.create(
            model_path="/path/to/model.gguf",
            working_directory="/home/user/project",
            n_ctx=4096
        )
        assert session.metadata.model_path == "/path/to/model.gguf"
        assert session.metadata.n_ctx == 4096
        assert session.metadata.tag is None  # No tag by default
        assert session.history == []
        assert session.transcript == []

    def test_add_message_to_both(self):
        """Test adding message to both history and transcript."""
        session = Session.create(
            model_path="/path/to/model.gguf",
            working_directory="/home/user"
        )
        session.add_message("user", "Hello")
        assert len(session.history) == 1
        assert len(session.transcript) == 1
        assert session.history[0].content == "Hello"

    def test_add_message_transcript_only(self):
        """Test adding message to transcript only."""
        session = Session.create(
            model_path="/path/to/model.gguf",
            working_directory="/home/user"
        )
        session.add_message("user", "Hidden", to_history=False, to_transcript=True)
        assert len(session.history) == 0
        assert len(session.transcript) == 1

    def test_sync_history_from_engine(self):
        """Test syncing history from ChatEngine format."""
        session = Session.create(
            model_path="/path/to/model.gguf",
            working_directory="/home/user"
        )
        engine_history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
        ]
        session.sync_history_from_engine(engine_history)
        assert len(session.history) == 2
        assert session.history[0].content == "msg1"

    def test_get_engine_history(self):
        """Test converting history to ChatEngine format."""
        session = Session.create(
            model_path="/path/to/model.gguf",
            working_directory="/home/user"
        )
        session.add_message("user", "Hello")
        engine_history = session.get_engine_history()
        assert len(engine_history) == 1
        assert engine_history[0]["role"] == "user"
        assert engine_history[0]["content"] == "Hello"

    def test_get_display_title_with_title(self):
        """Test display title when title is set."""
        session = Session.create(
            model_path="/path/to/model.gguf",
            working_directory="/home/user"
        )
        session.metadata.title = "My Chat"
        assert session.get_display_title() == "My Chat"

    def test_get_display_title_from_first_message(self):
        """Test display title from first user message."""
        session = Session.create(
            model_path="/path/to/model.gguf",
            working_directory="/home/user"
        )
        session.add_message("user", "Tell me about Python", to_history=False)
        assert "Tell me about Python" in session.get_display_title()


# ============================================================================
# SessionManager Tests
# ============================================================================

class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture
    def temp_sessions_dir(self, tmp_path):
        """Create temporary session directories."""
        sessions_dir = tmp_path / "sessions"
        anonymous_dir = tmp_path / "sessions" / "anonymous"
        incognito_dir = tmp_path / "incognito"
        sessions_dir.mkdir()
        anonymous_dir.mkdir()
        incognito_dir.mkdir()
        return sessions_dir, anonymous_dir, incognito_dir

    def test_create_session(self, temp_sessions_dir):
        """Test creating a new session."""
        sessions_dir, anonymous_dir, incognito_dir = temp_sessions_dir

        with patch.object(SessionManager, 'SESSIONS_DIR', sessions_dir):
            with patch.object(SessionManager, 'ANONYMOUS_DIR', anonymous_dir):
                with patch.object(SessionManager, 'INCOGNITO_DIR', incognito_dir):
                    mgr = SessionManager()
                    session = mgr.create(
                        model_path="/path/to/model.gguf",
                        working_directory="/home/user",
                        n_ctx=4096
                    )
                    assert session is not None
                    assert session.metadata.model_path == "/path/to/model.gguf"

    def test_save_and_load_session(self, temp_sessions_dir):
        """Test saving and loading a session."""
        sessions_dir, anonymous_dir, incognito_dir = temp_sessions_dir

        with patch.object(SessionManager, 'SESSIONS_DIR', sessions_dir):
            with patch.object(SessionManager, 'ANONYMOUS_DIR', anonymous_dir):
                with patch.object(SessionManager, 'INCOGNITO_DIR', incognito_dir):
                    # Create and save
                    mgr = SessionManager()
                    session = mgr.create(
                        model_path="/path/to/model.gguf",
                        working_directory="/home/user"
                    )
                    session.add_message("user", "Hello")
                    path = mgr.save()
                    assert path.exists()

                    # Load
                    session_id = session.metadata.id
                    mgr2 = SessionManager()
                    loaded = mgr2.load(session_id)
                    assert loaded.metadata.id == session_id
                    assert len(loaded.history) == 1
                    assert loaded.history[0].content == "Hello"

    def test_incognito_session(self, temp_sessions_dir):
        """Test incognito session is saved to temp dir."""
        sessions_dir, anonymous_dir, incognito_dir = temp_sessions_dir

        with patch.object(SessionManager, 'SESSIONS_DIR', sessions_dir):
            with patch.object(SessionManager, 'ANONYMOUS_DIR', anonymous_dir):
                with patch.object(SessionManager, 'INCOGNITO_DIR', incognito_dir):
                    mgr = SessionManager()
                    session = mgr.create(
                        model_path="/path/to/model.gguf",
                        working_directory="/home/user",
                        incognito=True
                    )
                    path = mgr.save()
                    assert incognito_dir in path.parents or path.parent == incognito_dir

    def test_anonymous_session(self, temp_sessions_dir):
        """Test anonymous session (no title) is saved to anonymous dir."""
        sessions_dir, anonymous_dir, incognito_dir = temp_sessions_dir

        with patch.object(SessionManager, 'SESSIONS_DIR', sessions_dir):
            with patch.object(SessionManager, 'ANONYMOUS_DIR', anonymous_dir):
                with patch.object(SessionManager, 'INCOGNITO_DIR', incognito_dir):
                    mgr = SessionManager()
                    session = mgr.create(
                        model_path="/path/to/model.gguf",
                        working_directory="/home/user"
                    )
                    # No title set, should go to anonymous dir
                    path = mgr.save()
                    assert anonymous_dir in path.parents or path.parent == anonymous_dir

    def test_titled_session_not_anonymous(self, temp_sessions_dir):
        """Test titled session is saved to main sessions dir."""
        sessions_dir, anonymous_dir, incognito_dir = temp_sessions_dir

        with patch.object(SessionManager, 'SESSIONS_DIR', sessions_dir):
            with patch.object(SessionManager, 'ANONYMOUS_DIR', anonymous_dir):
                with patch.object(SessionManager, 'INCOGNITO_DIR', incognito_dir):
                    mgr = SessionManager()
                    session = mgr.create(
                        model_path="/path/to/model.gguf",
                        working_directory="/home/user"
                    )
                    session.metadata.title = "My Chat"
                    path = mgr.save()
                    # Should be in main sessions dir, not anonymous
                    assert path.parent == sessions_dir

    def test_list_sessions(self, temp_sessions_dir):
        """Test listing sessions."""
        sessions_dir, anonymous_dir, incognito_dir = temp_sessions_dir

        with patch.object(SessionManager, 'SESSIONS_DIR', sessions_dir):
            with patch.object(SessionManager, 'ANONYMOUS_DIR', anonymous_dir):
                with patch.object(SessionManager, 'INCOGNITO_DIR', incognito_dir):
                    # Create some sessions with titles (go to main dir)
                    mgr = SessionManager()
                    for i in range(3):
                        session = mgr.create(
                            model_path="/path/to/model.gguf",
                            working_directory="/home/user",
                            tag=f"tag{i}"
                        )
                        session.metadata.title = f"Session {i}"
                        mgr.save()

                    # List all (includes anonymous by default)
                    sessions = mgr.list_sessions()
                    assert len(sessions) == 3

                    # Filter by tag
                    sessions = mgr.list_sessions(tag="tag1")
                    assert len(sessions) == 1

    def test_search_sessions(self, temp_sessions_dir):
        """Test searching sessions."""
        sessions_dir, anonymous_dir, incognito_dir = temp_sessions_dir

        with patch.object(SessionManager, 'SESSIONS_DIR', sessions_dir):
            with patch.object(SessionManager, 'ANONYMOUS_DIR', anonymous_dir):
                with patch.object(SessionManager, 'INCOGNITO_DIR', incognito_dir):
                    mgr = SessionManager()

                    # Create session with title
                    session = mgr.create(
                        model_path="/path/to/model.gguf",
                        working_directory="/home/user"
                    )
                    session.metadata.title = "Python Discussion"
                    mgr.save()

                    # Create another
                    session2 = mgr.create(
                        model_path="/path/to/model.gguf",
                        working_directory="/home/user"
                    )
                    session2.metadata.title = "JavaScript Tutorial"
                    mgr.save()

                    # Search
                    results = mgr.search("python")
                    assert len(results) == 1
                    assert "Python" in results[0].title

    def test_symlink_creation(self, temp_sessions_dir, tmp_path):
        """Test symlink creation in working directory."""
        sessions_dir, anonymous_dir, incognito_dir = temp_sessions_dir
        cwd = tmp_path / "project"
        cwd.mkdir()

        with patch.object(SessionManager, 'SESSIONS_DIR', sessions_dir):
            with patch.object(SessionManager, 'ANONYMOUS_DIR', anonymous_dir):
                with patch.object(SessionManager, 'INCOGNITO_DIR', incognito_dir):
                    mgr = SessionManager()
                    mgr.create(
                        model_path="/path/to/model.gguf",
                        working_directory=str(cwd)
                    )
                    mgr.save()
                    symlink = mgr.create_symlink(cwd)

                    assert symlink is not None
                    assert symlink.is_symlink()

    def test_find_session_in_cwd(self, temp_sessions_dir, tmp_path):
        """Test finding session from symlink."""
        sessions_dir, anonymous_dir, incognito_dir = temp_sessions_dir
        cwd = tmp_path / "project"
        cwd.mkdir()

        with patch.object(SessionManager, 'SESSIONS_DIR', sessions_dir):
            with patch.object(SessionManager, 'ANONYMOUS_DIR', anonymous_dir):
                with patch.object(SessionManager, 'INCOGNITO_DIR', incognito_dir):
                    mgr = SessionManager()
                    session = mgr.create(
                        model_path="/path/to/model.gguf",
                        working_directory=str(cwd)
                    )
                    mgr.save()
                    mgr.create_symlink(cwd)

                    # Find it - returns short ID (first 8 chars)
                    found_id = mgr.find_session_in_cwd(cwd)
                    assert found_id == session.metadata.id[:8]

    def test_delete_session(self, temp_sessions_dir):
        """Test deleting a session."""
        sessions_dir, anonymous_dir, incognito_dir = temp_sessions_dir

        with patch.object(SessionManager, 'SESSIONS_DIR', sessions_dir):
            with patch.object(SessionManager, 'ANONYMOUS_DIR', anonymous_dir):
                with patch.object(SessionManager, 'INCOGNITO_DIR', incognito_dir):
                    mgr = SessionManager()
                    session = mgr.create(
                        model_path="/path/to/model.gguf",
                        working_directory="/home/user"
                    )
                    mgr.save()
                    session_id = session.metadata.id

                    # Delete
                    result = mgr.delete(session_id)
                    assert result is True

                    # Should not be findable
                    sessions = mgr.list_sessions()
                    assert len(sessions) == 0

    def test_session_moves_when_titled(self, temp_sessions_dir):
        """Test that a session moves from anonymous to main when titled."""
        sessions_dir, anonymous_dir, incognito_dir = temp_sessions_dir

        with patch.object(SessionManager, 'SESSIONS_DIR', sessions_dir):
            with patch.object(SessionManager, 'ANONYMOUS_DIR', anonymous_dir):
                with patch.object(SessionManager, 'INCOGNITO_DIR', incognito_dir):
                    mgr = SessionManager()
                    session = mgr.create(
                        model_path="/path/to/model.gguf",
                        working_directory="/home/user"
                    )
                    # Save without title (goes to anonymous)
                    path1 = mgr.save()
                    assert path1.parent == anonymous_dir

                    # Now add a title and save again
                    session.metadata.title = "Now I have a title"
                    path2 = mgr.save()

                    # Should now be in main sessions dir
                    assert path2.parent == sessions_dir
                    # Old file should be gone
                    assert not path1.exists()


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
