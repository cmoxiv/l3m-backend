"""
Session management module for l3m-chat.

Provides session persistence, history management, and search functionality.
"""

from l3m_backend.session.manager import SessionManager
from l3m_backend.session.data_models import (
    Session,
    SessionMessage,
    SessionMetadata,
    SessionSummary,
)

__all__ = [
    "Session",
    "SessionManager",
    "SessionMessage",
    "SessionMetadata",
    "SessionSummary",
]
