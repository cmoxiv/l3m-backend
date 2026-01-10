"""
Storage utilities for persistent tool data.
"""

from pathlib import Path

_STORAGE_DIR = Path.home() / ".llm_tools"
_NOTES_FILE = _STORAGE_DIR / "notes.json"
_TODOS_FILE = _STORAGE_DIR / "todos.json"
_REMINDERS_FILE = _STORAGE_DIR / "reminders.json"


def _ensure_storage():
    """Ensure storage directory exists."""
    _STORAGE_DIR.mkdir(exist_ok=True)
