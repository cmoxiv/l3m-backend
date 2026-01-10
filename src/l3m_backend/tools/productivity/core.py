"""
Productivity tools implementation - notes, todos, reminders, timer.
"""

import json
from datetime import datetime
from typing import Any, Literal

from l3m_backend.core import tool_output
from l3m_backend.tools._registry import registry
from l3m_backend.tools import _storage


# -----------------------------
# Note Tool
# -----------------------------

@registry.register(aliases=["notes"])
@tool_output(llm_format=lambda x: x.get('message', x.get('content', str(x))))
def note(
    action: Literal["save", "get", "list", "delete"],
    name: str = "",
    content: str = "",
) -> dict[str, Any]:
    """Manage persistent notes.

    Args:
        action: Operation to perform (save, get, list, delete).
        name: Name/key for the note.
        content: Content to save (for save action).

    Returns:
        Dictionary with result of the operation.
    """
    _storage._ensure_storage()

    # Load existing notes
    notes = {}
    if _storage._NOTES_FILE.exists():
        notes = json.loads(_storage._NOTES_FILE.read_text())

    if action == "save":
        if not name:
            return {"error": "Note name is required for save"}
        notes[name] = {"content": content, "updated": datetime.now().isoformat()}
        _storage._NOTES_FILE.write_text(json.dumps(notes, indent=2))
        return {"message": f"Note '{name}' saved"}

    elif action == "get":
        if not name:
            return {"error": "Note name is required for get"}
        if name in notes:
            return {"name": name, "content": notes[name]["content"]}
        return {"error": f"Note '{name}' not found"}

    elif action == "list":
        return {"notes": list(notes.keys()), "count": len(notes)}

    elif action == "delete":
        if not name:
            return {"error": "Note name is required for delete"}
        if name in notes:
            del notes[name]
            _storage._NOTES_FILE.write_text(json.dumps(notes, indent=2))
            return {"message": f"Note '{name}' deleted"}
        return {"error": f"Note '{name}' not found"}

    return {"error": f"Unknown action: {action}"}


# -----------------------------
# Todo Tool
# -----------------------------

@registry.register(aliases=["todos", "task", "tasks"])
@tool_output(llm_format=lambda x: x.get('message', '\n'.join(f"[{'x' if t.get('done') else ' '}] {t['task']}" for t in x.get('items', [])) if x.get('items') else str(x)))
def todo(
    action: Literal["add", "list", "done", "remove", "clear"],
    task: str = "",
) -> dict[str, Any]:
    """Manage a todo list.

    Args:
        action: Operation to perform.
        task: Task text (for add) or task number (for done/remove).

    Returns:
        Dictionary with result of the operation.
    """
    _storage._ensure_storage()

    # Load existing todos
    todos = []
    if _storage._TODOS_FILE.exists():
        todos = json.loads(_storage._TODOS_FILE.read_text())

    if action == "add":
        if not task:
            return {"error": "Task description is required"}
        todos.append({"task": task, "done": False, "created": datetime.now().isoformat()})
        _storage._TODOS_FILE.write_text(json.dumps(todos, indent=2))
        return {"message": f"Added: {task}", "total": len(todos)}

    elif action == "list":
        return {"items": todos, "total": len(todos), "pending": sum(1 for t in todos if not t["done"])}

    elif action == "done":
        try:
            idx = int(task) - 1  # 1-indexed for user
            if 0 <= idx < len(todos):
                todos[idx]["done"] = True
                _storage._TODOS_FILE.write_text(json.dumps(todos, indent=2))
                return {"message": f"Marked done: {todos[idx]['task']}"}
            return {"error": f"Invalid task number: {task}"}
        except ValueError:
            return {"error": "Task number must be an integer"}

    elif action == "remove":
        try:
            idx = int(task) - 1
            if 0 <= idx < len(todos):
                removed = todos.pop(idx)
                _storage._TODOS_FILE.write_text(json.dumps(todos, indent=2))
                return {"message": f"Removed: {removed['task']}"}
            return {"error": f"Invalid task number: {task}"}
        except ValueError:
            return {"error": "Task number must be an integer"}

    elif action == "clear":
        _storage._TODOS_FILE.write_text("[]")
        return {"message": "Todo list cleared"}

    return {"error": f"Unknown action: {action}"}


# -----------------------------
# Reminder Tool
# -----------------------------

@registry.register(aliases=["remind", "reminders"])
@tool_output(llm_format=lambda x: x.get('message', str(x)))
def reminder(
    action: Literal["add", "list", "remove"],
    text: str = "",
    time: str = "",
) -> dict[str, Any]:
    """Manage reminders.

    Note: This saves reminders persistently but does not actively notify.
    Use with a separate notification system.

    Args:
        action: Operation to perform.
        text: Reminder text.
        time: Time for reminder (for add action).

    Returns:
        Dictionary with result of the operation.
    """
    _storage._ensure_storage()

    # Load existing reminders
    reminders = []
    if _storage._REMINDERS_FILE.exists():
        reminders = json.loads(_storage._REMINDERS_FILE.read_text())

    if action == "add":
        if not text:
            return {"error": "Reminder text is required"}
        reminders.append({
            "text": text,
            "time": time or "unspecified",
            "created": datetime.now().isoformat(),
        })
        _storage._REMINDERS_FILE.write_text(json.dumps(reminders, indent=2))
        return {"message": f"Reminder set: '{text}' at {time or 'unspecified time'}"}

    elif action == "list":
        return {"reminders": reminders, "count": len(reminders)}

    elif action == "remove":
        try:
            idx = int(text) - 1
            if 0 <= idx < len(reminders):
                removed = reminders.pop(idx)
                _storage._REMINDERS_FILE.write_text(json.dumps(reminders, indent=2))
                return {"message": f"Removed reminder: {removed['text']}"}
            return {"error": f"Invalid reminder number: {text}"}
        except ValueError:
            return {"error": "Reminder number must be an integer"}

    return {"error": f"Unknown action: {action}"}


# -----------------------------
# Timer Tool
# -----------------------------

@registry.register(aliases=["countdown"])
@tool_output(llm_format=lambda x: x.get('message', str(x)))
def timer(
    seconds: int,
    label: str = "Timer",
) -> dict[str, Any]:
    """Create a countdown timer.

    Note: This returns timer info immediately. For actual countdown,
    integrate with a notification system.

    Args:
        seconds: Duration in seconds.
        label: Optional label for the timer.

    Returns:
        Dictionary with timer details.
    """
    end_time = datetime.now().timestamp() + seconds
    end_dt = datetime.fromtimestamp(end_time)

    minutes = seconds // 60
    secs = seconds % 60

    duration_str = f"{minutes}m {secs}s" if minutes else f"{secs}s"

    return {
        "message": f"Timer '{label}' set for {duration_str}",
        "label": label,
        "duration_seconds": seconds,
        "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
    }
