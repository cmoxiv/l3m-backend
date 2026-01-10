"""
Development tools implementation - Python execution, file operations, shell commands, HTTP requests.
"""

import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

from l3m_backend.core import tool_output
from l3m_backend.tools._registry import registry


# -----------------------------
# Python Execution Tool
# -----------------------------

@registry.register(aliases=["python", "py", "exec"])
@tool_output(llm_format=lambda x: f"Output:\n{x.get('output', x.get('error', 'No output'))}")
def run_python(
    code: str,
    timeout: int = 10,
) -> dict[str, Any]:
    """Execute Python code in a subprocess.

    Runs code in an isolated subprocess with timeout protection.

    Args:
        code: Python code to execute.
        timeout: Maximum execution time (default 10s, max 30s).

    Returns:
        Dictionary with stdout output or error message.

    Security Note:
        Code runs in a subprocess but is NOT fully sandboxed.
        Do not run untrusted code.
    """
    timeout = min(timeout, 30)  # Cap at 30 seconds

    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}" if output else result.stderr

        return {
            "output": output.strip() or "(no output)",
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Execution timed out after {timeout} seconds"}
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# File Read Tool
# -----------------------------

@registry.register(aliases=["read", "cat"])
@tool_output(llm_format=lambda x: x.get('content', x.get('error', 'No content')))
def read_file(
    path: str,
    max_lines: int = 100,
) -> dict[str, Any]:
    """Read contents of a file.

    Args:
        path: Path to the file (relative or absolute).
        max_lines: Maximum number of lines to return (default 100).

    Returns:
        Dictionary with file content or error.
    """
    try:
        file_path = Path(path).expanduser().resolve()

        if not file_path.exists():
            return {"error": f"File not found: {path}"}

        if not file_path.is_file():
            return {"error": f"Not a file: {path}"}

        # Check file size (limit to 1MB)
        if file_path.stat().st_size > 1_000_000:
            return {"error": "File too large (>1MB)"}

        content = file_path.read_text()
        lines = content.splitlines()

        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines])
            return {
                "content": content,
                "truncated": True,
                "total_lines": len(lines),
                "shown_lines": max_lines,
            }

        return {"content": content, "lines": len(lines)}
    except PermissionError:
        return {"error": f"Permission denied: {path}"}
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# File Write Tool
# -----------------------------

@registry.register(aliases=["write", "save_file"])
@tool_output(llm_format=lambda x: x.get('message', x.get('error', 'Unknown result')))
def write_file(
    path: str,
    content: str,
    append: bool = False,
) -> dict[str, Any]:
    """Write content to a file.

    Args:
        path: Path to the file.
        content: Content to write.
        append: If True, append to file instead of overwriting.

    Returns:
        Dictionary with result or error.
    """
    try:
        file_path = Path(path).expanduser().resolve()

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        with open(file_path, mode) as f:
            f.write(content)

        action = "Appended to" if append else "Wrote"
        return {
            "message": f"{action} {file_path}",
            "bytes": len(content),
            "path": str(file_path),
        }
    except PermissionError:
        return {"error": f"Permission denied: {path}"}
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Shell Command Tool
# -----------------------------

_SHELL_BLACKLIST = {"rm", "sudo", "mkfs", "dd", "format", ">", ">>", "|"}


@registry.register(aliases=["shell", "sh", "cmd"])
@tool_output(llm_format=lambda x: x.get('output', x.get('error', 'No output')))
def shell_cmd(
    command: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """Execute a shell command.

    Args:
        command: Shell command to run.
        timeout: Maximum execution time (default 30s, max 60s).

    Returns:
        Dictionary with command output or error.

    Security Note:
        Some dangerous commands are blocked, but this is NOT a
        security sandbox. Do not run untrusted commands.
    """
    timeout = min(timeout, 60)

    # Basic safety check
    cmd_lower = command.lower()
    for blocked in _SHELL_BLACKLIST:
        if blocked in cmd_lower:
            return {"error": f"Command contains blocked pattern: {blocked}"}

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}" if output else result.stderr

        return {
            "output": output.strip() or "(no output)",
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout} seconds"}
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# HTTP Request Tool
# -----------------------------

@registry.register(aliases=["http", "fetch", "curl"])
@tool_output(llm_format=lambda x: f"Status: {x.get('status', 'unknown')}\n{x.get('body', x.get('error', ''))[:500]}")
def http_request(
    url: str,
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET",
    body: str = "",
    headers: str = "",
) -> dict[str, Any]:
    """Make an HTTP request.

    Args:
        url: The URL to request.
        method: HTTP method (GET, POST, PUT, DELETE).
        body: Request body for POST/PUT.
        headers: Optional headers as JSON string.

    Returns:
        Dictionary with response status and body.
    """
    try:
        # Parse headers
        hdrs = {"User-Agent": "LLMTools/1.0"}
        if headers:
            try:
                hdrs.update(json.loads(headers))
            except json.JSONDecodeError:
                return {"error": "Invalid headers JSON"}

        # Build request
        data = body.encode() if body else None
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)

        with urllib.request.urlopen(req, timeout=30) as resp:
            body_content = resp.read().decode("utf-8", errors="replace")

            # Truncate large responses
            if len(body_content) > 10000:
                body_content = body_content[:10000] + "\n... (truncated)"

            return {
                "status": resp.status,
                "headers": dict(resp.headers),
                "body": body_content,
            }
    except urllib.error.HTTPError as e:
        return {
            "status": e.code,
            "error": str(e.reason),
            "body": e.read().decode("utf-8", errors="replace")[:1000],
        }
    except Exception as e:
        return {"error": str(e)}
