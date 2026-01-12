"""MCP logging configuration.

Provides session-specific file logging for MCP exceptions and debug info.
Logs are written to ~/.l3m/logs/<session-id>.log
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Optional

# Directory for log files
LOGS_DIR = Path.home() / ".l3m" / "logs"

# Module-level state
_session_handler: Optional[logging.FileHandler] = None
_session_log_path: Optional[Path] = None


def _ensure_logs_dir() -> None:
    """Ensure the logs directory exists."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_log_path(session_id: str) -> Path:
    """Get the log file path for a session.

    Args:
        session_id: The session ID (or short ID)

    Returns:
        Path to the log file
    """
    _ensure_logs_dir()
    return LOGS_DIR / f"{session_id}.log"


def configure_session_logging(
    session_id: str,
    level: int = logging.DEBUG,
) -> Path:
    """Configure file logging for an MCP session.

    Sets up a file handler that captures all MCP-related logs
    for the given session. Call this when starting a session.

    Args:
        session_id: The session ID (uses first 8 chars)
        level: Logging level for file output (default DEBUG)

    Returns:
        Path to the log file
    """
    global _session_handler, _session_log_path

    # Use short session ID for filename
    short_id = session_id[:8] if len(session_id) > 8 else session_id
    log_path = get_log_path(short_id)

    # Remove existing session handler if any
    close_session_logging()

    # Create file handler
    _session_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    _session_handler.setLevel(level)

    # Format with timestamp and full context
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _session_handler.setFormatter(formatter)

    # Add handler to MCP loggers
    mcp_logger = logging.getLogger("l3m_backend.mcp")
    mcp_logger.addHandler(_session_handler)
    mcp_logger.setLevel(min(mcp_logger.level or logging.DEBUG, level))

    _session_log_path = log_path

    # Log session start
    mcp_logger.info(f"=== Session started: {session_id} ===")

    return log_path


def close_session_logging() -> None:
    """Close the current session's file logging.

    Call this when ending a session to flush and close the log file.
    """
    global _session_handler, _session_log_path

    if _session_handler is not None:
        # Log session end
        mcp_logger = logging.getLogger("l3m_backend.mcp")
        mcp_logger.info("=== Session ended ===")

        # Remove and close handler
        mcp_logger.removeHandler(_session_handler)
        _session_handler.close()
        _session_handler = None
        _session_log_path = None


def get_current_log_path() -> Optional[Path]:
    """Get the current session's log file path.

    Returns:
        Path to current log file, or None if no session logging active
    """
    return _session_log_path


def log_mcp_exception(
    error: Exception,
    context: str = "",
    include_traceback: bool = True,
) -> str:
    """Log an MCP exception with full details to the session log.

    This logs the full traceback to the file while returning a
    clean, user-friendly message for display.

    Args:
        error: The exception to log
        context: Additional context about what was happening
        include_traceback: Whether to include full traceback in log

    Returns:
        User-friendly error message (without traceback)
    """
    logger = logging.getLogger("l3m_backend.mcp")

    # Build the log message
    error_type = type(error).__name__
    error_msg = str(error)

    # User-friendly message (no traceback)
    if context:
        user_msg = f"{context}: {error_msg}"
    else:
        user_msg = f"{error_type}: {error_msg}"

    # Full log message with traceback
    if include_traceback:
        tb_str = traceback.format_exc()
        log_msg = f"{context}\n{error_type}: {error_msg}\n\nTraceback:\n{tb_str}"
    else:
        log_msg = f"{context} - {error_type}: {error_msg}"

    # Log to file (full details)
    logger.error(log_msg)

    return user_msg


def log_mcp_warning(message: str, context: str = "") -> None:
    """Log an MCP warning to the session log.

    Args:
        message: Warning message
        context: Additional context
    """
    logger = logging.getLogger("l3m_backend.mcp")
    if context:
        logger.warning(f"{context}: {message}")
    else:
        logger.warning(message)


def log_mcp_info(message: str) -> None:
    """Log an MCP info message to the session log.

    Args:
        message: Info message
    """
    logger = logging.getLogger("l3m_backend.mcp")
    logger.info(message)


def log_mcp_debug(message: str) -> None:
    """Log an MCP debug message to the session log.

    Args:
        message: Debug message
    """
    logger = logging.getLogger("l3m_backend.mcp")
    logger.debug(message)
