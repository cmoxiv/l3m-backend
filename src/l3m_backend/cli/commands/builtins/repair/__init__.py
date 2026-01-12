"""Repair command - fix history alternation issues."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register(
    "repair",
    "Fix history alternation",
    usage="/repair [--dry-run]",
    has_args=True,
)
def cmd_repair(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Repair message history to ensure proper user/assistant alternation.

    LLMs expect messages to alternate between user and assistant.
    This command inserts placeholder messages where alternation is broken.

    Usage:
        /repair           - Fix history and save
        /repair --dry-run - Show what would be fixed without changing anything
    """
    dry_run = "--dry-run" in args

    if not engine.history:
        print("\nNo history to repair.\n")
        return

    # Analyze history
    issues = []
    repaired = []
    prev_role = None

    for i, msg in enumerate(engine.history):
        role = msg.get("role", "unknown")

        # Skip system messages
        if role == "system":
            repaired.append(msg)
            continue

        # Check for alternation issues
        if role == "assistant" and prev_role != "user":
            issues.append(f"[{i}] assistant without preceding user")
            # Insert placeholder user message
            repaired.append({
                "role": "user",
                "content": "(continued from previous context)",
            })
            repaired.append(msg)
        elif role == "user" and prev_role == "user":
            issues.append(f"[{i}] consecutive user messages")
            # Merge with previous or just add
            repaired.append(msg)
        else:
            repaired.append(msg)

        if role in ("user", "assistant"):
            prev_role = role

    if not issues:
        print("\nHistory alternation is correct. No repairs needed.\n")
        return

    print(f"\nFound {len(issues)} alternation issue(s):")
    for issue in issues:
        print(f"  - {issue}")

    if dry_run:
        print(f"\nDry run: would insert {len(repaired) - len(engine.history)} placeholder message(s)")
        print("Run '/repair' without --dry-run to apply fixes.\n")
        return

    # Apply repair
    engine.history = repaired

    # Sync to session if available
    if session_mgr and session_mgr.session:
        session_mgr.session.sync_history_from_engine(repaired)
        session_mgr.save()
        print(f"\nRepaired history and saved session.")
    else:
        print(f"\nRepaired engine history (session not saved).")

    print(f"Inserted {len(repaired) - len(engine.history) + len(issues)} placeholder message(s).\n")
