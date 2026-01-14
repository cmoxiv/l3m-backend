"""
REPL (Read-Eval-Print Loop) implementation for the chat CLI.
"""

from __future__ import annotations

import atexit
import html
import json
import readline
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from l3m_backend.cli._magic import MAGIC_COMMANDS, MagicCommands
from l3m_backend.cli.commands import command_registry, load_user_commands

if TYPE_CHECKING:
    from l3m_backend.core import ToolRegistry
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


# ANSI escape codes for grey text
GREY = "\033[90m"
WHITE = "\033[97m"
RESET = "\033[0m"


class WaitingAnimation:
    """Animated waiting indicator with spinning bar."""

    SPINNER = ["|", "/", "-", "\\"]

    def __init__(self):
        self._stop = False
        self._thread: threading.Thread | None = None
        self._showing = False

    def start(self):
        """Start the animation in a background thread."""
        self._stop = False
        self._showing = False
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the animation and clear the spinner."""
        self._stop = True
        if self._thread:
            self._thread.join(timeout=0.15)
        # Clear the spinner character
        if self._showing:
            print("\b \b", end="", flush=True)
        self._showing = False

    def _animate(self):
        """Animation loop - cycles through spinner characters."""
        idx = 0
        while not self._stop:
            # Erase previous character if showing
            if self._showing:
                print("\b", end="", flush=True)
            # Print next spinner character
            print(self.SPINNER[idx], end="", flush=True)
            self._showing = True
            idx = (idx + 1) % len(self.SPINNER)
            time.sleep(0.1)


def feedback(msg: str) -> None:
    """Print feedback message in grey to stderr."""
    print(f"{GREY}{msg}{RESET}", file=sys.stderr)


# History file path
HISTORY_FILE = Path.home() / ".l3m" / "prompt_history"

# Default models directory
MODEL_DIR = Path.home() / ".l3m" / "models"

# Load user commands from ~/.l3m/commands/
load_user_commands()

# Commands for completion
COMMANDS = [
    "/clear", "/tools", "/system", "/schema", "/contract", "/history",
    "/undo", "/pop", "/pop1", "/push1", "/context", "/model", "/config", "/debug",
    "/session", "/sessions", "/session-save", "/session-load", "/session-title",
    "/session-tag", "/help", "/quit", "/exit", "/q"
]

# Add user commands to completion
COMMANDS.extend(command_registry.get_completions().keys())


class HistoryCompleter:
    """Completer that suggests from command history and commands."""

    def __init__(self):
        self.matches: list[str] = []

    def complete(self, text: str, state: int) -> str | None:
        """Return the state-th completion for text."""
        if state == 0:
            # Build matches on first call
            self.matches = []

            # Get full line for context
            line = readline.get_line_buffer()

            # Complete /config subcommands
            if line.startswith("/config"):
                self.matches = self._complete_config(line, text)
            # Complete /command subcommands
            elif line.startswith("/command"):
                self.matches = self._complete_command(line, text)
            # Complete / commands
            elif text.startswith("/"):
                self.matches = [cmd for cmd in COMMANDS if cmd.startswith(text)]
            # Complete % magic commands
            elif text.startswith("%"):
                self.matches = [cmd for cmd in MAGIC_COMMANDS if cmd.startswith(text)]
            elif text:
                # Complete from history (case-insensitive prefix match)
                seen = set()
                history_len = readline.get_current_history_length()
                for i in range(history_len, 0, -1):
                    entry = readline.get_history_item(i)
                    if entry and entry not in seen:
                        seen.add(entry)
                        # Skip commands
                        if entry.startswith("/") or entry.startswith("!") or entry.startswith("%"):
                            continue
                        if entry.lower().startswith(text.lower()):
                            self.matches.append(entry)

        if state < len(self.matches):
            return self.matches[state]
        return None

    def _complete_config(self, line: str, text: str) -> list[str]:
        """Complete /config subcommands and keys."""
        parts = line.split()

        # "/config " or "/config s" - complete subcommand
        if len(parts) <= 2:
            subcommands = ["set", "del"]
            return [s for s in subcommands if s.startswith(text)]

        # "/config set " or "/config set c" - complete config keys
        if len(parts) >= 2 and parts[1] in ("set", "del"):
            from l3m_backend.config.config import Config
            keys = list(Config.model_fields.keys())
            return [k for k in keys if k.startswith(text)]

        return []

    def _complete_command(self, line: str, text: str) -> list[str]:
        """Complete /command subcommands and command names."""
        parts = line.split()

        # "/command " or "/command e" - complete subcommand
        if len(parts) <= 2:
            subcommands = ["ls", "edit", "new"]
            return [s for s in subcommands if s.startswith(text)]

        # "/command edit " or "/command edit c" - complete command names
        if len(parts) >= 2 and parts[1] == "edit":
            from l3m_backend.cli.commands.loader import USER_COMMANDS_DIR
            names = []
            if USER_COMMANDS_DIR.exists():
                for cmd_dir in sorted(USER_COMMANDS_DIR.iterdir()):
                    if cmd_dir.is_dir() and not cmd_dir.name.startswith((".", "_")):
                        names.append(cmd_dir.name)
            return [n for n in names if n.startswith(text)]

        return []


def setup_readline():
    """Configure readline for history and completion."""
    # Ensure history directory exists
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Load history file if it exists
    if HISTORY_FILE.exists():
        try:
            readline.read_history_file(HISTORY_FILE)
        except (OSError, IOError):
            pass

    # Set history length
    readline.set_history_length(1000)

    # Save history on exit
    atexit.register(lambda: readline.write_history_file(HISTORY_FILE))

    # Setup completion
    completer = HistoryCompleter()
    readline.set_completer(completer.complete)
    readline.parse_and_bind("tab: complete")

    # Enable history search with up/down after typing
    readline.parse_and_bind('"\\e[A": history-search-backward')
    readline.parse_and_bind('"\\e[B": history-search-forward')


def print_tools(registry: ToolRegistry):
    """Print available tools with their descriptions and aliases.

    Args:
        registry: The ToolRegistry instance to list tools from.
    """
    print("\nAvailable tools:")
    for entry in registry:
        aliases = f" (aliases: {', '.join(entry.aliases)})" if entry.aliases else ""
        print(f"  - {entry.name}{aliases}")
        print(f"    {entry.get_description()}")
    print()


def repl(engine: ChatEngine, session_mgr: Optional["SessionManager"] = None):
    """Run the interactive REPL (Read-Eval-Print Loop).

    Provides an interactive command-line interface for chatting with
    the LLM and managing the conversation.

    Available commands:
        /clear    - Clear conversation history
        /tools    - List available tools
        /system   - Show system prompt
        /schema   - Show tool schema (OpenAI format)
        /contract - Show full system message with tools contract
        /history  - Show conversation history
        /debug    - Toggle debug mode on/off
        /session  - Show current session info
        /sessions - List available sessions
        /quit     - Exit

    Args:
        engine: The ChatEngine instance to use for conversations.
        session_mgr: Optional SessionManager for session persistence.
    """
    # Setup readline for history and completion
    setup_readline()

    # Initialize magic commands
    magic = MagicCommands(engine, session_mgr)

    print("=" * 50)
    print("LlamaCpp Chat REPL with Tool Calling")
    print("=" * 50)
    print("Tab: completion | Ctrl+R: search history")
    print("Ctrl+D twice: exit | /help for commands")
    print("=" * 50)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            # First Ctrl+D - wait for confirmation
            print("\nPress Ctrl+D again to exit, or Ctrl+C to cancel.")
            try:
                # Wait for second Ctrl+D or input
                confirm = input("")
                # If user typed something, cancel exit and process it
                if confirm.strip():
                    user_input = confirm.strip()
                    # Fall through to process user_input below
                else:
                    continue
            except EOFError:
                # Second Ctrl+D - exit
                if session_mgr and session_mgr.session:
                    cwd = Path(session_mgr.session.metadata.working_directory)
                    session_mgr.save(cwd)
                    session_mgr.create_symlink(cwd)
                print("Goodbye!")
                break
            except KeyboardInterrupt:
                # Ctrl+C - cancel exit
                print("\n[Exit cancelled]\n")
                continue  # Go back to show "You: " prompt
        except KeyboardInterrupt:
            # Ctrl+C during prompt - just continue
            print()
            continue

        if not user_input:
            continue

        # Handle shell commands
        if user_input.startswith("!"):
            shell_cmd = user_input[1:].strip()
            if shell_cmd:
                try:
                    result = subprocess.run(
                        shell_cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                    )
                    if result.stdout:
                        print(result.stdout, end="")
                    if result.stderr:
                        print(result.stderr, end="")
                except Exception as e:
                    print(f"Error: {e}")
            continue

        # Handle magic commands
        if user_input.startswith("%"):
            magic.execute(user_input[1:])
            continue

        # Handle commands
        if user_input.startswith("/"):
            # Dispatch to command registry
            entry, args = command_registry.match(user_input)
            if entry:
                try:
                    result = entry.handler(engine, session_mgr, args)
                    if result is True:  # Signal to exit
                        break
                except Exception as e:
                    print(f"Command error: {e}\n")
                continue
            print(f"Unknown command: {user_input}")
            print("Type /help for available commands.\n")
            continue

        # Get response
        try:
            # Add user message to transcript
            if session_mgr:
                session_mgr.add_message("user", user_input, to_history=False)

            # Stream response token-by-token
            print("\nAssistant: ", end="", flush=True)
            response = ""
            spinner = WaitingAnimation()
            spinner.start()
            first_token = True
            try:
                for token in engine.chat(user_input, stream=True):
                    if first_token:
                        spinner.stop()
                        first_token = False
                    # Decode HTML entities (e.g., &#34; -> ")
                    decoded = html.unescape(token)
                    print(decoded, end="", flush=True)
                    response += decoded
            finally:
                spinner.stop()
            print("\n")

            # Check if history was trimmed - generate summary if so
            if session_mgr and engine.history_trimmed:
                engine.history_trimmed = False  # Reset flag
                last_end = session_mgr.get_last_summary_end_idx()
                if len(session_mgr.session.transcript) > last_end:
                    feedback("Context trimmed, generating summary...")
                    session_mgr.generate_summary(engine, last_end)
                    cwd = Path(session_mgr.session.metadata.working_directory)
                    session_mgr.save(cwd)

            # Add assistant response to transcript and sync history
            if session_mgr:
                session_mgr.add_message("assistant", response, to_history=False)
                session_mgr.sync_from_engine(engine.history)

                # Get message count for auto-save and regeneration
                msg_count = len(session_mgr.session.transcript)
                cwd = Path(session_mgr.session.metadata.working_directory)

                # Check if we should regenerate title/tag (exponential intervals: 1, 2, 4, 8...)
                if session_mgr.should_regenerate(msg_count):
                    # Generate title if not set
                    if not session_mgr.session.metadata.title:
                        feedback("Generating session title...")
                        title = session_mgr.generate_title(engine)
                        if title:
                            feedback(f"Title: {title}")
                    # Generate tag if not set
                    if not session_mgr.session.metadata.tag:
                        feedback("Generating session tag...")
                        tag = session_mgr.generate_tag(engine)
                        if tag:
                            feedback(f"Tag: {tag}")
                    # Save after title/tag regeneration
                    session_mgr.save(cwd)
                # Auto-save every 2 messages
                elif msg_count % 2 == 0:
                    session_mgr.save(cwd)
        except Exception as e:
            print(f"\nError: {e}\n")
