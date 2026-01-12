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

            # Complete / commands
            if text.startswith("/"):
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
            cmd = user_input.lower()
            if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                # Generate summary before exit
                if session_mgr and session_mgr.session and session_mgr.session.transcript:
                    last_end = session_mgr.get_last_summary_end_idx()
                    if len(session_mgr.session.transcript) > last_end:
                        feedback("Generating session summary...")
                        session_mgr.generate_summary(engine, last_end)

                # Save and update symlink
                if session_mgr and session_mgr.session:
                    cwd = Path(session_mgr.session.metadata.working_directory)
                    session_mgr.save(cwd)
                    session_mgr.create_symlink(cwd)

                print("Goodbye!")
                break
            elif cmd == "/clear":
                engine.clear()
                print("Conversation cleared.\n")
                continue
            elif cmd == "/tools":
                print_tools(engine.registry)
                continue
            elif cmd == "/system":
                print(f"\n{engine.system_prompt}\n")
                continue
            elif cmd == "/schema":
                print(f"\n{json.dumps(engine.tools, indent=2)}\n")
                continue
            elif cmd == "/contract":
                print(f"\n{engine._build_system_message()['content']}\n")
                continue
            elif cmd == "/history":
                print()
                for i, msg in enumerate(engine.messages):
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    if role == "system":
                        print(f"[{i}] system: (system prompt)")
                    elif role == "tool":
                        print(f"[{i}] tool [{msg.get('tool_call_id', '?')}]: {content}")
                    elif msg.get("tool_calls"):
                        calls = ", ".join(tc["function"]["name"] for tc in msg["tool_calls"])
                        print(f"[{i}] {role}: [calls: {calls}]")
                    else:
                        preview = (content[:60] + "...") if content and len(content) > 60 else content
                        print(f"[{i}] {role}: {preview}")
                print()
                continue
            elif cmd == "/help":
                print("\nCommands:")
                print("  /clear         - Clear conversation history")
                print("  /tools         - List available tools")
                print("  /system        - Show system prompt")
                print("  /schema        - Show tool schema (OpenAI format)")
                print("  /contract      - Show full system message with tools contract")
                print("  /history       - Show conversation history")
                print("  /undo          - Remove last user+assistant exchange")
                print("  /pop [n]       - Remove last n message pairs")
                print("  /pop1          - Remove last single message")
                print("  /push1         - Push a single message")
                print("  /context       - Show context usage estimate")
                print("  /model         - Show model info")
                print("  /config        - Show current configuration")
                print("  /debug         - Toggle debug mode on/off")
                print("  /session       - Show current session info")
                print("  /sessions      - List available sessions")
                print("  /session-save  - Save session (optional: title)")
                print("  /session-load  - Load session as context")
                print("  /session-title - Generate/set session title")
                print("  /session-tag   - Add tag to session")
                print("  /help          - Show this help message")
                print("  /quit          - Exit (also /exit, /q)")
                print("  !<cmd>         - Run shell command")
                print("  %<cmd>         - Magic commands (type % for help)")
                print()
                continue
            elif cmd == "/undo":
                if len(engine.history) >= 2:
                    engine.history.pop()  # assistant
                    engine.history.pop()  # user
                    if session_mgr:
                        session_mgr.sync_from_engine(engine.history)
                    print("Removed last exchange.\n")
                elif engine.history:
                    engine.history.pop()
                    if session_mgr:
                        session_mgr.sync_from_engine(engine.history)
                    print("Removed last message.\n")
                else:
                    print("History is empty.\n")
                continue
            elif cmd == "/pop1":
                if engine.history:
                    msg = engine.history.pop()
                    role = msg.get("role", "?")
                    if session_mgr:
                        session_mgr.sync_from_engine(engine.history)
                    print(f"Removed 1 {role} message.\n")
                else:
                    print("History is empty.\n")
                continue
            elif cmd.startswith("/push1"):
                # Parse: /push1 <content>
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: /push1 <content>\n")
                    continue
                content = parts[1]
                # Infer role from last message
                if engine.history:
                    last_role = engine.history[-1].get("role", "user")
                    role = "assistant" if last_role == "user" else "user"
                else:
                    role = "user"
                engine.history.append({"role": role, "content": content})
                if session_mgr:
                    session_mgr.sync_from_engine(engine.history)
                print(f"Pushed 1 {role} message.\n")
                continue
            elif cmd.startswith("/pop"):
                # Parse optional argument: /pop or /pop 3
                parts = user_input.split()
                n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                removed_pairs = 0
                for _ in range(n):
                    if len(engine.history) >= 2:
                        engine.history.pop()  # assistant
                        engine.history.pop()  # user
                        removed_pairs += 1
                    else:
                        break  # Can't remove more pairs
                if removed_pairs > 0:
                    if session_mgr:
                        session_mgr.sync_from_engine(engine.history)
                    print(f"Removed {removed_pairs} message pair(s).\n")
                else:
                    print("History is empty or incomplete pair.\n")
                continue
            elif cmd == "/context":
                # Use accurate token counting if available
                messages = engine._build_messages()
                try:
                    total_tokens = engine._count_tokens(messages)
                    n_ctx = engine.llm.n_ctx()
                    if not isinstance(n_ctx, int):
                        n_ctx = 32768
                except (AttributeError, TypeError):
                    # Fallback: estimate 4 chars per token
                    total_chars = sum(len(m.get("content", "")) for m in messages)
                    total_tokens = total_chars // 4
                    n_ctx = 32768
                pct = (total_tokens / n_ctx) * 100
                print(f"\nContext: {total_tokens:,} / {n_ctx:,} tokens ({pct:.1f}%)")
                print(f"History: {len(engine.history)} messages\n")
                continue
            elif cmd.startswith("/model"):
                from l3m_backend.utils.gpu import get_gpu_info

                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    # No argument - show current model info
                    llm = engine.llm
                    model_path = getattr(llm, "model_path", "unknown")
                    try:
                        n_ctx = llm.n_ctx()
                    except (AttributeError, TypeError):
                        n_ctx = "?"
                    gpu_info = get_gpu_info(engine)
                    layers = gpu_info["gpu_layers"]
                    layers_str = "all" if layers == -1 else str(layers)

                    print(f"\nModel: {model_path}")
                    print(f"Context: {n_ctx}")
                    print(f"Backend: {gpu_info['backend']}")
                    print(f"GPU layers: {layers_str}")
                    print(f"\nAvailable models in {MODEL_DIR}:")
                    if MODEL_DIR.exists():
                        for m in sorted(MODEL_DIR.glob("*.gguf")):
                            size_mb = m.stat().st_size / (1024 * 1024)
                            print(f"  {m.name:<50} {size_mb:>8.1f} MB")
                    print("\nUsage: /model <name> to switch models\n")
                else:
                    # Switch to specified model
                    model_name = parts[1].strip()
                    new_model_path = MODEL_DIR / model_name
                    if not new_model_path.exists():
                        # Try as full path
                        new_model_path = Path(model_name)
                        if not new_model_path.exists():
                            print(f"Model not found: {model_name}")
                            print(f"Looked in: {MODEL_DIR / model_name}")
                            continue

                    print(f"Switching to {new_model_path.name}...")
                    try:
                        engine.switch_model(str(new_model_path))
                        gpu_info = get_gpu_info(engine)
                        print(f"Switched to: {new_model_path.name}")
                        print(f"Context: {engine.llm.n_ctx()}")
                        print(f"Backend: {gpu_info['backend']}\n")
                    except Exception as e:
                        print(f"Failed to switch model: {e}\n")
                continue
            elif cmd == "/config":
                from l3m_backend.config import get_config_manager
                cfg_mgr = get_config_manager()
                settings = cfg_mgr.list_settings()
                print(f"\nConfig file: {cfg_mgr.CONFIG_FILE}")
                if settings:
                    for key, value in settings.items():
                        print(f"  {key}: {value}")
                else:
                    print("  (no custom settings)")
                print()
                continue
            elif cmd == "/debug":
                engine.debug = not engine.debug
                status = "enabled" if engine.debug else "disabled"
                print(f"Debug mode {status}.\n")
                continue
            elif cmd == "/session":
                if session_mgr and session_mgr.session:
                    s = session_mgr.session
                    print(f"\nSession ID: {s.metadata.id}")
                    print(f"Title: {s.metadata.title or '(untitled)'}")
                    print(f"Tag: {s.metadata.tag or '(none)'}")
                    print(f"Created: {s.metadata.created_at[:19]}")
                    print(f"Updated: {s.metadata.updated_at[:19]}")
                    print(f"Initial: {s.metadata.initial_datetime}")
                    print(f"Last save: {s.metadata.last_save_datetime or '(not saved)'}")
                    print(f"Working dir: {s.metadata.working_directory}")
                    print(f"Incognito: {s.metadata.is_incognito}")
                    print(f"History: {len(s.history)} messages")
                    print(f"Transcript: {len(s.transcript)} messages")
                    print(f"Summaries: {len(s.metadata.summaries)}\n")
                else:
                    print("No active session.\n")
                continue
            elif cmd.startswith("/sessions"):
                if session_mgr:
                    parts = user_input.split(maxsplit=1)
                    query = parts[1] if len(parts) > 1 else None
                    if query:
                        sessions = session_mgr.search(query)
                        if not sessions:
                            print(f"No sessions matching '{query}'\n")
                            continue
                        print(f"\nSessions matching '{query}':")
                    else:
                        sessions = session_mgr.list_sessions()
                        if not sessions:
                            print("No sessions found.\n")
                            continue
                        print("\nAvailable sessions:")
                    for s in sessions[:10]:
                        title = s.title or "(untitled)"
                        print(f"  {s.id[:8]}  {title[:40]}")
                    if len(sessions) > 10:
                        print(f"  ... and {len(sessions) - 10} more")
                    print()
                else:
                    print("Session management not available.\n")
                continue
            elif cmd.startswith("/session-save"):
                if not session_mgr or not session_mgr.session:
                    print("No active session.\n")
                    continue
                s = session_mgr.session
                # Parse optional title argument
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    s.metadata.title = parts[1]
                    feedback(f"Set session title: {s.metadata.title}")
                elif not s.metadata.title and s.transcript:
                    feedback("Generating title...")
                    title = session_mgr.generate_title(engine)
                    if title:
                        feedback(f"Generated title: {title}")
                    else:
                        feedback("Could not generate title.")
                path = session_mgr.save()
                feedback(f"Session saved: {path}\n")
                continue
            elif cmd.startswith("/session-load"):
                if not session_mgr:
                    print("Session management not available.\n")
                    continue
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: /session-load <session_id>")
                    print("Use /sessions to list available sessions.\n")
                    continue
                session_id = parts[1].strip()
                print(f"Loading session {session_id[:8]}...")
                loaded = session_mgr.load_as_context([session_id], engine)
                if loaded:
                    print(f"Loaded: {', '.join(loaded)}\n")
                else:
                    print(f"Could not load session: {session_id}\n")
                continue
            elif cmd.startswith("/session-title"):
                if not session_mgr or not session_mgr.session:
                    print("No active session.\n")
                    continue
                s = session_mgr.session
                # Parse optional title argument
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    s.metadata.title = parts[1]
                    feedback(f"Title set: {s.metadata.title}")
                    session_mgr.save()
                elif s.transcript:
                    feedback("Generating title...")
                    title = session_mgr.generate_title(engine)
                    if title:
                        feedback(f"Generated title: {title}")
                        session_mgr.save()
                    else:
                        feedback("Could not generate title.")
                else:
                    print("No conversation yet to generate title from.")
                print()
                continue
            elif cmd.startswith("/session-tag"):
                if not session_mgr or not session_mgr.session:
                    print("No active session.\n")
                    continue
                s = session_mgr.session
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print(f"Usage: /session-tag <tag>")
                    print(f"Current tag: {s.metadata.tag or '(none)'}\n")
                    continue
                tag = parts[1].strip().lower()
                old_tag = s.metadata.tag
                s.metadata.tag = tag
                cwd = Path(s.metadata.working_directory)
                session_mgr.save(cwd)
                if old_tag:
                    feedback(f"Changed tag: {old_tag} -> {tag}\n")
                else:
                    feedback(f"Set tag: {tag}\n")
                continue
            else:
                # Try user-defined commands from ~/.l3m/commands/
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
