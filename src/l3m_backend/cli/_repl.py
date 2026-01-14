"""
Feature-rich REPL (Read-Eval-Print Loop) implementation using prompt_toolkit.

Provides command history, auto-completion, multi-line input, and better UX.
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from l3m_backend.cli._magic import MAGIC_COMMANDS, MagicCommands
from l3m_backend.cli.commands import command_registry, load_user_commands

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, merge_completers
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

if TYPE_CHECKING:
    from l3m_backend.core import ToolRegistry
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


# ANSI escape codes for colored text
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


# Default models directory
MODEL_DIR = Path.home() / ".l3m" / "models"

# Load user commands from ~/.l3m/commands/
load_user_commands()

# Get all commands from registry for completion
COMMANDS = command_registry.get_completions()


class CommandCompleter(Completer):
    """Completer for REPL commands and magic commands."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Complete / commands
        if text.startswith("/"):
            for cmd, description in COMMANDS.items():
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=description,
                    )
            # Also complete /exit and /q as aliases for /quit
            if "/exit".startswith(text):
                yield Completion("/exit", start_position=-len(text), display_meta="Exit")
            if "/q".startswith(text):
                yield Completion("/q", start_position=-len(text), display_meta="Exit")

        # Complete % magic commands
        elif text.startswith("%"):
            for cmd, description in MAGIC_COMMANDS.items():
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=description,
                    )


class HistoryCompleter(Completer):
    """Completer that suggests from command history."""

    def __init__(self, history: FileHistory):
        self.history = history

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.strip()

        # Don't complete commands or empty text
        if not text or text.startswith("/") or text.startswith("!") or text.startswith("%"):
            return

        # Get unique history entries that start with the typed text
        seen = set()
        for entry in reversed(list(self.history.get_strings())):
            entry = entry.strip()
            # Skip commands and duplicates
            if entry.startswith("/") or entry.startswith("!") or entry.startswith("%") or entry in seen:
                continue
            seen.add(entry)

            if entry.lower().startswith(text.lower()):
                yield Completion(
                    entry,
                    start_position=-len(text),
                    display_meta="history",
                )


class ModelCompleter(Completer):
    """Completer for model names from ~/.l3m/models/"""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only complete after "/model "
        if text.startswith("/model "):
            prefix = text[7:].lower()  # After "/model "
            if MODEL_DIR.exists():
                for model_path in MODEL_DIR.glob("*.gguf"):
                    if model_path.name.lower().startswith(prefix):
                        size_mb = model_path.stat().st_size / (1024 * 1024)
                        yield Completion(
                            model_path.name,
                            start_position=-len(prefix),
                            display_meta=f"{size_mb:.0f}MB",
                        )


class ResourceCompleter(Completer):
    """Completer for MCP resource URIs with @ prefix."""

    def __init__(self, engine: "ChatEngine"):
        self.engine = engine

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only complete @ prefix
        if not text.startswith("@"):
            return

        uri_prefix = text[1:]  # After @

        for uri, server in self._get_resources():
            if uri.lower().startswith(uri_prefix.lower()):
                yield Completion(
                    "@" + uri,
                    start_position=-len(text),
                    display_meta=server,
                )

    def _get_resources(self) -> list[tuple[str, str]]:
        """Get list of (uri, server_name) tuples."""
        resources = []
        mcp_client = getattr(self.engine, "_mcp_client", None)
        if not mcp_client:
            return resources

        for server_name in mcp_client.connected_servers:
            conn = mcp_client.get_connection(server_name)
            if not conn:
                continue

            # Static resources
            for resource in conn.resources:
                uri = str(resource.uri) if hasattr(resource, "uri") else str(resource)
                resources.append((uri, server_name))

            # Resource templates
            if hasattr(conn, "resource_templates"):
                for template in conn.resource_templates:
                    uri = str(template.uriTemplate) if hasattr(template, "uriTemplate") else ""
                    if uri:
                        resources.append((uri, server_name))

        return resources


class PromptCompleter(Completer):
    """Completer for MCP prompts with # prefix."""

    def __init__(self, engine: "ChatEngine"):
        self.engine = engine

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only complete # prefix
        if not text.startswith("#"):
            return

        after_hash = text[1:]  # After #

        # Check if we're completing arguments (has a space after prompt name)
        if " " in after_hash:
            # Completing arguments
            prompt_name = after_hash.split()[0]
            yield from self._get_argument_completions(text, prompt_name)
        else:
            # Completing prompt name
            for name, desc, server in self._get_prompts():
                if name.lower().startswith(after_hash.lower()):
                    yield Completion(
                        "#" + name,
                        start_position=-len(text),
                        display_meta=f"{server}: {desc}" if desc else server,
                    )

    def _get_argument_completions(self, text: str, prompt_name: str):
        """Get completions for prompt arguments."""
        mcp_client = getattr(self.engine, "_mcp_client", None)
        if not mcp_client:
            return

        # Find the prompt definition
        for server_name in mcp_client.connected_servers:
            conn = mcp_client.get_connection(server_name)
            if not conn:
                continue

            for prompt in conn.prompts:
                if prompt.name == prompt_name:
                    if hasattr(prompt, "arguments") and prompt.arguments:
                        # Get arguments already in the input
                        existing_args = set()
                        parts = text.split()
                        for part in parts[1:]:  # Skip #prompt_name
                            if "=" in part:
                                existing_args.add(part.split("=")[0])

                        # Suggest missing arguments
                        for arg in prompt.arguments:
                            if arg.name not in existing_args:
                                required = "" if arg.required else "?"
                                desc = arg.description if hasattr(arg, "description") and arg.description else ""
                                yield Completion(
                                    f"{arg.name}=",
                                    start_position=0,
                                    display=f"{arg.name}{required}=",
                                    display_meta=desc[:30] if desc else None,
                                )
                    return

    def _get_prompts(self) -> list[tuple[str, str, str]]:
        """Get list of (name, description, server_name) tuples."""
        prompts = []
        mcp_client = getattr(self.engine, "_mcp_client", None)
        if not mcp_client:
            return prompts

        for server_name in mcp_client.connected_servers:
            conn = mcp_client.get_connection(server_name)
            if not conn:
                continue

            for prompt in conn.prompts:
                desc = prompt.description if hasattr(prompt, "description") else ""
                # Truncate description to first line
                if desc:
                    desc = desc.split("\n")[0][:40]
                prompts.append((prompt.name, desc, server_name))

        return prompts


class ConfigCompleter(Completer):
    """Completer for /config set and /config del commands."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only complete /config commands
        if not text.startswith("/config"):
            return

        # Just "/config" without space - let CommandCompleter handle it
        if text == "/config":
            return

        parts = text.split()

        # "/config " - suggest subcommands
        if len(parts) == 1 and text.endswith(" "):
            for sub in ["set", "del"]:
                yield Completion(
                    sub,
                    start_position=0,
                    display_meta="Set config value" if sub == "set" else "Delete config value",
                )
            return

        # "/config s" - complete partial subcommand
        if len(parts) == 2 and not text.endswith(" "):
            subcommand = parts[1]
            for sub in ["set", "del"]:
                if sub.startswith(subcommand):
                    yield Completion(
                        sub,
                        start_position=-len(subcommand),
                        display_meta="Set config value" if sub == "set" else "Delete config value",
                    )
            return

        subcommand = parts[1] if len(parts) >= 2 else ""

        # "/config set " or "/config del " - suggest config keys
        if subcommand in ("set", "del") and len(parts) == 2 and text.endswith(" "):
            from l3m_backend.config.config import Config
            for field_name, field_info in Config.model_fields.items():
                desc = field_info.description or ""
                if len(desc) > 40:
                    desc = desc[:37] + "..."
                yield Completion(
                    field_name,
                    start_position=0,
                    display_meta=desc,
                )
            return

        # "/config set c" - complete partial key
        if subcommand in ("set", "del") and len(parts) == 3 and not text.endswith(" "):
            from l3m_backend.config.config import Config
            partial_key = parts[2]
            for field_name, field_info in Config.model_fields.items():
                if field_name.startswith(partial_key):
                    desc = field_info.description or ""
                    if len(desc) > 40:
                        desc = desc[:37] + "..."
                    yield Completion(
                        field_name,
                        start_position=-len(partial_key),
                        display_meta=desc,
                    )


class CommandNameCompleter(Completer):
    """Completer for /command edit and /command new commands."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only complete /command commands
        if not text.startswith("/command"):
            return

        # Just "/command" without space - let CommandCompleter handle it
        if text == "/command":
            return

        parts = text.split()

        # "/command " - suggest subcommands
        if len(parts) == 1 and text.endswith(" "):
            for sub, desc in [("ls", "List commands"), ("edit", "Edit command"), ("new", "Create command")]:
                yield Completion(sub, start_position=0, display_meta=desc)
            return

        # "/command e" - complete partial subcommand
        if len(parts) == 2 and not text.endswith(" "):
            subcommand = parts[1]
            for sub, desc in [("ls", "List commands"), ("edit", "Edit command"), ("new", "Create command")]:
                if sub.startswith(subcommand):
                    yield Completion(sub, start_position=-len(subcommand), display_meta=desc)
            return

        subcommand = parts[1] if len(parts) >= 2 else ""

        # "/command edit " - suggest command names
        if subcommand == "edit" and len(parts) == 2 and text.endswith(" "):
            from l3m_backend.cli.commands.loader import USER_COMMANDS_DIR
            if USER_COMMANDS_DIR.exists():
                for cmd_dir in sorted(USER_COMMANDS_DIR.iterdir()):
                    if cmd_dir.is_dir() and not cmd_dir.name.startswith((".", "_")):
                        is_symlink = cmd_dir.is_symlink()
                        yield Completion(
                            cmd_dir.name,
                            start_position=0,
                            display_meta="builtin" if is_symlink else "custom",
                        )
            return

        # "/command edit c" - complete partial command name
        if subcommand == "edit" and len(parts) == 3 and not text.endswith(" "):
            from l3m_backend.cli.commands.loader import USER_COMMANDS_DIR
            partial_name = parts[2]
            if USER_COMMANDS_DIR.exists():
                for cmd_dir in sorted(USER_COMMANDS_DIR.iterdir()):
                    if cmd_dir.is_dir() and not cmd_dir.name.startswith((".", "_")):
                        if cmd_dir.name.startswith(partial_name):
                            is_symlink = cmd_dir.is_symlink()
                            yield Completion(
                                cmd_dir.name,
                                start_position=-len(partial_name),
                                display_meta="builtin" if is_symlink else "custom",
                            )


def _expand_mcp_prompt_inline(engine: "ChatEngine", prompt_name: str) -> str | None:
    """Expand an MCP prompt by name for inline expansion (space key).

    This version is used when the user selects a prompt and presses space.

    For prompts WITH required arguments:
        Returns "#prompt_name {arg1} {arg2}" format so user can fill in args
        and press Enter to trigger full expansion.

    For prompts WITHOUT required arguments:
        Returns the fully expanded prompt text directly.

    Args:
        engine: ChatEngine instance
        prompt_name: Name of the prompt (without #)

    Returns:
        Expanded prompt text or template with placeholders, or None if not found
    """
    mcp_client = getattr(engine, "_mcp_client", None)
    if not mcp_client or not mcp_client.connected_servers:
        return None

    # Find the prompt
    for server_name in mcp_client.connected_servers:
        conn = mcp_client.get_connection(server_name)
        if not conn:
            continue

        for prompt in conn.prompts:
            if prompt.name == prompt_name:
                # Check if prompt has arguments
                has_args = hasattr(prompt, "arguments") and prompt.arguments

                if has_args:
                    # Has arguments - return template with placeholders
                    # User will fill these in and press Enter to expand
                    arg_placeholders = " ".join(
                        f'{a.name}="{{{a.name}}}"' for a in prompt.arguments
                    )
                    return f"#{prompt_name} {arg_placeholders}"

                # No arguments - expand immediately
                try:
                    from l3m_backend.mcp.client.client import run_async
                    result = run_async(conn.session.get_prompt(prompt_name, arguments={}))

                    expanded_parts = []
                    for msg in result.messages:
                        content = msg.content
                        if hasattr(content, "text"):
                            content = content.text
                        expanded_parts.append(str(content))

                    return "\n".join(expanded_parts)
                except Exception:
                    return None

    return None


def _expand_mcp_prompt(engine: "ChatEngine", prompt_input: str) -> str | None:
    """Expand an MCP prompt and return the expanded text.

    Args:
        engine: ChatEngine instance
        prompt_input: Input after # (e.g., "summarize text=\"hello\"")

    Returns:
        Expanded prompt text, or None if expansion failed
    """
    import shlex

    prompt_input = prompt_input.strip()
    if not prompt_input:
        print("Usage: #<prompt_name> [arg=value ...]")
        print("Example: #summarize text=\"Hello world\"")
        return None

    mcp_client = getattr(engine, "_mcp_client", None)
    if not mcp_client:
        print("No MCP client initialized.")
        return None

    if not mcp_client.connected_servers:
        print("No MCP servers connected.")
        return None

    # Parse: first word is prompt name, rest are key=value pairs
    try:
        parts = shlex.split(prompt_input)
    except ValueError:
        parts = prompt_input.split()

    prompt_name = parts[0]
    prompt_args = {}

    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            prompt_args[key] = value

    try:
        from l3m_backend.mcp.client.client import run_async

        # Find and execute the prompt
        for server_name in mcp_client.connected_servers:
            conn = mcp_client.get_connection(server_name)
            if not conn:
                continue

            for prompt in conn.prompts:
                if prompt.name == prompt_name:
                    result = run_async(conn.session.get_prompt(prompt_name, arguments=prompt_args))

                    # Extract the expanded prompt text
                    expanded_parts = []
                    for msg in result.messages:
                        content = msg.content
                        if hasattr(content, "text"):
                            content = content.text
                        expanded_parts.append(str(content))

                    expanded = "\n".join(expanded_parts)

                    # Show expanded prompt
                    print(f"{GREY}[#{prompt_name} expanded]{RESET}")
                    print(f"{GREY}{expanded}{RESET}")
                    print()

                    return expanded

        print(f"Prompt not found: {prompt_name}")
        print("Use /mcp prompts to list available prompts.")
        return None

    except Exception as e:
        # Clean error message (no traceback)
        error_msg = str(e).split('\n')[0]
        print(f"Error: {error_msg}")
        return None


def _fetch_mcp_resource(engine: "ChatEngine", uri: str) -> None:
    """Fetch an MCP resource and add to chat history."""
    uri = uri.strip()
    if not uri:
        print("Usage: @<uri>")
        print("Example: @test://info")
        return

    mcp_client = getattr(engine, "_mcp_client", None)
    if not mcp_client:
        print("No MCP client initialized.")
        return

    if not mcp_client.connected_servers:
        print("No MCP servers connected.")
        return

    try:
        from l3m_backend.mcp.client.client import run_async

        # Try each connected server
        for server_name in mcp_client.connected_servers:
            conn = mcp_client.get_connection(server_name)
            if not conn:
                continue

            # Check static resources
            for resource in conn.resources:
                resource_uri = str(resource.uri) if hasattr(resource, "uri") else str(resource)
                if resource_uri == uri:
                    result = run_async(conn.session.read_resource(uri))
                    content = _extract_resource_content(result)
                    _add_resource_to_history(engine, uri, content)
                    return

            # Check resource templates
            if hasattr(conn, "resource_templates"):
                for template in conn.resource_templates:
                    template_uri = str(template.uriTemplate) if hasattr(template, "uriTemplate") else ""
                    # Simple check: if the URI scheme matches
                    if template_uri and uri.split("/")[0] == template_uri.split("/")[0]:
                        try:
                            result = run_async(conn.session.read_resource(uri))
                            content = _extract_resource_content(result)
                            _add_resource_to_history(engine, uri, content)
                            return
                        except Exception:
                            continue

        print(f"Resource not found: {uri}")
        print("Use /mcp resources to list available resources.")

    except Exception as e:
        # Clean error message (no traceback)
        error_msg = str(e).split('\n')[0]
        print(f"Error: {error_msg}")


def _extract_resource_content(result) -> str:
    """Extract text content from MCP resource result."""
    contents = []
    for content in result.contents:
        if hasattr(content, "text"):
            contents.append(content.text)
        else:
            contents.append(str(content))
    return "\n".join(contents)


def _add_resource_to_history(engine: "ChatEngine", uri: str, content: str) -> None:
    """Add fetched resource to chat history and display it.

    Assistant responses are stored as plain text to match the contract
    which says final responses should be plain text (enables streaming).
    """
    user_msg = f"@{uri}"

    engine.history.append({"role": "user", "content": user_msg})
    engine.history.append({"role": "assistant", "content": content})

    # Display the resource content
    print(f"{GREY}[{uri}]{RESET}")
    print(content)
    print()


def get_style() -> Style:
    """Get the prompt style."""
    return Style.from_dict({
        "prompt": "ansicyan bold",
        "command": "ansigreen",
    })


def print_tools(registry: ToolRegistry):
    """Print available tools with their descriptions and aliases."""
    print("\nAvailable tools:")
    for entry in registry:
        aliases = f" (aliases: {', '.join(entry.aliases)})" if entry.aliases else ""
        print(f"  - {entry.name}{aliases}")
        print(f"    {entry.get_description()}")
    print()


def print_help():
    """Print help message with available commands."""
    print("\nCommands:")
    for cmd, description in COMMANDS.items():
        print(f"  {cmd:<12} - {description}")
    print(f"  {'!<cmd>':<12} - Run shell command")
    print(f"  {'%<cmd>':<12} - Magic commands (type %help)")
    print()


def repl(engine: ChatEngine, session_mgr: Optional["SessionManager"] = None):
    """Run the interactive REPL (Read-Eval-Print Loop).

    Provides an interactive command-line interface for chatting with
    the LLM and managing the conversation.

    Features:
        - Command history (persistent across sessions)
        - Tab completion for /commands
        - Ctrl+C to cancel input, Ctrl+D to exit
        - Multi-line input support (paste or type)
        - Session persistence (if session_mgr provided)

    Args:
        engine: The ChatEngine instance to use for conversations.
        session_mgr: Optional SessionManager for session persistence.
    """
    # Setup history file
    history_file = Path.home() / ".l3m" / "prompt_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history = FileHistory(str(history_file))

    # Create key bindings
    bindings = KeyBindings()

    @bindings.add("c-c")
    def _(event):
        """Handle Ctrl+C - cancel current input."""
        event.app.current_buffer.reset()
        print()
        print("You: ", end="", flush=True)

    @bindings.add(" ")
    def handle_space_for_prompt_expansion(event):
        """Handle space - expand MCP prompt or !! shortcut."""
        import re
        buf = event.app.current_buffer
        text = buf.text

        # Check for !! - expand with last response
        if "!!" in text:
            # Find the last assistant response
            last_response = None
            for msg in reversed(engine.history):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # Try to parse as JSON (tool response format)
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "content" in parsed:
                            last_response = parsed["content"]
                        else:
                            last_response = content
                    except (json.JSONDecodeError, TypeError):
                        last_response = content
                    break

            if last_response:
                # Replace !! with last response
                cursor_offset = text.find("!!")
                new_text = text.replace("!!", last_response, 1)
                buf.text = new_text
                # Position cursor after the expanded text
                buf.cursor_position = cursor_offset + len(last_response)
                return

        text = text.strip()

        # Check if this looks like "#prompt_name" (no space yet, no args)
        if text.startswith("#") and " " not in text and len(text) > 1:
            prompt_name = text[1:]

            # Try to expand the prompt
            expanded = _expand_mcp_prompt_inline(engine, prompt_name)
            if expanded:
                # Replace buffer with expanded text
                buf.text = expanded

                # Find placeholder like {text}, position cursor at end, and delete it
                placeholder_match = re.search(r'\{(\w+)\}', expanded)
                if placeholder_match:
                    buf.cursor_position = placeholder_match.end()
                    # Delete the placeholder using repeated backspace
                    placeholder_len = placeholder_match.end() - placeholder_match.start()
                    buf.delete_before_cursor(count=placeholder_len)
                else:
                    # No placeholder, cursor at end
                    buf.cursor_position = len(expanded)
                return

        # Normal space behavior
        buf.insert_text(" ")

    # Create merged completer
    completer = merge_completers([
        CommandCompleter(),
        ModelCompleter(),
        ResourceCompleter(engine),
        PromptCompleter(engine),
        ConfigCompleter(),
        CommandNameCompleter(),
    ])

    # Bottom toolbar showing context stats
    def get_bottom_toolbar():
        """Generate bottom toolbar with context stats."""
        try:
            n_ctx = engine.llm.n_ctx()
            # Count tokens for each component
            system_tokens = engine._count_tokens([engine._build_system_message()])
            priming_tokens = engine._count_tokens(engine.priming_messages) if engine.priming_messages else 0
            history_tokens = engine._count_tokens(engine.history) if engine.history else 0
            total_tokens = system_tokens + priming_tokens + history_tokens
            pct = (total_tokens / n_ctx) * 100 if n_ctx > 0 else 0

            # Color based on usage
            if pct > 80:
                color = "ansired"
            elif pct > 60:
                color = "ansiyellow"
            else:
                color = "ansigreen"

            return HTML(
                f' <b>Context:</b> <{color}>{total_tokens:,}</{color}>/{n_ctx:,} ({pct:.0f}%) '
                f'| <b>History:</b> {len(engine.history)} msgs'
            )
        except Exception:
            return ""

    # Create prompt session
    session: PromptSession = PromptSession(
        history=history,
        completer=completer,
        auto_suggest=AutoSuggestFromHistory(),
        style=get_style(),
        key_bindings=bindings,
        complete_while_typing=True,
        enable_history_search=True,
        bottom_toolbar=get_bottom_toolbar,
    )

    # Initialize magic commands
    magic = MagicCommands(engine, session_mgr)

    # Get model info for banner
    model_name = Path(engine._model_path).stem if hasattr(engine, '_model_path') else "unknown"
    n_ctx = engine.llm.n_ctx() if hasattr(engine, 'llm') else 0

    # Print welcome banner
    print("=" * 50)
    print("LlamaCpp Chat REPL with Tool Calling")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Context: {n_ctx:,} tokens")
    print("-" * 50)
    print("Tab: completion | Ctrl+R: search history")
    print("Ctrl+C: cancel | Ctrl+D twice: exit")
    print("/help for commands")
    print("=" * 50)
    print()

    # Pre-fill text for next prompt (used by # prompt expansion on Enter)
    prefill_input = ""
    prefill_cursor_pos = -1  # -1 means end of input

    while True:
        try:
            # Set up cursor position callback if needed
            if prefill_cursor_pos >= 0:
                def set_cursor():
                    session.app.current_buffer.cursor_position = prefill_cursor_pos
                user_input = session.prompt("You: ", default=prefill_input, pre_run=set_cursor).strip()
            else:
                user_input = session.prompt("You: ", default=prefill_input).strip()
            prefill_input = ""  # Reset after use
            prefill_cursor_pos = -1
        except EOFError:
            # First Ctrl+D - wait for confirmation
            print("\nPress Ctrl+D again to exit, or ESC to cancel.")
            try:
                # Use a minimal prompt to wait for second Ctrl+D or ESC
                confirm = session.prompt("")
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
                # Ctrl+C or ESC - cancel exit
                print("\n[Exit cancelled]\n")
                continue  # Go back to show "You: " prompt
        except KeyboardInterrupt:
            # Ctrl+C during prompt - show fresh prompt
            print()  # Move to new line after ^C
            continue  # Loop will show "You: " prompt

        if not user_input:
            continue

        # Handle !! shortcut - expand with last response
        if "!!" in user_input:
            # Find the last assistant response
            last_response = None
            for msg in reversed(engine.history):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # Try to parse as JSON (tool response format)
                    try:
                        import json
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "content" in parsed:
                            last_response = parsed["content"]
                        else:
                            last_response = content
                    except (json.JSONDecodeError, TypeError):
                        last_response = content
                    break

            if last_response:
                user_input = user_input.replace("!!", last_response)
                print(f"{GREY}[!! expanded]{RESET}")
            else:
                print("No previous response to expand.")
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

        # Handle MCP resource fetch (@uri)
        if user_input.startswith("@"):
            _fetch_mcp_resource(engine, user_input[1:])
            continue

        # Handle MCP prompt expansion (#prompt_name arg=value)
        if user_input.startswith("#"):
            prompt_part = user_input[1:].strip()

            # Check if this is just "#prompt_name" without arguments
            if " " not in prompt_part:
                # No arguments - expand inline like Space does
                expanded = _expand_mcp_prompt_inline(engine, prompt_part)
                if expanded:
                    # If it still starts with #, it has placeholders - prefill for editing
                    if expanded.startswith("#"):
                        # Remove placeholder braces and position for editing
                        import re
                        # Find first placeholder and remove it
                        match = re.search(r'\{(\w+)\}', expanded)
                        if match:
                            # Remove the placeholder, set cursor at that position
                            prefill_input = expanded[:match.start()] + expanded[match.end():]
                            prefill_cursor_pos = match.start()
                        else:
                            prefill_input = expanded
                            prefill_cursor_pos = -1
                        # Clear the previous line (move up, clear line)
                        print("\033[A\033[2K", end="")
                        continue
                    else:
                        # No placeholders - use as input to model
                        user_input = expanded
                        print(f"{GREY}[#{prompt_part} expanded]{RESET}")
                        print(f"{GREY}{expanded}{RESET}")
                        print()
                        # Fall through to send to model
                else:
                    print(f"Prompt not found: {prompt_part}")
                    print("Use /mcp prompts to list available prompts.")
                    continue
            else:
                # Has arguments - expand with them
                expanded = _expand_mcp_prompt(engine, prompt_part)
                if expanded:
                    # Use the expanded prompt as the input to send to the model
                    user_input = expanded
                else:
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
            # Don't show spinner in debug mode - it interferes with debug output
            use_spinner = not getattr(engine, 'debug', False)
            spinner = WaitingAnimation() if use_spinner else None
            if spinner:
                spinner.start()
            first_token = True
            suggested_tool = None
            try:
                for token in engine.chat(user_input, stream=True):
                    # Check for tool suggestion (dict marker from weaker models)
                    if isinstance(token, dict) and token.get("type") == "tool_suggestion":
                        suggested_tool = token.get("tool")
                        continue

                    if first_token:
                        # Stop animation on first token (empty string sentinel)
                        if spinner:
                            spinner.stop()
                        first_token = False
                        # Skip empty sentinel token
                        if not token:
                            continue
                    # Decode HTML entities (e.g., &#34; -> ")
                    decoded = html.unescape(token)
                    print(decoded, end="", flush=True)
                    response += decoded
            finally:
                if spinner:
                    spinner.stop()
            print("\n")

            # Handle suggested tool call (from weaker models that mix text + JSON)
            if suggested_tool:
                tool_name = suggested_tool.get("name", "unknown")
                tool_args = suggested_tool.get("arguments", {})
                print(f"{GREY}[Detected tool: {tool_name}({tool_args})]{RESET}")
                try:
                    confirm = input(f"{GREY}Execute this tool? [y/N]: {RESET}").strip().lower()
                    if confirm == "y":
                        feedback(f"Executing {tool_name}...")
                        result = engine.execute_suggested_tool(suggested_tool)
                        print(f"{GREY}[Result: {result}]{RESET}\n")
                except (EOFError, KeyboardInterrupt):
                    print()  # Clean line after interrupt

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
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except Exception as e:
            print(f"\nError: {e}")
