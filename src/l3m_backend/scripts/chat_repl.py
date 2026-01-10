#!/usr/bin/env python3
"""
Standalone feature-rich Chat REPL for testing and development.

Usage:
    pip install -e '.[llm]'
    python scripts/chat_repl.py ./model.gguf
    python scripts/chat_repl.py ./model.gguf --ctx 16384 --gpu 32

Features:
    - Command history (persistent across sessions)
    - Tab completion for /commands
    - Multi-line input (paste or type)
    - Streaming output display
    - Tool execution with formatted results
    - Conversation export/import
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


# Available commands
COMMANDS = {
    "/clear": "Clear conversation history",
    "/tools": "List available tools",
    "/system": "Show system prompt",
    "/schema": "Show tool schema (OpenAI format)",
    "/contract": "Show full system message with tools contract",
    "/history": "Show conversation history",
    "/export": "Export conversation to JSON file",
    "/import": "Import conversation from JSON file",
    "/stats": "Show session statistics",
    "/help": "Show this help message",
    "/quit": "Exit (also /exit, /q)",
}


class CommandCompleter(Completer):
    """Completer for REPL commands."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
            for cmd, description in COMMANDS.items():
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text), display_meta=description)
            if "/exit".startswith(text):
                yield Completion("/exit", start_position=-len(text), display_meta="Exit")
            if "/q".startswith(text):
                yield Completion("/q", start_position=-len(text), display_meta="Exit")


def get_style():
    """Get the prompt style."""
    return Style.from_dict({
        "prompt": "ansicyan bold",
        "command": "ansigreen",
    })


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("  LlamaCpp Chat REPL with Tool Calling")
    print("=" * 60)
    if HAS_PROMPT_TOOLKIT:
        print("  Type /help for commands, Tab for completion")
        print("  Ctrl+C to cancel, Ctrl+D to exit")
    else:
        print("  Type /help for commands")
        print("  (Install prompt_toolkit for enhanced features)")
    print("=" * 60)
    print()


def print_tools(registry):
    """Print available tools."""
    print("\nAvailable tools:")
    for entry in registry:
        aliases = f" (aliases: {', '.join(entry.aliases)})" if entry.aliases else ""
        print(f"  - {entry.name}{aliases}")
        desc = entry.get_description()
        if desc:
            print(f"    {desc}")
    print()


def print_help():
    """Print help message."""
    print("\nCommands:")
    for cmd, description in COMMANDS.items():
        print(f"  {cmd:<12} - {description}")
    print()


def print_history(messages):
    """Print conversation history."""
    print()
    for i, msg in enumerate(messages):
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


def export_conversation(engine, filepath: str):
    """Export conversation to JSON file."""
    data = {
        "exported_at": datetime.now().isoformat(),
        "system_prompt": engine.system_prompt,
        "messages": engine.messages,
    }
    path = Path(filepath)
    path.write_text(json.dumps(data, indent=2))
    print(f"Conversation exported to: {path.absolute()}\n")


def import_conversation(engine, filepath: str):
    """Import conversation from JSON file."""
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}\n")
        return
    try:
        data = json.loads(path.read_text())
        engine.messages = data.get("messages", [])
        print(f"Imported {len(engine.messages)} messages from: {filepath}\n")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}\n")


def print_stats(engine, start_time, message_count):
    """Print session statistics."""
    elapsed = datetime.now() - start_time
    print(f"\nSession Statistics:")
    print(f"  Duration: {elapsed}")
    print(f"  Messages sent: {message_count}")
    print(f"  History length: {len(engine.messages)}")
    print(f"  Tools available: {len(list(engine.registry))}")
    print()


def handle_command(cmd: str, engine, start_time, message_count) -> bool:
    """Handle a command. Returns True if should exit."""
    cmd_lower = cmd.lower()
    parts = cmd.split(maxsplit=1)
    cmd_name = parts[0].lower()
    cmd_arg = parts[1] if len(parts) > 1 else ""

    if cmd_lower in ("/quit", "/exit", "/q"):
        print("Goodbye!")
        return True
    elif cmd_lower == "/clear":
        engine.clear()
        print("Conversation cleared.\n")
    elif cmd_lower == "/tools":
        print_tools(engine.registry)
    elif cmd_lower == "/system":
        print(f"\n{engine.system_prompt}\n")
    elif cmd_lower == "/schema":
        print(f"\n{json.dumps(engine.tools, indent=2)}\n")
    elif cmd_lower == "/contract":
        print(f"\n{engine._build_system_message()['content']}\n")
    elif cmd_lower == "/history":
        print_history(engine.messages)
    elif cmd_name == "/export":
        filename = cmd_arg or f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_conversation(engine, filename)
    elif cmd_name == "/import":
        if not cmd_arg:
            print("Usage: /import <filename>\n")
        else:
            import_conversation(engine, cmd_arg)
    elif cmd_lower == "/stats":
        print_stats(engine, start_time, message_count)
    elif cmd_lower == "/help":
        print_help()
    else:
        print(f"Unknown command: {cmd}")
        print("Type /help for available commands.\n")
    return False


def repl_with_prompt_toolkit(engine):
    """Run REPL with prompt_toolkit features."""
    history_file = Path.home() / ".l3m" / "prompt_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history = FileHistory(str(history_file))

    bindings = KeyBindings()

    @bindings.add("c-c")
    def _(event):
        event.app.current_buffer.reset()
        print()

    session = PromptSession(
        history=history,
        completer=CommandCompleter(),
        style=get_style(),
        key_bindings=bindings,
        complete_while_typing=True,
        enable_history_search=True,
    )

    print_banner()
    start_time = datetime.now()
    message_count = 0

    while True:
        try:
            user_input = session.prompt("You: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:
            print()
            continue

        if not user_input:
            continue

        if user_input.startswith("/"):
            if handle_command(user_input, engine, start_time, message_count):
                break
            continue

        try:
            message_count += 1
            response = engine.chat(user_input)
            print(f"\nAssistant: {response}\n")
        except KeyboardInterrupt:
            print("\n[Interrupted]\n")
        except Exception as e:
            print(f"\nError: {e}\n")


def repl_simple(engine):
    """Run simple REPL without prompt_toolkit."""
    print_banner()
    start_time = datetime.now()
    message_count = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if handle_command(user_input, engine, start_time, message_count):
                break
            continue

        try:
            message_count += 1
            response = engine.chat(user_input)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Feature-rich Chat REPL with tool calling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/chat_repl.py ./model.gguf
    python scripts/chat_repl.py ./model.gguf --ctx 16384
    python scripts/chat_repl.py ./model.gguf --gpu 0  # CPU only
    python scripts/chat_repl.py ./model.gguf --simple
        """,
    )
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("--ctx", type=int, default=32768, help="Context size (default: 32768)")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU layers (-1=all, 0=none)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose llama.cpp output")
    parser.add_argument("--simple", action="store_true", help="Use simple REPL (no prompt_toolkit)")
    parser.add_argument("--system", type=str, help="Custom system prompt")
    args = parser.parse_args()

    # Check for llama-cpp-python
    try:
        from llama_cpp import Llama  # noqa: F401
    except ImportError:
        print("Error: llama-cpp-python is required.")
        print("Install with: pip install 'l3m-backend[llm]'")
        sys.exit(1)

    # Check model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {args.model}")
        print("\nTo download a model, run:")
        print("  python scripts/download_gguf.py --help")
        sys.exit(1)

    from l3m_backend.engine import ChatEngine
    from l3m_backend.tools import registry

    print(f"Loading model: {args.model}")
    print(f"Context size: {args.ctx}, GPU layers: {args.gpu}")

    engine = ChatEngine(
        model_path=args.model,
        registry=registry,
        n_ctx=args.ctx,
        n_gpu_layers=args.gpu,
        verbose=args.verbose,
        system_prompt=args.system,
    )

    if args.simple or not HAS_PROMPT_TOOLKIT:
        repl_simple(engine)
    else:
        repl_with_prompt_toolkit(engine)


if __name__ == "__main__":
    main()
