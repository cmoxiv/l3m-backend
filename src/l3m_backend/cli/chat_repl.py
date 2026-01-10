#!/usr/bin/env python3
"""
CLI entry point for the chat REPL (l3m-chat command).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from l3m_backend.config import DEFAULTS, get_config_manager

if TYPE_CHECKING:
    from l3m_backend.session import SessionManager

# Default models directory
DEFAULT_MODELS_DIR = Path.home() / ".l3m" / "models"


def list_models() -> list[Path]:
    """List available GGUF models in the default directory."""
    if not DEFAULT_MODELS_DIR.exists():
        return []
    return sorted(DEFAULT_MODELS_DIR.glob("*.gguf"))


def print_models():
    """Print available models."""
    models = list_models()
    if not models:
        print(f"No models found in {DEFAULT_MODELS_DIR}")
        print("\nDownload a model with:")
        print("  l3m-download --preset llama3.2-1b")
        return

    print(f"Available models in {DEFAULT_MODELS_DIR}:\n")
    for model in models:
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"  {model.name:<50} {size_mb:>8.1f} MB")
    print()


def resolve_model(model_arg: str | None) -> Path | None:
    """Resolve model path from argument or default directory."""
    if model_arg:
        # Check if it's a full path
        path = Path(model_arg)
        if path.exists():
            return path
        # Check if it's a name in the default directory
        default_path = DEFAULT_MODELS_DIR / model_arg
        if default_path.exists():
            return default_path
        # Not found
        print(f"Error: Model not found: {model_arg}")
        print(f"\nLooked in:")
        print(f"  - {path.absolute()}")
        print(f"  - {default_path}")
        return None

    # No model specified, try to find one in default directory
    models = list_models()
    if not models:
        print(f"Error: No model specified and no models found in {DEFAULT_MODELS_DIR}")
        print("\nDownload a model with:")
        print("  l3m-download --preset llama3.2-1b")
        print("\nOr specify a model path:")
        print("  l3m-chat /path/to/model.gguf")
        return None

    if len(models) == 1:
        print(f"Using model: {models[0].name}")
        return models[0]

    # Multiple models, ask user to choose
    print(f"Multiple models found in {DEFAULT_MODELS_DIR}:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model.name}")
    print("\nSpecify which model to use:")
    print(f"  l3m-chat {models[0].name}")
    return None


def print_config():
    """Print current configuration."""
    cfg_mgr = get_config_manager()
    settings = cfg_mgr.list_settings()

    print(f"Config file: {cfg_mgr.CONFIG_FILE}")

    if settings:
        print("\nCustom settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")

    print("\nDefaults (used when not set):")
    for key, value in DEFAULTS.items():
        if key not in settings:
            print(f"  {key}: {value}")

    print("\nSet with: l3m-chat --set-config key=value")
    print("Available keys: default_model, ctx, gpu, verbose, simple,")
    print("                no_native_tools, no_flash_attn, incognito,")
    print("                auto_resume, temperature, system_prompt,")
    print("                chat_format, max_tokens")
    print()


def main():
    """Main entry point for the l3m-chat CLI."""
    # Load config for defaults
    cfg_mgr = get_config_manager()
    cfg = cfg_mgr.config

    parser = argparse.ArgumentParser(
        description="Chat REPL with tool calling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Models are loaded from {DEFAULT_MODELS_DIR} by default.
Config file: {cfg_mgr.CONFIG_FILE}

Examples:
    l3m-chat                           # Use model from default dir
    l3m-chat model.gguf                # Use model by name
    l3m-chat /path/to/model.gguf       # Use model by path
    l3m-chat --list                    # List available models
    l3m-chat --config                  # Show current config
    l3m-chat --set-config ctx=8192     # Set default context size
        """,
    )
    default_model = cfg.get("default_model")
    model_help = "Model name or path to GGUF file"
    if default_model:
        model_help += f" (default: {default_model})"
    parser.add_argument("model", nargs="?", default=default_model, help=model_help)
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    parser.add_argument("--ctx", type=int, default=cfg.get("ctx"),
                        help=f"Context size (default: {cfg.get('ctx')})")
    parser.add_argument("--gpu", type=int, default=cfg.get("gpu"),
                        help=f"GPU layers (default: {cfg.get('gpu')})")
    parser.add_argument("-v", "--verbose", action="store_true",
                        default=cfg.get("verbose"),
                        help="Verbose llama.cpp output")
    parser.add_argument("--show-warnings", action="store_true",
                        default=cfg.get("show_warnings"),
                        help="Show Metal/GPU initialization warnings")
    parser.add_argument("--simple", action="store_true",
                        default=cfg.get("simple"),
                        help="Use simple REPL (no prompt_toolkit features)")

    # Native tool calling
    parser.add_argument("--no-native-tools", action="store_true",
                        default=cfg.get("no_native_tools"),
                        help=f"Disable native function calling (default: {cfg.get('no_native_tools')})")
    parser.add_argument("--chat-format", type=str,
                        default=cfg.get("chat_format"),
                        help="Chat format (e.g., 'chatml-function-calling')")
    parser.add_argument("--temperature", "-t", type=float,
                        default=cfg.get("temperature"),
                        help=f"Temperature for generation (default: {cfg.get('temperature')})")
    parser.add_argument("--no-flash-attn", action="store_true",
                        default=cfg.get("no_flash_attn"),
                        help=f"Disable flash attention (default: {cfg.get('no_flash_attn')})")
    parser.add_argument("--system-prompt", type=str,
                        default=cfg.get("system_prompt"),
                        help="System prompt for the assistant")
    parser.add_argument("--minimal-contract", action="store_true",
                        default=cfg.get("minimal_contract"),
                        help="Use compact tool list instead of full JSON schemas (faster)")

    # Session management
    parser.add_argument("--session", "-s", metavar="NAME",
                        help="Session name or ID to create/resume")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume session from CWD symlink")
    parser.add_argument("--incognito", action="store_true",
                        default=cfg.get("incognito"),
                        help="Incognito mode (session stored in /tmp)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip KV cache warmup when resuming sessions")
    parser.add_argument("--summary-ctx", type=int, default=0,
                        help="Context tokens for session summaries (0=disabled, -1=4096, >0=fixed)")
    parser.add_argument("--transcript-ctx", type=int, default=0,
                        help="Context tokens for transcript excerpts (0=disabled, -1=4096, >0=fixed)")
    parser.add_argument("--list-sessions", action="store_true",
                        help="List available sessions")
    parser.add_argument("--search-sessions", metavar="QUERY",
                        help="Search sessions by title/content")

    # Debug mode
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with timing profiling")
    parser.add_argument("--test-tools", action="store_true",
                        help="Register test tools (get_flumbuster, calculate_zorbix)")

    # Config management
    parser.add_argument("--config", action="store_true",
                        help="Show current configuration")
    parser.add_argument("--set-config", metavar="KEY=VALUE",
                        help=("Set a config value. Keys: default_model, ctx, gpu, "
                              "verbose, show_warnings, simple, no_native_tools, no_flash_attn, "
                              "minimal_contract, incognito, auto_resume, temperature, "
                              "system_prompt, chat_format, max_tokens"))
    parser.add_argument("--unset-config", metavar="KEY",
                        help="Unset a config value (reset to default)")

    # MCP support
    parser.add_argument("--mcp", nargs="*", metavar="SERVER",
                        help="Connect to MCP server(s) at startup")
    parser.add_argument("--no-mcp", action="store_true",
                        help="Disable MCP server connections")
    parser.add_argument("--mcp-add", nargs=2, metavar=("NAME", "COMMAND"),
                        help="Add MCP server config: --mcp-add name 'command args'")
    parser.add_argument("--mcp-list", action="store_true",
                        help="List configured MCP servers")
    parser.add_argument("--mcp-remove", metavar="NAME",
                        help="Remove MCP server config")

    args = parser.parse_args()

    # Handle --config
    if args.config:
        print_config()
        return

    # Handle --set-config
    if args.set_config:
        try:
            key, value = args.set_config.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Type conversion based on key
            if key in ("ctx", "gpu", "max_tokens"):
                value = int(value)
            elif key in ("temperature",):
                value = float(value)
            elif key in ("verbose", "show_warnings", "simple", "no_native_tools", "no_flash_attn", "minimal_contract", "incognito", "auto_resume"):
                value = value.lower() in ("true", "1", "yes")

            cfg_mgr.set(key, value)
            print(f"Set {key} = {value}")
            print(f"Saved to {cfg_mgr.CONFIG_FILE}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    # Handle --unset-config
    if args.unset_config:
        try:
            cfg_mgr.unset(args.unset_config)
            print(f"Unset {args.unset_config}")
            print(f"Saved to {cfg_mgr.CONFIG_FILE}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    # Handle --mcp-list
    if args.mcp_list:
        try:
            from l3m_backend.mcp import get_mcp_config_manager
            mgr = get_mcp_config_manager()
            servers = mgr.list_servers()
            if not servers:
                print("No MCP servers configured.")
                print("Add one with: l3m-chat --mcp-add <name> '<command>'")
            else:
                print("Configured MCP servers:")
                config = mgr.load()
                for name in servers:
                    srv = config.servers[name]
                    auto = " [auto]" if srv.auto_connect else ""
                    print(f"  {name}{auto}")
                    if srv.command:
                        print(f"    Command: {srv.command} {' '.join(srv.args)}")
        except ImportError:
            print("MCP support not installed. Install with: pip install 'l3m-backend[mcp]'")
        return

    # Handle --mcp-add
    if args.mcp_add:
        try:
            from l3m_backend.mcp import get_mcp_config_manager
            name, command = args.mcp_add
            # Parse command string into command and args
            parts = command.split()
            cmd = parts[0]
            cmd_args = parts[1:] if len(parts) > 1 else []
            mgr = get_mcp_config_manager()
            mgr.add_server(name=name, transport="stdio", command=cmd, args=cmd_args)
            print(f"Added MCP server: {name}")
            print(f"  Command: {command}")
            print(f"Connect with: l3m-chat --mcp {name}")
        except ImportError:
            print("MCP support not installed. Install with: pip install 'l3m-backend[mcp]'")
        return

    # Handle --mcp-remove
    if args.mcp_remove:
        try:
            from l3m_backend.mcp import get_mcp_config_manager
            mgr = get_mcp_config_manager()
            if mgr.remove_server(args.mcp_remove):
                print(f"Removed MCP server: {args.mcp_remove}")
            else:
                print(f"Server not found: {args.mcp_remove}")
        except ImportError:
            print("MCP support not installed. Install with: pip install 'l3m-backend[mcp]'")
        return

    # Handle --list
    if args.list:
        print_models()
        return

    # Handle --list-sessions or --search-sessions (doesn't need model)
    if args.list_sessions or args.search_sessions:
        from l3m_backend.session import SessionManager
        mgr = SessionManager()

        if args.search_sessions:
            sessions = mgr.search(args.search_sessions)
            if not sessions:
                print(f"No sessions matching '{args.search_sessions}'")
                return
            print(f"Sessions matching '{args.search_sessions}':\n")
        else:
            sessions = mgr.list_sessions()
            if not sessions:
                print("No sessions found.")
                return
            print("Available sessions:\n")

        for s in sessions[:20]:  # Limit to 20
            title = s.title or "(untitled)"
            tag_str = f" [{s.tag}]" if s.tag else ""
            print(f"  {s.id[:8]}  {title:<40}{tag_str}")
            print(f"           {s.updated_at[:16]}  {s.working_directory}")
        if len(sessions) > 20:
            print(f"\n  ... and {len(sessions) - 20} more")
        print()
        return

    # Resolve model path
    model_path = resolve_model(args.model)
    if not model_path:
        sys.exit(1)

    # Check for llama-cpp-python
    try:
        from llama_cpp import Llama  # noqa: F401
    except ImportError:
        print("Error: llama-cpp-python is required for the chat REPL.")
        print("Install it with: pip install 'l3m-backend[llm]'")
        sys.exit(1)

    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager
    from l3m_backend.tools import registry

    # Register test tools if requested
    if args.test_tools:
        from l3m_backend.tools.test_tools import get_flumbuster, calculate_zorbix
        registry.register(get_flumbuster)
        registry.register(calculate_zorbix)
        print(f"Registered test tools: get_flumbuster, calculate_zorbix")
        print(f"Registry now has {len(registry.to_openai_tools())} tools")

    # Choose REPL implementation
    use_simple = args.simple
    if not use_simple:
        try:
            from l3m_backend.cli._repl import repl
        except ImportError:
            print("Note: prompt_toolkit not available, using simple REPL")
            use_simple = True

    if use_simple:
        from l3m_backend.cli._simple_repl import repl

    # Initialize session manager
    cwd = Path.cwd()
    session_mgr = SessionManager()
    session_id_to_resume = None

    # Determine if we should resume a session
    if args.resume:
        # Look for session symlink in CWD
        session_id_to_resume = session_mgr.find_session_in_cwd(cwd)
        if not session_id_to_resume:
            print("No session found in current directory.")
            print("Start a new session or specify --session <id>")
            sys.exit(1)
    elif args.session:
        # Session name/ID specified
        session_id_to_resume = args.session
    else:
        # Check if there's a session symlink in CWD and offer to resume
        existing_session = session_mgr.find_session_in_cwd(cwd)
        if existing_session:
            # Auto-resume if configured, otherwise prompt
            if cfg.get("auto_resume"):
                print(f"Auto-resuming session: {existing_session[:8]}")
                session_id_to_resume = existing_session
            else:
                print(f"Found existing session in this directory: {existing_session[:8]}")
                try:
                    # Only prompt if stdin is a tty (not in tests or pipes)
                    if sys.stdin.isatty():
                        response = input("Resume this session? [Y/n] ").strip().lower()
                        if response in ("", "y", "yes"):
                            session_id_to_resume = existing_session
                except (EOFError, KeyboardInterrupt, OSError):
                    print()

    # Create context partition if needed
    from l3m_backend.engine.context import ContextPartition
    partition = None
    if args.summary_ctx != 0 or args.transcript_ctx != 0:
        partition = ContextPartition(
            summary_tokens=args.summary_ctx,
            transcript_tokens=args.transcript_ctx,
        )

    # Connect to MCP servers (on by default, use --no-mcp to disable)
    mcp_client = None
    mcp_adapter = None
    if not args.no_mcp:
        try:
            from l3m_backend.mcp import MCPClient, MCPRegistryAdapter, get_mcp_config_manager
            from l3m_backend.mcp.client.client import run_async
            import logging

            # Configure MCP logging to use grey color
            mcp_logger = logging.getLogger("mcp")
            mcp_logger.handlers = []
            mcp_handler = logging.StreamHandler(sys.stderr)
            mcp_handler.setFormatter(logging.Formatter("\033[90m%(message)s\033[0m"))
            mcp_logger.addHandler(mcp_handler)
            mcp_logger.setLevel(logging.WARNING)  # Only show warnings and errors

            mcp_client = MCPClient()
            mcp_adapter = MCPRegistryAdapter(registry, mcp_client)

            # Determine which servers to connect
            if args.mcp:  # Explicit servers take priority
                servers_to_connect = args.mcp
            else:  # Default: connect auto_connect servers
                servers_to_connect = get_mcp_config_manager().get_auto_connect_servers()

            for server_name in servers_to_connect:
                try:
                    run_async(mcp_client.connect(server_name))
                    conn = mcp_client.get_connection(server_name)
                    if conn:
                        print(f"\033[90mConnected to MCP server: {server_name} ({len(conn.tools)} tools)\033[0m")
                except Exception as e:
                    print(f"\033[33mWarning: Failed to connect to MCP server '{server_name}': {e}\033[0m")

            # Register MCP tools with registry
            if mcp_client.connected_servers:
                tool_count = run_async(mcp_adapter.register_all())
                if tool_count > 0:
                    print(f"\033[90mRegistered {tool_count} MCP tools\033[0m")
        except ImportError:
            pass  # MCP not installed, silently skip

    # Create engine
    engine = ChatEngine(
        model_path=str(model_path),
        registry=registry,
        system_prompt=args.system_prompt,
        n_ctx=args.ctx,
        n_gpu_layers=args.gpu,
        verbose=args.verbose,
        show_warnings=args.show_warnings,
        use_native_tools=not args.no_native_tools,
        chat_format=args.chat_format,
        temperature=args.temperature,
        flash_attn=not args.no_flash_attn,
        debug=args.debug,
        minimal_contract=args.minimal_contract,
        context_partition=partition,
    )

    # Store MCP client on engine for /mcp command access
    engine._mcp_client = mcp_client
    engine._mcp_adapter = mcp_adapter

    # Show GPU backend info
    from l3m_backend.utils.gpu import format_gpu_status
    print(f"\033[90mBackend: {format_gpu_status(engine)}\033[0m", file=sys.stderr)

    # Load or create session
    if session_id_to_resume:
        try:
            session_mgr.load(session_id_to_resume)
            # Restore engine history
            engine.history = session_mgr.get_engine_history()

            # Populate context partitions if enabled
            if engine.partition.has_partitions():
                # Load summaries into partition
                if engine.partition.summary_tokens != 0:
                    summaries = session_mgr.session.metadata.summaries or []
                    engine.partition.loaded_summaries = [s.summary for s in summaries]
                    if engine.partition.loaded_summaries:
                        print(f"\033[90mLoaded {len(engine.partition.loaded_summaries)} summaries into context\033[0m")

                # Load transcript into partition
                if engine.partition.transcript_tokens != 0:
                    transcript = session_mgr.session.transcript or []
                    engine.partition.loaded_transcript = [
                        {"role": msg.role, "content": msg.content}
                        for msg in transcript
                    ]
                    if engine.partition.loaded_transcript:
                        print(f"\033[90mLoaded {len(engine.partition.loaded_transcript)} transcript messages into context\033[0m")

            # Display session details
            meta = session_mgr.session.metadata
            print(f"\nResumed session: {session_mgr.session.get_display_title()}")
            print(f"  ID:      {meta.id[:8]}")
            if meta.tag:
                print(f"  Tag:     {meta.tag}")
            print(f"  Created: {meta.initial_datetime}")
            if meta.last_save_datetime:
                print(f"  Saved:   {meta.last_save_datetime}")
            if meta.is_incognito:
                print(f"  (Incognito: {session_mgr._get_session_path(session_mgr.session)})")
            print()

            # Display transcript
            if session_mgr.session.transcript:
                print("--- Previous conversation ---")
                for msg in session_mgr.session.transcript:
                    role = msg.role.capitalize()
                    content = msg.content
                    if len(content) > 500:
                        content = content[:500] + "..."
                    print(f"\n{role}: {content}")
                print("\n--- End of previous conversation ---\n")

            # Run warmup to prime KV cache (unless --no-warmup)
            if not args.no_warmup and session_mgr.session.transcript:
                print("\033[90mWarming up KV cache...\033[0m", end="", flush=True)
                # Prepare warmup data
                transcript = [
                    {"role": msg.role, "content": msg.content}
                    for msg in session_mgr.session.transcript
                ]
                summaries = [
                    s.summary for s in session_mgr.session.metadata.summaries
                ] if session_mgr.session.metadata.summaries else None

                warmup_info = engine.warmup(transcript=transcript, summaries=summaries)
                print(f"\r\033[90mWarmed up: {warmup_info['tokens']} tokens in {warmup_info['time_s']:.1f}s\033[0m")
        except FileNotFoundError:
            print(f"Session not found: {session_id_to_resume}")
            print("Creating new session instead.")
            session_mgr.create(
                model_path=str(model_path),
                loaded_model_path=str(model_path.resolve()),
                working_directory=str(cwd),
                n_ctx=args.ctx,
                incognito=args.incognito,
            )
    else:
        # Create new session
        session_mgr.create(
            model_path=str(model_path),
            loaded_model_path=str(model_path.resolve()),
            working_directory=str(cwd),
            n_ctx=args.ctx,
            incognito=args.incognito,
        )
        if args.incognito:
            path = session_mgr._get_session_path(session_mgr.session)
            print(f"Incognito session: {path}")

    # Save session and create symlink
    session_mgr.save()
    session_mgr.create_symlink(cwd)

    try:
        repl(engine, session_mgr)
    finally:
        # Cleanup MCP connections
        if mcp_client:
            try:
                from l3m_backend.mcp.client.client import run_async
                run_async(mcp_client.disconnect_all())
            except Exception:
                pass


if __name__ == "__main__":
    main()
