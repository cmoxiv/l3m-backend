"""
Example user-defined command for l3m-chat.

This file demonstrates how to create custom /commands.
Place your command in ~/.l3m/commands/<name>/__init__.py

Commands receive:
    - engine: ChatEngine instance
    - session_mgr: SessionManager instance (or None)
    - args: Arguments string after the command name

Return True to exit the REPL, or None/False to continue.
"""

from l3m_backend.cli.commands import command_registry


@command_registry.register(
    name="example",
    description="Example custom command",
    usage="/example [message]",
    has_args=True,
)
def cmd_example(engine, session_mgr, args):
    """Example command that prints a greeting."""
    message = args.strip() if args else "Hello from custom command!"
    print(f"\n[Example Command]\n{message}\n")

    # Example: Access engine info
    print(f"Current history: {len(engine.history)} messages")

    # Example: Access session info
    if session_mgr and session_mgr.session:
        print(f"Session ID: {session_mgr.session.metadata.id[:8]}")

    print()
    # Return None or False to continue, True to exit REPL
