"""
Example user-defined magic command for l3m-chat.

This file demonstrates how to create custom %magic commands.
Place your command in ~/.l3m/magic/<name>/__init__.py

Magic commands receive:
    - engine: ChatEngine instance
    - session_mgr: SessionManager instance (or None)
    - args: Arguments string after the command name
    - add_to_history: Function to add user/assistant exchange to history

Magic commands typically add their output to conversation history,
making it visible to the LLM in future turns.
"""

from l3m_backend.cli.magic import magic_registry


@magic_registry.register(
    name="example",
    description="Example magic command (adds to history)",
    usage="%example [message]",
)
def magic_example(engine, session_mgr, args, add_to_history):
    """Example magic command that adds a greeting to history."""
    message = args.strip() if args else "Hello from magic command!"

    # Create the output that will be shown and added to history
    output = f"[Magic Example]\n{message}\nHistory has {len(engine.history)} messages."

    # Print the output
    print(output)

    # Add to history as user message + assistant response
    # This makes the exchange visible to the LLM
    user_msg = f"%example {args}" if args else "%example"
    add_to_history(user_msg, output)

    print()
    return True  # Command was handled
