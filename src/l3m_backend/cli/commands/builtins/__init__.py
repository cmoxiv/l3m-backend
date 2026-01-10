"""
Built-in commands package.

Commands are loaded from individual subdirectories, each containing an __init__.py
that registers the command using @command_registry.register().

This allows commands to be copied to ~/.l3m/commands/ for user customization,
with package versions serving as fallback.
"""
