"""
Built-in magic commands package.

Magic commands are loaded from individual subdirectories, each containing an __init__.py
that registers the command using @magic_registry.register().

This allows commands to be copied to ~/.l3m/magic/ for user customization,
with package versions serving as fallback.
"""
