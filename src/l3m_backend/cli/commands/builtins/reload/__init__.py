"""Reload command - reload tools, commands, and magic commands."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("reload", "Reload tools/commands/magic")
def cmd_reload(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Reload all tools, commands, and magic commands from user directories."""
    print("\nReloading...")

    # Track counts before/after
    results = []

    # Reload commands
    from l3m_backend.cli.commands.registry import command_registry
    from l3m_backend.cli.commands.loader import load_all_commands, USER_COMMANDS_DIR

    # Clear loaded command modules from sys.modules
    modules_to_remove = [m for m in sys.modules if m.startswith("l3m_cmd.")]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # Clear and reload commands
    old_count = len(command_registry._commands)
    command_registry._commands.clear()
    command_registry._aliases.clear()

    # Re-register the reload command itself (since we just cleared it!)
    command_registry.register("reload", "Reload tools/commands/magic")(cmd_reload)

    new_count = load_all_commands(USER_COMMANDS_DIR)
    results.append(f"  Commands: {new_count} loaded (was {old_count})")

    # Reload magic commands
    from l3m_backend.cli.magic.registry import magic_registry
    from l3m_backend.cli.magic.loader import load_all_magic, USER_MAGIC_DIR

    # Clear loaded magic modules from sys.modules
    modules_to_remove = [m for m in sys.modules if m.startswith("l3m_magic.")]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # Clear and reload magic
    old_count = len(magic_registry._commands)
    magic_registry._commands.clear()
    magic_registry._aliases.clear()
    new_count = load_all_magic(USER_MAGIC_DIR)
    results.append(f"  Magic: {new_count} loaded (was {old_count})")

    # Reload tools
    from l3m_backend.tools._registry import registry as tool_registry
    from l3m_backend.tools.loader import load_all_tools, USER_TOOLS_DIR

    # Clear loaded tool modules from sys.modules
    modules_to_remove = [m for m in sys.modules if m.startswith("l3m_tool.")]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # Clear and reload tools
    old_count = len(tool_registry._tools)
    tool_registry._tools.clear()
    tool_registry._aliases.clear()
    new_count = load_all_tools(USER_TOOLS_DIR)
    results.append(f"  Tools: {new_count} loaded (was {old_count})")

    # Update engine's tool contract after reloading tools
    try:
        from l3m_backend.engine.contract import load_contract_template, MINIMAL_CONTRACT_TEMPLATE
        from l3m_backend.engine.legacy_priming import generate_legacy_priming

        # Rebuild tools list
        engine.tools = tool_registry.to_openai_tools()

        # Rebuild priming messages
        engine.priming_messages = generate_legacy_priming(tool_registry)

        # Rebuild tools contract
        if engine.minimal_contract:
            engine.tools_contract = MINIMAL_CONTRACT_TEMPLATE.format(
                tool_list=tool_registry.to_minimal_list(),
            )
        else:
            engine.tools_contract = load_contract_template().format(
                registry_json=tool_registry.to_registry_json(),
            )
        results.append(f"  Engine: tools contract updated")
    except Exception as e:
        results.append(f"  Engine: failed to update ({e})")

    # Print results
    for line in results:
        print(line)
    print()
