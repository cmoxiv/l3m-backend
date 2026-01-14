"""Config command - show/set/delete configuration."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("config", "Config (set/del/show)", usage="/config [set <key> <value> | del <key>]", has_args=True)
def cmd_config(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show, set, or delete configuration values."""
    from l3m_backend.config import get_config_manager, DEFAULTS

    cfg_mgr = get_config_manager()
    parts = args.split(maxsplit=2) if args else []

    if not parts:
        # /config - show current settings
        settings = cfg_mgr.list_settings()
        print(f"\nConfig file: {cfg_mgr.CONFIG_FILE}")
        if settings:
            for key, value in settings.items():
                print(f"  {key}: {value}")
        else:
            print("  (no custom settings)")
        print()
    elif parts[0] == "set" and len(parts) >= 3:
        # /config set <key> <value>
        key = parts[1]
        value_str = parts[2]
        try:
            # Parse value based on type
            current = cfg_mgr.get(key)
            if current is None:
                current = DEFAULTS.get(key)

            if isinstance(current, bool) or value_str.lower() in ("true", "false"):
                value = value_str.lower() == "true"
            elif isinstance(current, int):
                value = int(value_str)
            elif isinstance(current, float):
                value = float(value_str)
            elif isinstance(current, list):
                import json
                value = json.loads(value_str)
            else:
                value = value_str

            cfg_mgr.set(key, value)
            print(f"Set {key} = {value}\n")
        except ValueError as e:
            print(f"Error: {e}\n")
        except Exception as e:
            print(f"Error setting {key}: {e}\n")
    elif parts[0] == "del" and len(parts) >= 2:
        # /config del <key>
        key = parts[1]
        try:
            cfg_mgr.unset(key)
            default = DEFAULTS.get(key, "null")
            print(f"Deleted {key} (default: {default})\n")
        except ValueError as e:
            print(f"Error: {e}\n")
    else:
        print("Usage: /config [set <key> <value> | del <key>]\n")
