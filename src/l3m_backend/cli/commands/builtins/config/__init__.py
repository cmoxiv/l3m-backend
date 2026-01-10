"""Config command - show current configuration."""
from __future__ import annotations

from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@command_registry.register("config", "Show current configuration")
def cmd_config(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show current configuration."""
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
