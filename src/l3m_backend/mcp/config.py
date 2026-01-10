"""MCP server configuration management."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from l3m_backend.mcp.exceptions import MCPConfigError, MCPServerNotFoundError


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    transport: str = Field(default="stdio", description="Transport type: stdio, http, sse")
    command: str | None = Field(default=None, description="Command to run (for stdio)")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    url: str | None = Field(default=None, description="Server URL (for http/sse)")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    auto_connect: bool = Field(default=False, description="Connect automatically on startup")
    description: str = Field(default="", description="Human-readable description")

    def resolve_env(self) -> dict[str, str]:
        """Resolve environment variable references like ${VAR}."""
        resolved = {}
        for key, value in self.env.items():
            if value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                resolved[key] = os.environ.get(env_var, "")
            else:
                resolved[key] = value
        return resolved


class MCPConfig(BaseModel):
    """Configuration for all MCP servers."""

    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    default_servers: list[str] = Field(default_factory=list)


class MCPConfigManager:
    """Manages MCP server configuration."""

    CONFIG_FILE = Path.home() / ".l3m" / "mcp_servers.json"

    def __init__(self, config_file: Path | None = None):
        self.config_file = config_file or self.CONFIG_FILE
        self._config: MCPConfig | None = None

    def load(self) -> MCPConfig:
        """Load configuration from file."""
        if self._config is not None:
            return self._config

        if not self.config_file.exists():
            self._config = MCPConfig()
            return self._config

        try:
            data = json.loads(self.config_file.read_text())
            self._config = MCPConfig.model_validate(data)
            return self._config
        except json.JSONDecodeError as e:
            raise MCPConfigError(f"Invalid JSON in {self.config_file}: {e}")
        except Exception as e:
            raise MCPConfigError(f"Error loading MCP config: {e}")

    def save(self) -> None:
        """Save configuration to file."""
        if self._config is None:
            return

        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        data = self._config.model_dump(mode="json")
        self.config_file.write_text(json.dumps(data, indent=2))

    def get_server(self, name: str) -> MCPServerConfig:
        """Get configuration for a specific server."""
        config = self.load()
        if name not in config.servers:
            raise MCPServerNotFoundError(f"MCP server '{name}' not found in configuration")
        return config.servers[name]

    def add_server(
        self,
        name: str,
        transport: str = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        env: dict[str, str] | None = None,
        auto_connect: bool = False,
        description: str = "",
    ) -> None:
        """Add or update a server configuration."""
        config = self.load()
        config.servers[name] = MCPServerConfig(
            transport=transport,
            command=command,
            args=args or [],
            url=url,
            env=env or {},
            auto_connect=auto_connect,
            description=description,
        )
        self.save()

    def remove_server(self, name: str) -> bool:
        """Remove a server configuration. Returns True if removed."""
        config = self.load()
        if name in config.servers:
            del config.servers[name]
            if name in config.default_servers:
                config.default_servers.remove(name)
            self.save()
            return True
        return False

    def list_servers(self) -> list[str]:
        """List all configured server names."""
        return list(self.load().servers.keys())

    def get_auto_connect_servers(self) -> list[str]:
        """Get list of servers configured for auto-connect."""
        config = self.load()
        return [name for name, srv in config.servers.items() if srv.auto_connect]

    def get_default_servers(self) -> list[str]:
        """Get list of default servers."""
        return self.load().default_servers


# Global config manager instance
_config_manager: MCPConfigManager | None = None


def get_mcp_config_manager() -> MCPConfigManager:
    """Get the global MCP config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = MCPConfigManager()
    return _config_manager
