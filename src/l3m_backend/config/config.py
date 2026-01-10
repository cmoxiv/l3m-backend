"""
Configuration management for l3m-backend.

Provides a configuration file at ~/.l3m/config.json for default settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


# Default values - single source of truth
DEFAULTS = {
    "ctx": 8192,
    "gpu": -1,
    "verbose": False,
    "show_warnings": False,
    "simple": False,
    "no_native_tools": False,
    "no_flash_attn": False,
    "minimal_contract": False,
    "incognito": False,
    "auto_resume": False,
    "temperature": 0.1,
    "system_prompt": "You are a helpful assistant with access to tools.",
}


class Config(BaseModel):
    """Configuration settings for l3m-backend.

    All settings are optional. Use DEFAULTS for default values.
    """

    model_config = {"extra": "ignore"}  # Ignore unknown fields like _comment

    # Model settings
    default_model: Optional[str] = Field(
        default=None,
        description="Default model name or path to use"
    )
    ctx: Optional[int] = Field(
        default=None,
        description="Default context size (tokens)"
    )
    gpu: Optional[int] = Field(
        default=None,
        description="Default GPU layers (-1 for all, 0 for none)"
    )
    verbose: Optional[bool] = Field(
        default=None,
        description="Enable verbose llama.cpp output"
    )
    show_warnings: Optional[bool] = Field(
        default=None,
        description="Show Metal/GPU initialization warnings"
    )
    no_native_tools: Optional[bool] = Field(
        default=None,
        description="Disable native function calling (use contract-only)"
    )
    no_flash_attn: Optional[bool] = Field(
        default=None,
        description="Disable flash attention (enabled by default)"
    )
    minimal_contract: Optional[bool] = Field(
        default=None,
        description="Use minimal tool contract (smaller, faster)"
    )

    # REPL settings
    simple: Optional[bool] = Field(
        default=None,
        description="Use simple REPL (no prompt_toolkit)"
    )

    # Session settings
    incognito: Optional[bool] = Field(
        default=None,
        description="Default to incognito mode"
    )
    auto_resume: Optional[bool] = Field(
        default=None,
        description="Automatically resume session from CWD without prompting"
    )

    # Chat settings
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt to use"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Default temperature for generation"
    )
    chat_format: Optional[str] = Field(
        default=None,
        description="Chat format (e.g., 'chatml-function-calling')"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Default max tokens for generation"
    )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value with fallback to DEFAULTS, then to provided default."""
        value = getattr(self, key, None)
        if value is not None:
            return value
        # Fall back to DEFAULTS, then to provided default
        return DEFAULTS.get(key, default)


class ConfigManager:
    """Manages loading and saving configuration."""

    CONFIG_DIR = Path.home() / ".l3m"
    CONFIG_FILE = CONFIG_DIR / "config.json"

    def __init__(self):
        self._config: Optional[Config] = None

    def _ensure_dir(self) -> None:
        """Ensure config directory exists."""
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> Config:
        """Get the current config, loading if necessary."""
        if self._config is None:
            self._config = self.load()
        return self._config

    def load(self, create_if_missing: bool = True) -> Config:
        """Load configuration from file.

        Args:
            create_if_missing: If True, create default config file if it doesn't exist.

        Returns:
            Config object with loaded settings, or defaults if file doesn't exist.
        """
        if not self.CONFIG_FILE.exists():
            if create_if_missing:
                self._create_default_config()
            return Config()

        try:
            data = json.loads(self.CONFIG_FILE.read_text())
            return Config.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            # Invalid config file, return defaults
            print(f"Warning: Invalid config file ({e}), using defaults")
            return Config()

    def _create_default_config(self) -> None:
        """Create default config file with actual default values."""
        self._ensure_dir()

        # Config with actual default values (matching argparse defaults)
        default_config = {
            "_comment": "l3m-backend configuration file",
            "default_model": None,
            "ctx": DEFAULTS["ctx"],
            "gpu": DEFAULTS["gpu"],
            "verbose": DEFAULTS["verbose"],
            "show_warnings": DEFAULTS["show_warnings"],
            "simple": DEFAULTS["simple"],
            "no_native_tools": DEFAULTS["no_native_tools"],
            "no_flash_attn": DEFAULTS["no_flash_attn"],
            "minimal_contract": DEFAULTS["minimal_contract"],
            "incognito": DEFAULTS["incognito"],
            "auto_resume": DEFAULTS["auto_resume"],
            "temperature": DEFAULTS["temperature"],
            "system_prompt": DEFAULTS["system_prompt"],
            "chat_format": None,
            "max_tokens": None,
        }
        self.CONFIG_FILE.write_text(json.dumps(default_config, indent=2) + "\n")

    def save(self, config: Optional[Config] = None) -> Path:
        """Save configuration to file, preserving existing structure.

        Args:
            config: Config to save. If None, saves current config.

        Returns:
            Path to saved config file.
        """
        self._ensure_dir()
        if config is not None:
            self._config = config

        if self._config is None:
            self._config = Config()

        # Read existing file to preserve comments and structure
        existing_data = {}
        if self.CONFIG_FILE.exists():
            try:
                existing_data = json.loads(self.CONFIG_FILE.read_text())
            except json.JSONDecodeError:
                pass

        # Update only non-None config values, preserving everything else
        for key, value in self._config.model_dump().items():
            if value is not None:
                existing_data[key] = value

        self.CONFIG_FILE.write_text(json.dumps(existing_data, indent=2) + "\n")
        return self.CONFIG_FILE

    def set(self, key: str, value: Any) -> None:
        """Set a config value and save.

        Args:
            key: Config key to set.
            value: Value to set.
        """
        # Always reload from file to get latest values
        self._config = self.load(create_if_missing=True)

        if not hasattr(self._config, key):
            raise ValueError(f"Unknown config key: {key}")

        setattr(self._config, key, value)
        self.save()

    def unset(self, key: str) -> None:
        """Remove a config value (reset to default).

        Args:
            key: Config key to unset.
        """
        # Reload config
        self._config = self.load(create_if_missing=True)

        if not hasattr(self._config, key):
            raise ValueError(f"Unknown config key: {key}")

        # Set to None in config
        setattr(self._config, key, None)

        # Read existing file and remove the key
        existing_data = {}
        if self.CONFIG_FILE.exists():
            try:
                existing_data = json.loads(self.CONFIG_FILE.read_text())
            except json.JSONDecodeError:
                pass

        # Remove the key if present, or set to null
        if key in existing_data:
            existing_data[key] = None

        self.CONFIG_FILE.write_text(json.dumps(existing_data, indent=2) + "\n")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value with fallback.

        Args:
            key: Config key to get.
            default: Default value if not set.

        Returns:
            Config value or default.
        """
        return self.config.get(key, default)

    def list_settings(self) -> dict[str, Any]:
        """List user-customized settings (values that differ from defaults).

        Returns:
            Dict of settings that differ from DEFAULTS.
        """
        result = {}
        for k, v in self.config.model_dump().items():
            if v is None:
                continue
            # Include if not in DEFAULTS or differs from default
            if k not in DEFAULTS or v != DEFAULTS[k]:
                result[k] = v
        return result

    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._config = Config()
        if self.CONFIG_FILE.exists():
            self.CONFIG_FILE.unlink()


# Singleton instance
_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the singleton ConfigManager instance."""
    global _manager
    if _manager is None:
        _manager = ConfigManager()
    return _manager


def get_config() -> Config:
    """Get the current configuration."""
    return get_config_manager().config
