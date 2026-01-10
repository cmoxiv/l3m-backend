#!/usr/bin/env python3
"""
Tests for configuration management module.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from l3m_backend.config import Config, ConfigManager, get_config, get_config_manager


# ============================================================================
# Config Model Tests
# ============================================================================

class TestConfig:
    """Tests for Config model."""

    def test_create_empty_config(self):
        """Test creating config with all defaults."""
        cfg = Config()
        assert cfg.default_model is None
        assert cfg.ctx is None
        assert cfg.gpu is None
        assert cfg.verbose is None
        assert cfg.simple is None
        assert cfg.incognito is None
        assert cfg.auto_resume is None
        assert cfg.system_prompt is None
        assert cfg.temperature is None
        assert cfg.max_tokens is None

    def test_create_config_with_values(self):
        """Test creating config with specific values."""
        cfg = Config(
            default_model="llama3.gguf",
            ctx=8192,
            gpu=0,
            verbose=True,
        )
        assert cfg.default_model == "llama3.gguf"
        assert cfg.ctx == 8192
        assert cfg.gpu == 0
        assert cfg.verbose is True

    def test_get_with_value(self):
        """Test get method when value exists."""
        cfg = Config(ctx=4096)
        assert cfg.get("ctx") == 4096
        assert cfg.get("ctx", 8192) == 4096

    def test_get_with_none(self):
        """Test get method when value is None falls back to DEFAULTS."""
        from l3m_backend.config import DEFAULTS
        cfg = Config()
        # When config value is None, get() returns from DEFAULTS
        assert cfg.get("ctx") == DEFAULTS["ctx"]
        # Explicit default is ignored when DEFAULTS has the key
        assert cfg.get("ctx", 8192) == DEFAULTS["ctx"]

    def test_get_unknown_key(self):
        """Test get with unknown key returns default."""
        cfg = Config()
        assert cfg.get("unknown_key") is None
        assert cfg.get("unknown_key", "default") == "default"


# ============================================================================
# ConfigManager Tests
# ============================================================================

class TestConfigManager:
    """Tests for ConfigManager."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create temporary config directory."""
        config_dir = tmp_path / ".l3m"
        config_dir.mkdir()
        return config_dir

    def test_load_nonexistent_config(self, temp_config_dir):
        """Test loading config when file doesn't exist (without creating)."""
        config_file = temp_config_dir / "config.json"
        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', config_file):
                mgr = ConfigManager()
                cfg = mgr.load(create_if_missing=False)
                assert cfg.ctx is None
                assert cfg.default_model is None
                assert not config_file.exists()

    def test_load_creates_default_config(self, temp_config_dir):
        """Test that load creates default config file if missing."""
        config_file = temp_config_dir / "config.json"
        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', config_file):
                mgr = ConfigManager()
                cfg = mgr.load(create_if_missing=True)
                assert cfg.ctx is None
                assert config_file.exists()
                # Verify it's valid JSON with expected structure
                content = config_file.read_text()
                assert '"default_model"' in content

    def test_save_and_load_config(self, temp_config_dir):
        """Test saving and loading config."""
        config_file = temp_config_dir / "config.json"

        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', config_file):
                # Save
                mgr = ConfigManager()
                cfg = Config(ctx=8192, default_model="test.gguf")
                mgr.save(cfg)
                assert config_file.exists()

                # Load
                mgr2 = ConfigManager()
                loaded = mgr2.load()
                assert loaded.ctx == 8192
                assert loaded.default_model == "test.gguf"

    def test_save_only_non_none_values(self, temp_config_dir):
        """Test that save only writes non-None values."""
        config_file = temp_config_dir / "config.json"

        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', config_file):
                mgr = ConfigManager()
                cfg = Config(ctx=8192)  # Only ctx is set
                mgr.save(cfg)

                # Read raw JSON
                data = json.loads(config_file.read_text())
                assert data == {"ctx": 8192}
                assert "default_model" not in data

    def test_set_value(self, temp_config_dir):
        """Test setting a config value."""
        config_file = temp_config_dir / "config.json"

        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', config_file):
                mgr = ConfigManager()
                mgr.set("ctx", 4096)

                # Verify saved
                loaded = ConfigManager().load()
                assert loaded.ctx == 4096

    def test_set_preserves_other_values(self, temp_config_dir):
        """Test that setting one value preserves other existing values."""
        config_file = temp_config_dir / "config.json"

        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', config_file):
                # First set ctx
                mgr1 = ConfigManager()
                mgr1.set("ctx", 8192)

                # New manager (simulates new CLI invocation), set gpu
                mgr2 = ConfigManager()
                mgr2._config = None  # Clear cached config
                mgr2.set("gpu", 0)

                # Another new manager, set default_model
                mgr3 = ConfigManager()
                mgr3._config = None
                mgr3.set("default_model", "test.gguf")

                # Verify all values are preserved
                final = ConfigManager().load(create_if_missing=False)
                assert final.ctx == 8192
                assert final.gpu == 0
                assert final.default_model == "test.gguf"

    def test_set_unknown_key_raises(self, temp_config_dir):
        """Test that setting unknown key raises ValueError."""
        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', temp_config_dir / "config.json"):
                mgr = ConfigManager()
                with pytest.raises(ValueError, match="Unknown config key"):
                    mgr.set("unknown_key", "value")

    def test_unset_value(self, temp_config_dir):
        """Test unsetting a config value."""
        config_file = temp_config_dir / "config.json"

        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', config_file):
                mgr = ConfigManager()
                mgr.set("ctx", 4096)
                mgr.set("gpu", 0)

                # Unset ctx
                mgr.unset("ctx")

                # Verify ctx is None but gpu is still set
                loaded = ConfigManager().load()
                assert loaded.ctx is None
                assert loaded.gpu == 0

    def test_get_value(self, temp_config_dir):
        """Test getting a config value."""
        from l3m_backend.config import DEFAULTS
        config_file = temp_config_dir / "config.json"

        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', config_file):
                mgr = ConfigManager()
                mgr.set("ctx", 4096)

                assert mgr.get("ctx") == 4096
                # Unset values fall back to DEFAULTS
                assert mgr.get("gpu") == DEFAULTS["gpu"]
                assert mgr.get("gpu", -1) == -1

    def test_list_settings(self, temp_config_dir):
        """Test listing non-default settings."""
        config_file = temp_config_dir / "config.json"

        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', config_file):
                mgr = ConfigManager()
                mgr.set("ctx", 4096)  # Different from default (8192)
                mgr.set("verbose", True)

                settings = mgr.list_settings()
                assert settings == {"ctx": 4096, "verbose": True}

    def test_list_settings_empty(self, temp_config_dir):
        """Test listing settings when empty."""
        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', temp_config_dir / "config.json"):
                mgr = ConfigManager()
                settings = mgr.list_settings()
                assert settings == {}

    def test_reset(self, temp_config_dir):
        """Test resetting config to defaults."""
        config_file = temp_config_dir / "config.json"

        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', config_file):
                mgr = ConfigManager()
                mgr.set("ctx", 4096)
                assert config_file.exists()

                mgr.reset()
                assert not config_file.exists()

                # Load returns defaults
                loaded = mgr.load()
                assert loaded.ctx is None

    def test_load_invalid_json(self, temp_config_dir):
        """Test loading invalid JSON returns defaults."""
        config_file = temp_config_dir / "config.json"
        config_file.write_text("not valid json")

        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', config_file):
                mgr = ConfigManager()
                cfg = mgr.load()
                # Should return defaults, not crash
                assert cfg.ctx is None

    def test_load_invalid_schema(self, temp_config_dir):
        """Test loading invalid schema returns defaults."""
        config_file = temp_config_dir / "config.json"
        config_file.write_text('{"ctx": "not a number"}')

        with patch.object(ConfigManager, 'CONFIG_DIR', temp_config_dir):
            with patch.object(ConfigManager, 'CONFIG_FILE', config_file):
                mgr = ConfigManager()
                cfg = mgr.load()
                # Should return defaults, not crash
                assert cfg.ctx is None


# ============================================================================
# Singleton Tests
# ============================================================================

class TestSingleton:
    """Tests for singleton functions."""

    def test_get_config_manager_returns_same_instance(self, tmp_path):
        """Test that get_config_manager returns singleton."""
        import l3m_backend.config.config as config_module

        # Reset singleton
        config_module._manager = None

        with patch.object(ConfigManager, 'CONFIG_DIR', tmp_path):
            with patch.object(ConfigManager, 'CONFIG_FILE', tmp_path / "config.json"):
                mgr1 = get_config_manager()
                mgr2 = get_config_manager()
                assert mgr1 is mgr2

        # Clean up singleton
        config_module._manager = None

    def test_get_config_returns_config(self, tmp_path):
        """Test that get_config returns Config instance."""
        import l3m_backend.config.config as config_module

        # Reset singleton
        config_module._manager = None

        with patch.object(ConfigManager, 'CONFIG_DIR', tmp_path):
            with patch.object(ConfigManager, 'CONFIG_FILE', tmp_path / "config.json"):
                cfg = get_config()
                assert isinstance(cfg, Config)

        # Clean up singleton
        config_module._manager = None


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
