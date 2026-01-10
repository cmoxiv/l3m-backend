"""
Tests for user-defined tools functionality.
"""

import pytest
from pathlib import Path

from l3m_backend.tools.loader import (
    discover_user_tools,
    load_user_tool,
    load_user_tools,
)
from l3m_backend.tools.generator import (
    generate_tool_name,
    generate_tool,
)


class TestToolNameGeneration:
    """Tests for tool name validation/generation."""

    def test_simple_name(self):
        assert generate_tool_name("my_tool") == "my_tool"

    def test_hyphenated_name(self):
        assert generate_tool_name("my-tool") == "my_tool"

    def test_name_with_spaces(self):
        assert generate_tool_name("my tool") == "my_tool"

    def test_name_starting_with_number(self):
        assert generate_tool_name("123tool") == "_123tool"

    def test_uppercase_name(self):
        assert generate_tool_name("MyTool") == "mytool"

    def test_special_characters(self):
        assert generate_tool_name("my@tool!") == "mytool"

    def test_empty_name(self):
        assert generate_tool_name("") == ""

    def test_only_special_chars(self):
        assert generate_tool_name("@#$%") == ""


class TestToolDiscovery:
    """Tests for tool discovery."""

    def test_discover_nonexistent_dir(self, tmp_path):
        """Test discovery with non-existent directory."""
        tools = discover_user_tools(tmp_path / "nonexistent")
        assert tools == []

    def test_discover_empty_dir(self, tmp_path):
        """Test discovery with empty directory."""
        tools = discover_user_tools(tmp_path)
        assert tools == []

    def test_discover_valid_tool(self, tmp_path):
        """Test discovery of valid tool."""
        tool_dir = tmp_path / "my_tool"
        tool_dir.mkdir()
        (tool_dir / "__init__.py").write_text("# tool")

        tools = discover_user_tools(tmp_path)
        assert len(tools) == 1
        assert tools[0].parent.name == "my_tool"

    def test_discover_multiple_tools(self, tmp_path):
        """Test discovery of multiple tools."""
        for name in ["tool_a", "tool_b", "tool_c"]:
            tool_dir = tmp_path / name
            tool_dir.mkdir()
            (tool_dir / "__init__.py").write_text("# tool")

        tools = discover_user_tools(tmp_path)
        assert len(tools) == 3
        # Should be sorted
        assert [t.parent.name for t in tools] == ["tool_a", "tool_b", "tool_c"]

    def test_skip_hidden_dirs(self, tmp_path):
        """Test that hidden directories are skipped."""
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "__init__.py").write_text("# tool")

        tools = discover_user_tools(tmp_path)
        assert tools == []

    def test_skip_private_dirs(self, tmp_path):
        """Test that private directories (starting with _) are skipped."""
        private_dir = tmp_path / "_private"
        private_dir.mkdir()
        (private_dir / "__init__.py").write_text("# tool")

        tools = discover_user_tools(tmp_path)
        assert tools == []

    def test_skip_dirs_without_init(self, tmp_path):
        """Test that directories without __init__.py are skipped."""
        tool_dir = tmp_path / "incomplete_tool"
        tool_dir.mkdir()
        # No __init__.py

        tools = discover_user_tools(tmp_path)
        assert tools == []

    def test_skip_files(self, tmp_path):
        """Test that files at top level are ignored."""
        (tmp_path / "not_a_tool.py").write_text("# not a tool")

        tools = discover_user_tools(tmp_path)
        assert tools == []


class TestToolLoading:
    """Tests for tool loading."""

    def test_load_valid_tool(self, tmp_path):
        """Test loading a valid tool."""
        tool_dir = tmp_path / "valid_tool"
        tool_dir.mkdir()
        (tool_dir / "__init__.py").write_text(
            '''
from l3m_backend.tools import registry
from l3m_backend.core import tool_output

@registry.register()
@tool_output(llm_format="{result}")
def valid_tool_func() -> dict:
    return {"result": "ok"}
'''
        )

        name, success, error = load_user_tool(tool_dir / "__init__.py")
        assert success is True
        assert name == "valid_tool"
        assert error == ""

    def test_load_syntax_error(self, tmp_path):
        """Test loading tool with syntax error."""
        tool_dir = tmp_path / "broken_tool"
        tool_dir.mkdir()
        (tool_dir / "__init__.py").write_text("def broken( # syntax error")

        name, success, error = load_user_tool(tool_dir / "__init__.py")
        assert success is False
        assert "Syntax error" in error
        assert name == "broken_tool"

    def test_load_import_error(self, tmp_path):
        """Test loading tool with import error."""
        tool_dir = tmp_path / "import_error_tool"
        tool_dir.mkdir()
        (tool_dir / "__init__.py").write_text("import nonexistent_module_xyz")

        name, success, error = load_user_tool(tool_dir / "__init__.py")
        assert success is False
        assert "Import error" in error

    def test_load_runtime_error(self, tmp_path):
        """Test loading tool with runtime error during import."""
        tool_dir = tmp_path / "runtime_error_tool"
        tool_dir.mkdir()
        (tool_dir / "__init__.py").write_text("x = 1 / 0  # ZeroDivisionError")

        name, success, error = load_user_tool(tool_dir / "__init__.py")
        assert success is False
        assert "Error" in error

    def test_load_user_tools_empty(self, tmp_path):
        """Test load_user_tools with empty directory."""
        count = load_user_tools(tmp_path)
        assert count == 0

    def test_load_user_tools_mixed(self, tmp_path):
        """Test load_user_tools with mix of valid and invalid tools."""
        # Valid tool
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()
        (valid_dir / "__init__.py").write_text("# valid tool")

        # Invalid tool
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        (invalid_dir / "__init__.py").write_text("import nonexistent_xyz")

        count = load_user_tools(tmp_path)
        # Only the valid one should load
        assert count == 1


class TestToolGenerator:
    """Tests for tool generator."""

    def test_generate_basic_tool(self, tmp_path):
        """Test generating a basic tool."""
        success, message, path = generate_tool("my_tool", output_dir=tmp_path)

        assert success is True
        assert path is not None
        assert (path / "__init__.py").exists()

        content = (path / "__init__.py").read_text()
        assert "def my_tool" in content
        assert "@registry.register" in content
        assert "@tool_output" in content

    def test_generate_wrapper_tool(self, tmp_path):
        """Test generating a wrapper tool."""
        success, message, path = generate_tool(
            "git_status",
            wrap_command="git status",
            output_dir=tmp_path,
        )

        assert success is True
        content = (path / "__init__.py").read_text()
        assert "git status" in content
        assert "subprocess.run" in content

    def test_generate_refuses_overwrite(self, tmp_path):
        """Test generator refuses to overwrite existing tool."""
        # Create first
        generate_tool("my_tool", output_dir=tmp_path)

        # Try to create again
        success, message, path = generate_tool("my_tool", output_dir=tmp_path)

        assert success is False
        assert "already exists" in message

    def test_generate_force_overwrite(self, tmp_path):
        """Test generator can force overwrite."""
        generate_tool("my_tool", output_dir=tmp_path)
        success, message, path = generate_tool(
            "my_tool",
            output_dir=tmp_path,
            force=True,
        )

        assert success is True

    def test_generate_invalid_name(self, tmp_path):
        """Test generator with invalid name."""
        success, message, path = generate_tool("@#$%", output_dir=tmp_path)

        assert success is False
        assert "Invalid tool name" in message

    def test_generate_creates_directory(self, tmp_path):
        """Test generator creates parent directories."""
        nested = tmp_path / "nested" / "path"
        success, message, path = generate_tool("my_tool", output_dir=nested)

        assert success is True
        assert path.exists()

    def test_generate_hyphenated_name(self, tmp_path):
        """Test generator handles hyphenated names."""
        success, message, path = generate_tool("my-tool", output_dir=tmp_path)

        assert success is True
        assert path.name == "my_tool"  # Converted to underscore

    def test_generated_tool_is_loadable(self, tmp_path):
        """Test that generated tool can be loaded."""
        success, message, path = generate_tool("loadable_tool", output_dir=tmp_path)
        assert success is True

        name, load_success, error = load_user_tool(path / "__init__.py")
        assert load_success is True, f"Failed to load: {error}"

    def test_generated_wrapper_is_loadable(self, tmp_path):
        """Test that generated wrapper tool can be loaded."""
        success, message, path = generate_tool(
            "echo_wrapper",
            wrap_command="echo",
            output_dir=tmp_path,
        )
        assert success is True

        name, load_success, error = load_user_tool(path / "__init__.py")
        assert load_success is True, f"Failed to load: {error}"
