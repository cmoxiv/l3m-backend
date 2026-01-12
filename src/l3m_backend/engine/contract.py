"""
Tool contract template for LLM prompting.
"""

from pathlib import Path

DEFAULT_CONTRACT_PATH = Path.home() / ".l3m" / "tool-contract.txt"

DEFAULT_TOOL_CONTRACT = """
You are a tool-calling assistant. Use the tools provided to get real information.

TOOLS:
{registry_json}

OUTPUT FORMAT:

To call a tool, respond with JSON:
{{"type": "tool_call", "name": "TOOL_NAME", "arguments": {{"arg": "value"}}}}

To respond to the user, just write plain text (no JSON).

TOOL RESULTS:
After you call a tool, you will receive the result.
Use that result to respond to the user in plain text.

IMPORTANT:
- For required arguments: use values from context if available, otherwise ASK the user
- For optional arguments: use values from context if available, otherwise use defaults
- Only use JSON for tool calls, respond in plain text otherwise
""".strip()


def load_contract_template(path: Path | None = None) -> str:
    """Load contract template from file, creating default if needed.

    Args:
        path: Path to contract file. Defaults to ~/.l3m/tool-contract.txt

    Returns:
        Contract template string with placeholders.
    """
    contract_path = path or DEFAULT_CONTRACT_PATH

    # Create default file if it doesn't exist
    if not contract_path.exists():
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(DEFAULT_TOOL_CONTRACT)

    return contract_path.read_text().strip()


# For backwards compatibility
def get_tool_contract_template() -> str:
    """Get the tool contract template (loads from file)."""
    return load_contract_template()


MINIMAL_CONTRACT_TEMPLATE = """
You are a helpful assistant with tools.

TOOLS:
{tool_list}

RULES:
1. To call a tool: {{"type": "tool_call", "name": "NAME", "arguments": {{...}}}}
2. Wait for the tool result, then respond in plain text (no JSON)
3. For arguments: use context values if available, otherwise use defaults, otherwise ASK the user
4. Never call the same tool twice in a row
""".strip()

# TOOL_CONTRACT_TEMPLATE is now loaded from ~/.l3m/tool-contract.txt
# Use get_tool_contract_template() or load_contract_template() to get the template
TOOL_CONTRACT_TEMPLATE = DEFAULT_TOOL_CONTRACT  # Fallback for imports
