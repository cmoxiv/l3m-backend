"""
Obfuscated test tools for tool calling validation.

These tools use nonsense names to avoid training data contamination
when testing LLM tool calling capabilities.
"""

from l3m_backend.tools.test_tools.core import calculate_zorbix, get_flumbuster

__all__ = ["get_flumbuster", "calculate_zorbix"]
