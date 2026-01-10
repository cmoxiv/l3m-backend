"""
Engine module for the l3m_backend package.

Provides the ChatEngine for LLM interaction with tool calling.
"""

from l3m_backend.engine.chat import ChatEngine
from l3m_backend.engine.contract import TOOL_CONTRACT_TEMPLATE

__all__ = [
    "ChatEngine",
    "TOOL_CONTRACT_TEMPLATE",
]
