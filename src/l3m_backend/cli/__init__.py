"""
CLI module for the l3m_backend package.

Provides command-line interfaces for chat REPL and tool inspection.
"""

from l3m_backend.cli._simple_repl import print_tools
from l3m_backend.cli._simple_repl import repl as simple_repl

# Try to import feature-rich REPL, fallback to simple
try:
    from l3m_backend.cli._repl import repl
except ImportError:
    repl = simple_repl

__all__ = [
    "repl",
    "simple_repl",
    "print_tools",
]
