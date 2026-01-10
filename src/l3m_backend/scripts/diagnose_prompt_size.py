#!/usr/bin/env python3
"""Check prompt size.

Usage:
    pip install -e '.[llm]'
    python scripts/diagnose_prompt_size.py
"""

from l3m_backend.engine import ChatEngine
from l3m_backend.tools import registry

engine = ChatEngine(
    model_path="./model.gguf",
    registry=registry,
    n_ctx=2048,
    verbose=False,
)

system_msg = engine._build_system_message()
print(f"System message length: {len(system_msg['content'])} chars")
print("\n--- System message preview ---")
print(system_msg['content'][:500])
print("...")
