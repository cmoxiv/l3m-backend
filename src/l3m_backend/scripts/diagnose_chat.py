#!/usr/bin/env python3
"""Quick test for ChatEngine.

Usage:
    pip install -e '.[llm]'
    python scripts/diagnose_chat.py
"""

from l3m_backend.engine import ChatEngine
from l3m_backend.tools import registry

engine = ChatEngine(
    model_path="./model.gguf",
    registry=registry,
    verbose=False,
)

print("=== Test 1: Time ===")
response = engine.chat("What is the time now?")
print(f"Response: '{response}'")

print("\n=== Test 2: Weather (no location) ===")
response = engine.chat("What's the weather?")
print(f"Response: '{response}'")

print("\n=== Test 3: Weather (with location) ===")
response = engine.chat("Weather in Tokyo")
print(f"Response: '{response}'")
