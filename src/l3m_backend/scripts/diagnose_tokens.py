#!/usr/bin/env python3
"""Count tokens in system prompt.

Usage:
    pip install -e '.[llm]'
    python scripts/diagnose_tokens.py
"""

from l3m_backend.engine import ChatEngine
from l3m_backend.tools import registry

# Create engine
engine = ChatEngine(
    model_path="./model.gguf",
    registry=registry,
    verbose=False,
)

system_msg = engine._build_system_message()
print(f"System message: {len(system_msg['content'])} chars")

# Tokenize the system message
tokens = engine.llm.tokenize(system_msg['content'].encode())
print(f"System message: {len(tokens)} tokens")

# Build messages
messages = engine._build_messages([{"role": "user", "content": "What is the time?"}])
print(f"\nNumber of messages: {len(messages)}")

total_tokens = 0
for i, msg in enumerate(messages):
    content = msg.get('content', '')
    toks = engine.llm.tokenize(content.encode())
    total_tokens += len(toks)
    print(f"  [{i}] {msg['role']}: {len(content)} chars, {len(toks)} tokens")

print(f"\nTotal content tokens: {total_tokens}")
print(f"Context size: {engine.llm.n_ctx()}")

# Try a direct chat
print("\n=== Direct LLM call ===")
r = engine.llm.create_chat_completion(messages)
print(f"Response: {r['choices'][0]['message']['content']}")
