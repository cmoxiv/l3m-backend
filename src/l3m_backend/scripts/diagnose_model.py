#!/usr/bin/env python3
"""Basic model loading test.

Usage:
    pip install -e '.[llm]'
    python scripts/diagnose_model.py
"""

from llama_cpp import Llama

print("=== Loading model with 32K context ===")
m = Llama("./model.gguf", n_ctx=32768, n_gpu_layers=-1, verbose=True)
print("Model loaded!")

print("\n=== Chat test ===")
r = m.create_chat_completion([{"role": "user", "content": "Say hello"}])
print(f"Response: {r['choices'][0]['message']['content']}")
