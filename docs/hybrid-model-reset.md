# Why Hybrid Models Need `reset()` Between Calls

## Analogy: The Etch A Sketch vs. The Camera

**Transformer models** are like a **camera**: each photo is independent. You can take a picture, then take another completely unrelated picture. No state carries over.

**Mamba/SSM hybrid models** (like granite-hybrid) are like an **Etch A Sketch**: they maintain internal state as they draw. After you finish one drawing, if you want to start fresh, you must **shake it to reset** — otherwise the new drawing starts from where the old one left off, corrupting the output.

## Diagram

```
TRANSFORMER MODEL (stateless):
┌─────────────────────────────────────────────────────┐
│  Call 1: "What time?"     Call 2: "Hello"           │
│         ↓                        ↓                  │
│    [KV Cache]               [KV Cache]              │
│    (cleared)                (cleared)               │
│         ↓                        ↓                  │
│    Response 1               Response 2              │
│         ✓                        ✓                  │
└─────────────────────────────────────────────────────┘

MAMBA/SSM HYBRID (stateful):
┌─────────────────────────────────────────────────────┐
│  Call 1: "What time?"                               │
│         ↓                                           │
│    [SSM State] ← state accumulates                  │
│         ↓                                           │
│    Response 1  ✓                                    │
│                                                     │
│  Call 2: "Hello" (NO reset)                         │
│         ↓                                           │
│    [SSM State] ← CORRUPTED (old state + new input)  │
│         ↓                                           │
│    llama_decode returned -1                         │
└─────────────────────────────────────────────────────┘

WITH reset():
┌─────────────────────────────────────────────────────┐
│  Call 1 → Response 1  ✓                             │
│         ↓                                           │
│    reset() ← "shake the Etch A Sketch"              │
│         ↓                                           │
│  Call 2 → Response 2  ✓                             │
└─────────────────────────────────────────────────────┘
```

## Walkthrough

1. **First call** — Model processes messages, SSM layers build up internal state as tokens flow through, generates response. Works fine.

2. **Second call (without reset)** — Model tries to process new messages, but SSM layers still hold state from previous generation. The state vectors are the wrong size/shape for a fresh sequence. `llama_decode` fails with error `-1`.

3. **With `reset()`** — Clears the SSM state buffers back to zeros. Now the model can process a fresh sequence as if it just loaded.

```python
# l3m_backend/engine/chat_engine.py - ChatEngine.chat()
for round_num in range(max_tool_rounds):
    messages = self._build_messages(pending)

    self.llm.reset()  # Clear SSM state before each generation

    response = self.llm.create_chat_completion(messages=messages)
```

## Gotcha

**Misconception**: "KV cache is the only model state that matters"

With transformers, the KV cache is typically cleared/managed automatically between calls. But **Mamba/SSM models have additional recurrent state** (the "state-space" in State Space Models) that persists and isn't automatically cleared by llama-cpp-python.

This is why `reset()` is needed specifically for hybrid architectures like `granitehybrid`, but not for pure transformer models like Llama or Mistral.

## Why LM Studio Doesn't Have This Problem

If you've used hybrid models in LM Studio without issues, here's why:

```
LM STUDIO SERVER MODE:
┌─────────────────────────────────────────────────────────────┐
│  Request 1 ──→ [Fresh Context] ──→ Response 1              │
│                      ↓                                      │
│               (context destroyed or reset)                  │
│                      ↓                                      │
│  Request 2 ──→ [Fresh Context] ──→ Response 2              │
└─────────────────────────────────────────────────────────────┘

DIRECT llama-cpp-python (without reset):
┌─────────────────────────────────────────────────────────────┐
│  Request 1 ──→ [Context] ──→ Response 1                    │
│                    │                                        │
│               (SSM state persists)                          │
│                    ↓                                        │
│  Request 2 ──→ [Corrupted Context] ──→ llama_decode = -1   │
└─────────────────────────────────────────────────────────────┘
```

**LM Studio handles this automatically because:**

1. **Server architecture** - Runs llama.cpp's server mode, which manages state per-request
2. **Automatic reset** - The server resets model state between requests
3. **Application-level history** - Conversation tracked separately from model state, prompt rebuilt fresh each time

**Our solution mirrors LM Studio's approach:**

```python
self.llm.reset()  # Fresh model state (like LM Studio does internally)
response = self.llm.create_chat_completion(
    messages=self._build_messages(pending)  # Rebuild full context
)
```

The difference: LM Studio abstracts this away; with llama-cpp-python directly, we manage it ourselves.
