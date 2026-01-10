#!/usr/bin/env python3
"""
E2E Test Script for Native Tool Calling (llama-cpp-python).

Tests llama-cpp-python's native function calling via the tools= parameter.

KNOWN LIMITATIONS:
- chatml-function-calling format does NOT work with Hermes models
  - Tool calls work, but model ignores tool results
  - Hermes uses <tool_call>/<tool_response> XML format, not OpenAI format
- Use contract-based approach for Hermes (see e2e_test_hermes.py)

This script is for testing models specifically trained on OpenAI-style
function calling (e.g., functionary models).

Usage:
    source ~/.venvs/l3m/bin/activate
    python scripts/e2e_test_native.py
    python scripts/e2e_test_native.py --model <functionary-model>.gguf
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Model configuration
DEFAULT_MODEL = "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
MODEL_DIR = Path.home() / ".l3m" / "models"


@dataclass
class TestCase:
    """A single E2E test case."""
    name: str
    prompt: str
    expect_tool: str | None  # Tool name we expect to be called, or None for no tool
    expect_args: dict | None  # Expected arguments (partial match)
    expect_in_response: list[str] | None  # Strings expected in final response


# Test cases - same as contract-based for comparison
TEST_CASES = [
    # Basic tool call - get_flumbuster (weather)
    TestCase(
        name="flumbuster_basic",
        prompt="What is the flumbuster reading for cairo?",
        expect_tool="get_flumbuster",
        expect_args={"flumb": "cairo"},
        expect_in_response=["cairo"],
    ),
    # Tool call with optional arg
    TestCase(
        name="flumbuster_with_uster",
        prompt="Get the flumbuster for london in imperial uster",
        expect_tool="get_flumbuster",
        expect_args={"flumb": "london"},
        expect_in_response=["london"],
    ),
    # Math tool - calculate_zorbix
    TestCase(
        name="zorbix_add",
        prompt="Calculate the zorbix of 15 and 7",
        expect_tool="calculate_zorbix",
        expect_args={"zorb": 15, "bix": 7},
        expect_in_response=["22"],
    ),
    # Math with mode
    TestCase(
        name="zorbix_multiply",
        prompt="What is the zorbix of 6 and 8 in mul mode?",
        expect_tool="calculate_zorbix",
        expect_args={"zorb": 6, "bix": 8},
        expect_in_response=["48"],
    ),
    # No tool needed
    TestCase(
        name="no_tool_greeting",
        prompt="Hello, how are you?",
        expect_tool=None,
        expect_args=None,
        expect_in_response=None,
    ),
]


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    tool_called: str | None
    tool_args: dict | None
    response: str
    error: str | None
    duration_ms: float
    used_native: bool  # True if native tool_calls were detected


def create_test_registry():
    """Create a registry with only test tools."""
    from l3m_backend.core import ToolRegistry
    from l3m_backend.tools.test_tools import get_flumbuster, calculate_zorbix

    registry = ToolRegistry()
    registry.register(get_flumbuster)
    registry.register(calculate_zorbix)
    return registry


def run_test(engine, test: TestCase) -> TestResult:
    """Run a single test case."""
    tool_called = None
    tool_args = None
    error = None
    used_native = False

    # Clear history for clean test
    engine.clear()

    start = time.perf_counter()
    try:
        # Use non-streaming mode - native tool calling doesn't support streaming
        # with tool_choice="auto" in llama-cpp-python
        response = engine.chat(test.prompt)

        # Check history for native tool calls (tool_calls field)
        for msg in engine.history:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Native tool call detected
                used_native = True
                tc = msg["tool_calls"][0]
                tool_called = tc["function"]["name"]
                import json
                args_raw = tc["function"].get("arguments", "{}")
                if isinstance(args_raw, str):
                    tool_args = json.loads(args_raw)
                else:
                    tool_args = args_raw
                break

        # Fallback: check for contract-based tool calls in content
        if not used_native:
            from l3m_backend.engine.toolcall import get_handler
            handler = get_handler()
            for msg in engine.history:
                content = msg.get("content", "")
                parsed = handler.parse(content)
                if parsed:
                    tool_called = parsed.name
                    tool_args = parsed.arguments
                    break

    except Exception as e:
        response = ""
        error = str(e)
        import traceback
        traceback.print_exc()

    duration_ms = (time.perf_counter() - start) * 1000

    # Evaluate pass/fail
    passed = True

    if test.expect_tool:
        if tool_called != test.expect_tool:
            passed = False
            error = f"Expected tool {test.expect_tool}, got {tool_called}"
        elif test.expect_args and tool_args:
            for key, val in test.expect_args.items():
                actual = tool_args.get(key)
                # Normalize for comparison (case-insensitive for strings)
                if isinstance(val, str) and isinstance(actual, str):
                    if val.lower() != actual.lower():
                        passed = False
                        error = f"Expected arg {key}={val}, got {actual}"
                        break
                elif actual != val:
                    passed = False
                    error = f"Expected arg {key}={val}, got {actual}"
                    break
    elif test.expect_tool is None and tool_called:
        # Expected no tool but one was called
        passed = False
        error = f"Expected no tool call, but {tool_called} was called"

    if test.expect_in_response and passed:
        response_lower = response.lower()
        for expected in test.expect_in_response:
            if expected.lower() not in response_lower:
                passed = False
                error = f"Expected '{expected}' in response"
                break

    return TestResult(
        name=test.name,
        passed=passed,
        tool_called=tool_called,
        tool_args=tool_args,
        response=response[:200] + "..." if len(response) > 200 else response,
        error=error,
        duration_ms=duration_ms,
        used_native=used_native,
    )


def main():
    """Run E2E tests."""
    parser = argparse.ArgumentParser(description="E2E test for native tool calling")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model filename")
    parser.add_argument("--chat-format", default="chatml-function-calling",
                        help="Chat format for native tools")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-contract", action="store_true",
                        help="Disable contract prompt (pure native mode)")
    args = parser.parse_args()

    model_path = MODEL_DIR / args.model

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print(f"\nDownload with:")
        print(f"  l3m-download NousResearch/Hermes-3-Llama-3.1-8B-GGUF {args.model}")
        sys.exit(1)

    print(f"E2E Test: Native Tool Calling")
    print(f"=" * 50)
    print(f"Model: {args.model}")
    print(f"Chat format: {args.chat_format}")
    print(f"Method: Native (llama-cpp-python tools=)")
    print()

    # Create registry and engine
    print("Loading model...")
    registry = create_test_registry()

    from l3m_backend.engine.chat import ChatEngine

    engine = ChatEngine(
        model_path=str(model_path),
        registry=registry,
        n_ctx=8192,  # Smaller context for faster tests
        verbose=False,
        use_native_tools=True,  # Enable native tool calling
        chat_format=args.chat_format,  # Required for native tools
        minimal_contract=True,  # Still include contract as fallback context
        debug=args.debug,
    )
    print(f"Model loaded. Tools: {list(registry._tools.keys())}")
    print()

    # Run tests
    results: list[TestResult] = []
    native_count = 0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {test.name}...", end=" ", flush=True)
        result = run_test(engine, test)
        results.append(result)

        if result.used_native:
            native_count += 1

        status = "PASS" if result.passed else "FAIL"
        native_indicator = " (native)" if result.used_native else " (contract)"
        print(f"{status}{native_indicator} ({result.duration_ms:.0f}ms)")

        if not result.passed:
            print(f"       Error: {result.error}")
            print(f"       Response: {result.response}")
            if args.debug:
                print(f"       History:")
                for msg in engine.history:
                    role = msg.get("role", "?")
                    content = str(msg.get("content", ""))[:100]
                    tc = msg.get("tool_calls", [])
                    tcid = msg.get("tool_call_id", "")
                    extra = f" tool_calls={tc}" if tc else ""
                    extra += f" tool_call_id={tcid}" if tcid else ""
                    print(f"         [{role}] {content}{extra}")

    # Summary
    print()
    print("=" * 50)
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"Results: {passed}/{len(results)} passed, {failed} failed")
    print(f"Native tool calls: {native_count}/{len(results)}")

    total_time = sum(r.duration_ms for r in results)
    print(f"Total time: {total_time/1000:.1f}s")

    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
