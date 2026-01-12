#!/usr/bin/env python3
"""
E2E Test Script for Hermes 8B with Test Tools.

Tests that the Hermes model can:
1. Understand the tool contract
2. Call tools with correct arguments
3. Use tool results to formulate responses

Usage:
    source ~/.venvs/l3m/bin/activate
    python scripts/e2e_test_hermes.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Model configuration
MODEL_NAME = "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
MODEL_DIR = Path.home() / ".l3m" / "models"


@dataclass
class TestCase:
    """A single E2E test case."""
    name: str
    prompt: str
    expect_tool: str | None  # Tool name we expect to be called, or None for no tool
    expect_args: dict | None  # Expected arguments (partial match)
    expect_in_response: list[str] | None  # Strings expected in final response


# Test cases
TEST_CASES = [
    # Basic tool call - get_flumbuster (weather)
    TestCase(
        name="flumbuster_basic",
        prompt="What is the flumbuster reading for cairo?",
        expect_tool="get_flumbuster",
        expect_args={"flumb": "cairo"},
        expect_in_response=["cairo", "°"],
    ),
    # Tool call with optional arg
    TestCase(
        name="flumbuster_with_uster",
        prompt="Get the flumbuster for london in imperial uster",
        expect_tool="get_flumbuster",
        expect_args={"flumb": "london", "uster": "imperial"},
        expect_in_response=["london", "°F"],  # Model may use °F instead of fahrenheit
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
        expect_args={"zorb": 6, "bix": 8, "mode": "mul"},
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
    from l3m_backend.engine.toolcall import get_handler

    handler = get_handler()
    tool_called = None
    tool_args = None
    error = None

    # Clear history for clean test
    engine.clear()

    start = time.perf_counter()
    try:
        # Collect response (using streaming internally)
        response = ""
        for chunk in engine.chat(test.prompt, stream=True):
            response += chunk

        # Check history for tool calls
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

    duration_ms = (time.perf_counter() - start) * 1000

    # Evaluate pass/fail
    passed = True

    if test.expect_tool:
        if tool_called != test.expect_tool:
            passed = False
            error = f"Expected tool {test.expect_tool}, got {tool_called}"
        elif test.expect_args and tool_args:
            for key, val in test.expect_args.items():
                if key not in tool_args or tool_args[key] != val:
                    passed = False
                    error = f"Expected arg {key}={val}, got {tool_args.get(key)}"
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
    )


def main():
    """Run E2E tests."""
    model_path = MODEL_DIR / MODEL_NAME

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print(f"\nDownload with:")
        print(f"  l3m-download NousResearch/Hermes-3-Llama-3.1-8B-GGUF {MODEL_NAME}")
        sys.exit(1)

    print(f"E2E Test: Hermes 8B with Test Tools")
    print(f"=" * 50)
    print(f"Model: {MODEL_NAME}")
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
        minimal_contract=True,  # Test improved minimal contract
        debug=False,
    )
    print(f"Model loaded. Tools: {[t for t in registry._tools.keys()]}")
    print()

    # Run tests
    results: list[TestResult] = []
    for i, test in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {test.name}...", end=" ", flush=True)
        result = run_test(engine, test)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"{status} ({result.duration_ms:.0f}ms)")

        if not result.passed:
            print(f"       Error: {result.error}")
            print(f"       Response: {result.response[:100]}...")

    # Summary
    print()
    print("=" * 50)
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"Results: {passed}/{len(results)} passed, {failed} failed")

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
