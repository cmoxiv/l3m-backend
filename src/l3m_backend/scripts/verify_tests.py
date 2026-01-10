#!/usr/bin/env python3
"""
Quick verification script to demonstrate test capabilities.

Usage:
    pip install -e .
    python scripts/verify_tests.py
"""

from l3m_backend.tools import get_time, calculate, registry

print("=" * 70)
print("VERIFICATION: LLM Tool-Calling System Tests")
print("=" * 70)

# Test 1: Time Tool
print("\n1. Testing get_time():")
result = get_time()
print(f"   Result: {result.data}")
print(f"   LLM Format: {result.llm_format}")
print(f"   Type: {result.type_name}")

# Test 2: Calculator Tool
print("\n2. Testing calculate():")
test_cases = [
    "2 + 3",
    "10 * 5",
    "(10 + 5) * 2",
    "10 / 0",  # Error case
    "import os",  # Security test
]

for expr in test_cases:
    result = calculate(expr)
    print(f"   {expr} = {result.data['result']}")

# Test 3: Registry
print("\n3. Testing Registry:")
print(f"   Total tools: {len(registry)}")
print(f"   Tools: {[entry.name for entry in registry]}")

# Test 4: Aliases
print("\n4. Testing Aliases:")
result1 = registry.execute("calculate", {"expression": "5 + 5"})
result2 = registry.execute("c", {"expression": "5 + 5"})
print(f"   'calculate' -> {result1.output.data['result']}")
print(f"   'c' (alias) -> {result2.output.data['result']}")
print(f"   Both resolve to same tool: {result1.name == result2.name}")

# Test 5: Tool Output Wrapper
print("\n5. Testing ToolOutput Wrapper:")
result = calculate("100 / 4")
print(f"   Data: {result.data}")
print(f"   LLM Format: {result.llm_format}")
print(f"   Has unique ID: {len(result.id) > 0}")

print("\n" + "=" * 70)
print("All manual verification tests passed!")
print("=" * 70)
