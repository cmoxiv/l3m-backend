"""
Calculator tool implementation.
"""

from typing import Any

from pydantic import Field

from l3m_backend.core import tool_output
from l3m_backend.tools._registry import registry


@registry.register(aliases=["calc", "c"])
@tool_output(llm_format=lambda x: f"Result: {x['result']}")
def calculate(
    expression: str = Field(description="Math expression to evaluate"),
) -> dict[str, Any]:
    """Evaluate a mathematical expression.

    Safely evaluates basic arithmetic expressions using Python's eval.
    Only allows numbers and basic operators (+, -, *, /, parentheses).

    Args:
        expression: Mathematical expression string (e.g., "2 + 2", "10 * (5 + 3)").

    Returns:
        Dictionary with keys:
            - expression: The original expression
            - result: Evaluated result or error message

    Example:
        >>> calculate("2 + 2")
        {"expression": "2 + 2", "result": 4}

        >>> calculate("10 * (5 + 3)")
        {"expression": "10 * (5 + 3)", "result": 80}

    Note:
        Only basic arithmetic is supported. Advanced functions (sin, cos, etc.)
        will be rejected as invalid.
    """
    try:
        # Safe eval for basic math
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            result = eval(expression)
        else:
            result = "Invalid expression"
    except Exception as e:
        result = f"Error: {e}"
    return {"expression": expression, "result": result}
