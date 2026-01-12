"""Planning tool that uses the LLM to create execution plans.

This tool analyzes user requests and creates a plan of tools to use,
without updating the main conversation history.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from l3m_backend.core import tool_output
from l3m_backend.engine.context import get_current_engine
from l3m_backend.tools._registry import registry


def _get_available_tools() -> str:
    """Get a summary of available tools for planning."""
    tools = []
    for entry in registry:
        # Skip the plan tool itself to avoid recursion
        if entry.name == "plan":
            continue
        desc = entry.description or "No description"
        aliases = f" (aliases: {', '.join(entry.aliases)})" if entry.aliases else ""
        tools.append(f"- {entry.name}{aliases}: {desc}")
    return "\n".join(tools)


def _create_plan_prompt(task: str, tools_summary: str) -> str:
    """Create the prompt for plan generation."""
    return f"""Analyze this task and create a step-by-step plan using available tools.

TASK: {task}

AVAILABLE TOOLS:
{tools_summary}

Create a plan that:
1. Lists the tools to use in order
2. Explains what each step accomplishes
3. Notes any dependencies between steps

Respond with a clear, numbered plan. Be concise."""


@registry.register(aliases=["planning", "plan_task"])
@tool_output(llm_format=lambda x: x.get("plan", x.get("error", "No plan generated")))
def plan(
    task: str = Field(description="The task or question to create a plan for"),
) -> dict[str, Any]:
    """Create an execution plan for a complex task.

    Analyzes the task and determines which tools to use and in what order.
    Uses the LLM to generate the plan without affecting conversation history.

    Use this tool when:
    - A task requires multiple steps or tools
    - You need to think through a complex problem
    - You want to explain your approach before executing

    Args:
        task: The task or question to plan for.

    Returns:
        Dictionary with the plan or error message.
    """
    # Get the current engine from context
    engine = get_current_engine()
    if engine is None:
        return {"error": "Planning tool requires engine context (not available outside tool execution)"}

    # Get available tools summary
    tools_summary = _get_available_tools()

    # Create planning prompt
    prompt = _create_plan_prompt(task, tools_summary)

    # Create messages for planning (don't include conversation history)
    messages = [
        {"role": "system", "content": "You are a planning assistant. Create clear, actionable plans."},
        {"role": "user", "content": prompt},
    ]

    try:
        # Call the LLM directly without updating history
        engine.llm.reset()
        response = engine.llm.create_chat_completion(
            messages=messages,
            temperature=0.3,  # Slightly creative but focused
            max_tokens=500,
        )

        plan_text = response["choices"][0]["message"]["content"]
        return {
            "task": task,
            "plan": plan_text,
            "tools_considered": [e.name for e in registry if e.name != "plan"],
        }

    except Exception as e:
        return {"error": f"Planning failed: {e}"}
