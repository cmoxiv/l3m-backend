"""Tool magic command - invoke tool and add to history."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable

from l3m_backend.cli.magic.registry import magic_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager


@magic_registry.register("tool", "Invoke tool (adds to history)")
def magic_tool(
    engine: "ChatEngine",
    session_mgr: "SessionManager | None",
    args: str,
    add_to_history: Callable[[str, str], None],
) -> bool:
    """Invoke tool and add to history as user/assistant exchange."""
    parts = args.split(None, 1)
    if not parts:
        print("Usage: %tool <name> [json_args]")
        print("Example: %tool get_time {}")
        print('Example: %tool calculate {"expression": "2+2"}')
        return True

    name = parts[0]
    json_args = parts[1] if len(parts) > 1 else "{}"

    try:
        arguments = json.loads(json_args)
        result = engine.registry.execute(name, arguments)
        output = engine._result2llm(result.output)
        print(f"Tool result:\n{output}")

        # Add to history as user message + assistant response
        user_msg = f"%tool {name} {json_args}"
        add_to_history(user_msg, output)
    except json.JSONDecodeError:
        print(f"Invalid JSON arguments: {json_args}")
    except Exception as e:
        print(f"Tool error: {e}")
    return True
