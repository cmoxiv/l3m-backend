"""
Utility tools implementation - random numbers, UUIDs, hashing, base64, JSON formatting.
"""

import base64
import hashlib
import json
import random
from typing import Any, Literal
from uuid import uuid4

from l3m_backend.core import tool_output
from l3m_backend.tools._registry import registry


# -----------------------------
# Random Number Tool
# -----------------------------

@registry.register(aliases=["rand", "random"])
@tool_output(llm_format=lambda x: str(x.get('result', x)))
def random_number(
    min_val: int = 1,
    max_val: int = 100,
    count: int = 1,
) -> dict[str, Any]:
    """Generate random numbers.

    Args:
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).
        count: Number of values to generate (default 1).

    Returns:
        Dictionary with random number(s).
    """
    if min_val > max_val:
        return {"error": "min_val must be <= max_val"}

    count = min(max(count, 1), 100)  # Limit to 1-100

    if count == 1:
        return {"result": random.randint(min_val, max_val)}
    else:
        return {"results": [random.randint(min_val, max_val) for _ in range(count)]}


# -----------------------------
# UUID Generation Tool
# -----------------------------

@registry.register(aliases=["uuid"])
@tool_output(llm_format=lambda x: x.get('uuid', str(x)))
def uuid_generate(
    count: int = 1,
    format: Literal["standard", "hex", "urn"] = "standard",
) -> dict[str, Any]:
    """Generate UUID(s).

    Args:
        count: Number of UUIDs to generate (default 1, max 10).
        format: Output format (standard, hex, urn).

    Returns:
        Dictionary with generated UUID(s).
    """
    count = min(max(count, 1), 10)

    def format_uuid(u):
        if format == "hex":
            return u.hex
        elif format == "urn":
            return u.urn
        return str(u)

    if count == 1:
        return {"uuid": format_uuid(uuid4())}
    else:
        return {"uuids": [format_uuid(uuid4()) for _ in range(count)]}


# -----------------------------
# Hash Tool
# -----------------------------

@registry.register(aliases=["hash"])
@tool_output(llm_format=lambda x: f"{x.get('algorithm', 'hash')}: {x.get('hash', x.get('error'))}")
def hash_text(
    text: str,
    algorithm: Literal["md5", "sha1", "sha256", "sha512"] = "sha256",
) -> dict[str, Any]:
    """Compute hash of text.

    Args:
        text: Text to hash.
        algorithm: Hash algorithm (md5, sha1, sha256, sha512).

    Returns:
        Dictionary with hash value.
    """
    algorithms = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512,
    }

    hasher = algorithms.get(algorithm)
    if not hasher:
        return {"error": f"Unknown algorithm: {algorithm}"}

    hash_value = hasher(text.encode()).hexdigest()
    return {
        "algorithm": algorithm,
        "hash": hash_value,
        "input_length": len(text),
    }


# -----------------------------
# Base64 Tool
# -----------------------------

@registry.register(aliases=["b64"])
@tool_output(llm_format=lambda x: x.get('result', x.get('error', str(x))))
def base64_encode(
    text: str,
    action: Literal["encode", "decode"] = "encode",
) -> dict[str, Any]:
    """Encode or decode Base64.

    Args:
        text: Text to encode or Base64 string to decode.
        action: 'encode' or 'decode'.

    Returns:
        Dictionary with result.
    """
    try:
        if action == "encode":
            result = base64.b64encode(text.encode()).decode()
            return {"result": result, "action": "encoded"}
        else:
            result = base64.b64decode(text.encode()).decode()
            return {"result": result, "action": "decoded"}
    except Exception as e:
        return {"error": f"Base64 {action} failed: {e}"}


# -----------------------------
# JSON Format Tool
# -----------------------------

@registry.register(aliases=["json", "jsonformat"])
@tool_output(llm_format=lambda x: x.get('formatted', x.get('error', str(x))))
def json_format(
    text: str,
    indent: int = 2,
    minify: bool = False,
) -> dict[str, Any]:
    """Format or validate JSON.

    Args:
        text: JSON string to format.
        indent: Indentation level (default 2).
        minify: If True, output compact JSON.

    Returns:
        Dictionary with formatted JSON or error.
    """
    try:
        parsed = json.loads(text)

        if minify:
            formatted = json.dumps(parsed, separators=(",", ":"))
        else:
            formatted = json.dumps(parsed, indent=indent)

        return {
            "valid": True,
            "formatted": formatted,
        }
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "error": f"Invalid JSON: {e}",
        }
