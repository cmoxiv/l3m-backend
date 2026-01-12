"""Currency conversion tool."""

import json
import urllib.error
import urllib.request
from typing import Any

from pydantic import Field

from l3m_backend.core import tool_output
from l3m_backend.tools._registry import registry


@registry.register(aliases=["currency", "fx"])
@tool_output(llm_format=lambda x: f"{x['amount']} {x['from_currency']} = {x['result']} {x['to_currency']}" if 'result' in x else x.get('error'))
def currency_convert(
    amount: float = Field(description="Amount to convert"),
    from_currency: str = Field(description="Source currency code (e.g., USD, EUR)"),
    to_currency: str = Field(description="Target currency code (e.g., GBP, JPY)"),
) -> dict[str, Any]:
    """Convert between currencies using live exchange rates.

    Uses the free exchangerate-api.com API.

    Args:
        amount: The amount to convert.
        from_currency: Source currency code (e.g., USD, EUR, GBP).
        to_currency: Target currency code.

    Returns:
        Dictionary with conversion result or error.
    """
    try:
        from_c = from_currency.upper().strip()
        to_c = to_currency.upper().strip()

        url = f"https://api.exchangerate-api.com/v4/latest/{from_c}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())

        rates = data.get("rates", {})
        if to_c not in rates:
            return {"error": f"Unknown currency: {to_c}"}

        rate = rates[to_c]
        result = amount * rate

        return {
            "amount": amount,
            "from_currency": from_c,
            "to_currency": to_c,
            "rate": rate,
            "result": round(result, 2),
        }
    except urllib.error.HTTPError as e:
        return {"error": f"Unknown currency or API error: {e.code}"}
    except Exception as e:
        return {"error": str(e)}
