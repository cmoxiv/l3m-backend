"""Unit conversion tool."""

from typing import Any

from pydantic import Field

from l3m_backend.core import tool_output
from l3m_backend.tools._registry import registry


_UNIT_CONVERSIONS = {
    # Length
    ("km", "miles"): lambda x: x * 0.621371,
    ("miles", "km"): lambda x: x * 1.60934,
    ("m", "ft"): lambda x: x * 3.28084,
    ("ft", "m"): lambda x: x * 0.3048,
    ("cm", "in"): lambda x: x * 0.393701,
    ("in", "cm"): lambda x: x * 2.54,
    # Weight
    ("kg", "lb"): lambda x: x * 2.20462,
    ("lb", "kg"): lambda x: x * 0.453592,
    ("g", "oz"): lambda x: x * 0.035274,
    ("oz", "g"): lambda x: x * 28.3495,
    # Temperature
    ("c", "f"): lambda x: (x * 9/5) + 32,
    ("f", "c"): lambda x: (x - 32) * 5/9,
    ("c", "k"): lambda x: x + 273.15,
    ("k", "c"): lambda x: x - 273.15,
    # Volume
    ("l", "gal"): lambda x: x * 0.264172,
    ("gal", "l"): lambda x: x * 3.78541,
    ("ml", "floz"): lambda x: x * 0.033814,
    ("floz", "ml"): lambda x: x * 29.5735,
    # Speed
    ("kmh", "mph"): lambda x: x * 0.621371,
    ("mph", "kmh"): lambda x: x * 1.60934,
    # Data
    ("mb", "gb"): lambda x: x / 1024,
    ("gb", "mb"): lambda x: x * 1024,
    ("gb", "tb"): lambda x: x / 1024,
    ("tb", "gb"): lambda x: x * 1024,
}


@registry.register(aliases=["convert", "unit"])
@tool_output(llm_format=lambda x: f"{x['value']} {x['from_unit']} = {x['result']} {x['to_unit']}" if 'result' in x else x.get('error'))
def unit_convert(
    value: float = Field(description="Numeric value to convert"),
    from_unit: str = Field(description="Source unit (e.g., km, lb, c)"),
    to_unit: str = Field(description="Target unit (e.g., miles, kg, f)"),
) -> dict[str, Any]:
    """Convert between units.

    Supported conversions:
    - Length: km/miles, m/ft, cm/in
    - Weight: kg/lb, g/oz
    - Temperature: c/f/k (Celsius/Fahrenheit/Kelvin)
    - Volume: l/gal, ml/floz
    - Speed: kmh/mph
    - Data: mb/gb/tb

    Args:
        value: The numeric value to convert.
        from_unit: Source unit abbreviation.
        to_unit: Target unit abbreviation.

    Returns:
        Dictionary with conversion result or error.
    """
    from_u = from_unit.lower().strip()
    to_u = to_unit.lower().strip()

    converter = _UNIT_CONVERSIONS.get((from_u, to_u))
    if converter:
        result = converter(value)
        return {
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "result": round(result, 4),
        }
    else:
        available = sorted(set(f"{a}->{b}" for a, b in _UNIT_CONVERSIONS.keys()))
        return {
            "error": f"Unknown conversion: {from_unit} -> {to_unit}",
            "available": available,
        }
