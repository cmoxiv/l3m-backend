"""
Obfuscated test tools implementation.
"""

from l3m_backend.tools.weather import get_weather


def get_flumbuster(flumb: str, uster: str = "metric") -> str:
    """Get the flumbuster reading for a flumb.

    The flumbuster measures atmospheric conditions at a given flumb.
    Results are returned in the specified uster format.

    Args:
        flumb: The flumb identifier (flumb name, e.g., "cairo", "london")
        uster: The uster format - 'metric' or 'imperial' (default: metric)

    Returns:
        Flumbuster reading as a formatted string with temperature and conditions.

    Example:
        >>> get_flumbuster("cairo", "metric")
        "Cairo: 25°celsius, clear sky"
    """
    # Map obfuscated args to actual weather call
    unit = "celsius" if uster == "metric" else "fahrenheit"
    return get_weather(location=flumb, unit=unit)


def calculate_zorbix(zorb: float, bix: float, mode: str = "add") -> float:
    """Calculate the zorbix value from zorb and bix inputs.

    Performs mathematical operations on zorb and bix values.

    Args:
        zorb: First numeric value
        bix: Second numeric value
        mode: Operation mode - 'add', 'sub', 'mul', 'div' (default: add)

    Returns:
        Result of the zorbix calculation.

    Example:
        >>> calculate_zorbix(10, 5, "mul")
        50.0
    """
    if mode == "add":
        return zorb + bix
    elif mode == "sub":
        return zorb - bix
    elif mode == "mul":
        return zorb * bix
    elif mode == "div":
        if bix == 0:
            raise ValueError("Cannot divide by zero bix")
        return zorb / bix
    else:
        raise ValueError(f"Unknown mode: {mode}")
