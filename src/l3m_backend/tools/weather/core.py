"""
Weather tool - fetches real-time weather data from Open-Meteo API.
"""

import json
import urllib.parse
import urllib.request
from typing import Any, Literal

from pydantic import Field

from l3m_backend.core import tool_output
from l3m_backend.tools._registry import registry


def _geocode(city: str) -> tuple[float, float, str]:
    """Get coordinates for a city using Open-Meteo geocoding API.

    Args:
        city: Name of the city to geocode.

    Returns:
        Tuple of (latitude, longitude, canonical_city_name).

    Raises:
        ValueError: If the city is not found in the geocoding database.
        URLError: If the API request fails.
    """
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())
    if not data.get("results"):
        raise ValueError(f"City not found: {city}")
    result = data["results"][0]
    return result["latitude"], result["longitude"], result["name"]


def _get_weather_code_description(code: int) -> str:
    """Convert WMO weather code to human-readable description.

    Uses the World Meteorological Organization (WMO) standard weather codes.

    Args:
        code: WMO weather code integer.

    Returns:
        Human-readable weather condition string. Returns "unknown" for
        unrecognized codes.

    Example:
        >>> _get_weather_code_description(0)
        'clear sky'
        >>> _get_weather_code_description(63)
        'moderate rain'
    """
    codes = {
        0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
        45: "foggy", 48: "depositing rime fog",
        51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
        61: "slight rain", 63: "moderate rain", 65: "heavy rain",
        71: "slight snow", 73: "moderate snow", 75: "heavy snow",
        80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
        95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
    }
    return codes.get(code, "unknown")


@registry.register(aliases=["weather", "w"])
@tool_output(llm_format="{location}: {temperature}°{unit}, {condition}")
def get_weather(
    location: str = Field(description="City name"),
    unit: Literal["celsius", "fahrenheit"] = "celsius",
) -> dict[str, Any]:
    """Get the current weather for a location.

    Fetches real-time weather data from the Open-Meteo API (free, no API key required).
    Uses geocoding to resolve city names to coordinates, then retrieves current weather.

    Args:
        location: Name of the city (e.g., "Tokyo", "New York", "Paris").
        unit: Temperature unit - "celsius" or "fahrenheit". Defaults to "celsius".

    Returns:
        Dictionary with keys:
            - location: Canonical city name from geocoding
            - temperature: Current temperature (rounded to nearest integer)
            - unit: The unit used ("celsius" or "fahrenheit")
            - condition: Weather condition description (e.g., "clear sky", "moderate rain")

        On error, returns the same structure with "unknown" temperature and error in condition.

    Example:
        >>> get_weather("Paris")
        {
            "location": "Paris",
            "temperature": 15,
            "unit": "celsius",
            "condition": "partly cloudy"
        }
    """
    try:
        lat, lon, name = _geocode(location)

        # Get current weather from Open-Meteo
        temp_unit = "fahrenheit" if unit == "fahrenheit" else "celsius"
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,weather_code"
            f"&temperature_unit={temp_unit}"
        )
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())

        current = data["current"]
        return {
            "location": name,
            "temperature": round(current["temperature_2m"]),
            "unit": unit,
            "condition": _get_weather_code_description(current["weather_code"]),
        }
    except Exception as e:
        return {
            "location": location,
            "temperature": "unknown",
            "unit": unit,
            "condition": f"error: {e}",
        }
