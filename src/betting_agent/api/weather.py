"""
OpenWeatherMap client (optional modifier).
Returns temperature and wind for a given city/coordinates.
"""

from __future__ import annotations

import logging

import requests

from betting_agent.config import settings

logger = logging.getLogger(__name__)


class WeatherClient:
    BASE = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self):
        self.api_key = settings.weather_api_key

    def get_weather(self, city: str) -> dict | None:
        """
        Fetch current weather for a city.
        Returns dict with keys: temperature_f, wind_mph, description.
        """
        if not self.api_key:
            return None
        try:
            resp = requests.get(
                self.BASE,
                params={"q": city, "appid": self.api_key, "units": "imperial"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "temperature_f": data["main"]["temp"],
                "wind_mph": data["wind"]["speed"],
                "description": data["weather"][0]["description"],
            }
        except Exception as exc:
            logger.warning("Weather API error for %s: %s", city, exc)
            return None
