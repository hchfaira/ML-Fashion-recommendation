"""
Weather Service
===============
High-level façade for all weather functionality.

This module exposes a unified ``WeatherService`` (alias of
``RealWeatherService``) that:

  1. Tries the real-time ``WeatherAPIClient`` (Open-Meteo → OWM fallback).
  2. Falls back to stale cache, then static ``weather_data.json``.
  3. Converts ``WeatherData`` → ``WeatherContext`` for downstream consumers.
  4. Exposes ``WeatherOutfitAdvisor`` for scoring garment recommendations.

Backward compatibility
----------------------
The old ``WeatherService`` class is preserved as an alias so existing imports
work unchanged.  New code should call ``get_weather_data()`` for the rich DTO.
"""
from __future__ import annotations

from typing import Optional

from src.core.models import WeatherContext
from src.core import get_logger
from src.layer3_context.weather.weather_api_client import WeatherAPIClient
from src.layer3_context.weather.weather_cache import WeatherCache
from src.layer3_context.weather.weather_models import WeatherData
from src.layer3_context.weather.weather_outfit_advisor import WeatherOutfitAdvisor

logger = get_logger(__name__)


class WeatherService:
    """
    Real-time weather service.

    Uses Open-Meteo (no key required) as primary source, with
    OpenWeatherMap as optional secondary provider.  Falls back to
    stale cache → static JSON → sentinel on total failure.

    Parameters
    ----------
    cache : WeatherCache | None
        Custom cache; ``None`` uses the default file-backed cache.
    owm_api_key : str | None
        OpenWeatherMap key.  Falls back to ``WEATHER_API_KEY`` env-var.
    """

    def __init__(
        self,
        cache: WeatherCache | None = None,
        owm_api_key: str | None = None,
    ) -> None:
        self._client = WeatherAPIClient(cache=cache, owm_api_key=owm_api_key)

    # ------------------------------------------------------------------
    # Rich API (returns WeatherData)
    # ------------------------------------------------------------------

    async def get_weather_data(
        self,
        location: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        *,
        force_refresh: bool = False,
    ) -> WeatherData:
        """
        Fetch current weather as a rich ``WeatherData`` object.

        Priority: coordinates > city name > default location.
        """
        if lat is not None and lon is not None:
            return await self._client.get_by_coords(
                lat, lon, city=location, force_refresh=force_refresh
            )
        if location:
            return await self._client.get_by_city(location)
        return await self._client.get_default()

    def get_outfit_advisor(self, weather_data: WeatherData) -> WeatherOutfitAdvisor:
        """Return a ``WeatherOutfitAdvisor`` for the given weather snapshot."""
        return WeatherOutfitAdvisor(weather_data)

    # ------------------------------------------------------------------
    # Backward-compatible thin API (returns WeatherContext)
    # ------------------------------------------------------------------

    async def get_weather(self, location: str) -> Optional[WeatherContext]:
        """Backward-compatible.  Returns ``WeatherContext`` or ``None``."""
        try:
            data = await self._client.get_by_city(location)
            if data.condition.value == "unknown":
                return None
            return data.to_weather_context()
        except Exception as exc:
            logger.error("get_weather failed: %s", exc)
            return None

    async def get_weather_by_coords(
        self,
        lat: float,
        lon: float,
    ) -> Optional[WeatherContext]:
        """Backward-compatible.  Returns ``WeatherContext`` or ``None``."""
        try:
            data = await self._client.get_by_coords(lat, lon)
            if data.condition.value == "unknown":
                return None
            return data.to_weather_context()
        except Exception as exc:
            logger.error("get_weather_by_coords failed: %s", exc)
            return None

    def get_clothing_recommendations(self, weather: WeatherContext) -> dict:
        """
        Backward-compatible clothing recommendation dict.

        Internally delegates to ``WeatherOutfitAdvisor``.
        """
        from src.layer3_context.weather.weather_models import WeatherCondition as WC
        cond_map = {
            "sunny": WC.SUNNY, "cloudy": WC.CLOUDY, "rainy": WC.RAINY,
            "snowy": WC.SNOWY, "stormy": WC.STORMY, "foggy": WC.FOGGY,
            "drizzle": WC.DRIZZLE,
        }
        cond = cond_map.get(str(weather.condition).lower(), WC.CLOUDY)
        wd = WeatherData(
            temperature_celsius=weather.temperature_celsius,
            feels_like_celsius=getattr(weather, "feels_like_celsius", None)
            or weather.temperature_celsius,
            condition=cond,
            humidity=weather.humidity,
            wind_speed_kmh=weather.wind_speed_kmh or 0.0,
            uv_index=weather.uv_index,
        )
        advisor = WeatherOutfitAdvisor(wd)
        advice = advisor.get_advice()

        temp = weather.temperature_celsius
        recommendations: dict = {
            "layers": [],
            "avoid": advice["penalise_materials"] + advice["penalise_footwear"],
            "suggested_materials": advice["boost_materials"],
            "accessories": advice["boost_accessories"],
        }
        if temp < 5:
            recommendations["layers"] = ["heavy_coat", "sweater", "thermal"]
        elif temp < 12:
            recommendations["layers"] = ["jacket", "light_sweater"]
        elif temp < 18:
            recommendations["layers"] = ["light_jacket", "cardigan"]
        elif temp < 25:
            recommendations["layers"] = ["light_layers"]
        else:
            recommendations["layers"] = ["minimal"]
        return recommendations


# Backward-compat alias
RealWeatherService = WeatherService
