"""
WeatherNormalizer
=================
Converts raw API responses (Open-Meteo **and** OpenWeatherMap) into the
application's canonical ``WeatherData`` model.

Both providers return slightly different JSON schemas; this module hides
that complexity from the rest of the codebase.
"""
from __future__ import annotations

from typing import Any

from .weather_models import WeatherData, WeatherCondition
from src.core import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# WMO Weather Interpretation Codes → WeatherCondition
# https://open-meteo.com/en/docs#weathervariables
# ---------------------------------------------------------------------------
_WMO_TO_CONDITION: dict[int, WeatherCondition] = {
    0: WeatherCondition.SUNNY,        # Clear sky
    1: WeatherCondition.SUNNY,        # Mainly clear
    2: WeatherCondition.CLOUDY,       # Partly cloudy
    3: WeatherCondition.CLOUDY,       # Overcast
    45: WeatherCondition.FOGGY,       # Fog
    48: WeatherCondition.FOGGY,       # Rime fog
    51: WeatherCondition.DRIZZLE,     # Drizzle light
    53: WeatherCondition.DRIZZLE,     # Drizzle moderate
    55: WeatherCondition.DRIZZLE,     # Drizzle dense
    56: WeatherCondition.DRIZZLE,     # Freezing drizzle light
    57: WeatherCondition.DRIZZLE,     # Freezing drizzle dense
    61: WeatherCondition.RAINY,       # Rain slight
    63: WeatherCondition.RAINY,       # Rain moderate
    65: WeatherCondition.RAINY,       # Rain heavy
    66: WeatherCondition.RAINY,       # Freezing rain light
    67: WeatherCondition.RAINY,       # Freezing rain heavy
    71: WeatherCondition.SNOWY,       # Snow fall slight
    73: WeatherCondition.SNOWY,       # Snow fall moderate
    75: WeatherCondition.SNOWY,       # Snow fall heavy
    77: WeatherCondition.SNOWY,       # Snow grains
    80: WeatherCondition.RAINY,       # Rain showers slight
    81: WeatherCondition.RAINY,       # Rain showers moderate
    82: WeatherCondition.RAINY,       # Rain showers violent
    85: WeatherCondition.SNOWY,       # Snow showers slight
    86: WeatherCondition.SNOWY,       # Snow showers heavy
    95: WeatherCondition.STORMY,      # Thunderstorm slight/moderate
    96: WeatherCondition.STORMY,      # Thunderstorm with slight hail
    99: WeatherCondition.STORMY,      # Thunderstorm with heavy hail
}

# OWM "main" condition string → WeatherCondition
_OWM_MAIN_TO_CONDITION: dict[str, WeatherCondition] = {
    "clear": WeatherCondition.SUNNY,
    "clouds": WeatherCondition.CLOUDY,
    "rain": WeatherCondition.RAINY,
    "drizzle": WeatherCondition.DRIZZLE,
    "thunderstorm": WeatherCondition.STORMY,
    "snow": WeatherCondition.SNOWY,
    "mist": WeatherCondition.FOGGY,
    "smoke": WeatherCondition.FOGGY,
    "haze": WeatherCondition.FOGGY,
    "dust": WeatherCondition.FOGGY,
    "fog": WeatherCondition.FOGGY,
    "sand": WeatherCondition.FOGGY,
    "ash": WeatherCondition.FOGGY,
    "squall": WeatherCondition.STORMY,
    "tornado": WeatherCondition.STORMY,
}


class WeatherNormalizer:
    """
    Converts raw provider payloads into a ``WeatherData`` instance.

    Usage::

        normalizer = WeatherNormalizer()
        data = normalizer.from_open_meteo(raw_json, lat=48.85, lon=2.35)
        data = normalizer.from_openweathermap(raw_json)
    """

    # ------------------------------------------------------------------
    # Open-Meteo
    # ------------------------------------------------------------------

    def from_open_meteo(
        self,
        raw: dict[str, Any],
        lat: float,
        lon: float,
        city: str | None = None,
    ) -> WeatherData:
        """
        Parse the ``/v1/forecast`` response from Open-Meteo.

        The API returns a ``current_weather`` block when
        ``current_weather=true`` is passed, **and** optionally
        ``current`` blocks when individual variables are requested.
        We support both layouts.
        """
        try:
            cw = raw.get("current_weather", {})
            current = raw.get("current", {})
            cu = raw.get("current_units", {})

            # ---- temperature ------------------------------------------------
            temp = (
                current.get("temperature_2m")
                or cw.get("temperature")
                or 15.0
            )
            feels_like = current.get("apparent_temperature", temp)

            # ---- wind -------------------------------------------------------
            wind_ms = current.get("wind_speed_10m") or cw.get("windspeed") or 0.0
            # Open-Meteo default unit is km/h for current_weather, m/s for current
            if cu.get("wind_speed_10m") in ("m/s",):
                wind_kmh = wind_ms * 3.6
            else:
                wind_kmh = float(wind_ms)

            # ---- humidity ---------------------------------------------------
            humidity = current.get("relative_humidity_2m")

            # ---- precipitation ----------------------------------------------
            precip_mm = current.get("precipitation") or current.get("rain") or 0.0

            # ---- UV index ---------------------------------------------------
            uv = current.get("uv_index")
            if uv is not None:
                uv = max(0, min(11, int(round(uv))))

            # ---- weather code → condition -----------------------------------
            wmo_code = int(
                current.get("weather_code")
                or cw.get("weathercode")
                or 0
            )
            condition = _WMO_TO_CONDITION.get(wmo_code, WeatherCondition.CLOUDY)

            # ---- sunrise / sunset (optional) --------------------------------
            daily = raw.get("daily", {})
            sunrises = daily.get("sunrise") or []
            sunsets = daily.get("sunset") or []

            return WeatherData(
                temperature_celsius=float(temp),
                feels_like_celsius=float(feels_like),
                condition=condition,
                humidity=float(humidity) if humidity is not None else None,
                wind_speed_kmh=round(wind_kmh, 1),
                uv_index=uv,
                precipitation_mm=float(precip_mm),
                wmo_code=wmo_code,
                lat=lat,
                lon=lon,
                city=city,
                provider="open-meteo",
                sunrise=sunrises[0] if sunrises else None,
                sunset=sunsets[0] if sunsets else None,
            )

        except Exception as exc:  # pragma: no cover
            logger.error("open-meteo normalizer error: %s", exc)
            return WeatherData.unknown(lat=lat, lon=lon, city=city)

    # ------------------------------------------------------------------
    # OpenWeatherMap
    # ------------------------------------------------------------------

    def from_openweathermap(
        self,
        raw: dict[str, Any],
        city: str | None = None,
    ) -> WeatherData:
        """Parse an OpenWeatherMap ``/data/2.5/weather`` response."""
        try:
            main = raw.get("main", {})
            weather_list = raw.get("weather", [{}])
            weather_entry = weather_list[0] if weather_list else {}
            wind = raw.get("wind", {})
            rain = raw.get("rain", {})
            snow = raw.get("snow", {})
            coord = raw.get("coord", {})
            sys = raw.get("sys", {})

            # ---- condition --------------------------------------------------
            raw_main = weather_entry.get("main", "clouds").lower()
            condition = _OWM_MAIN_TO_CONDITION.get(raw_main, WeatherCondition.CLOUDY)

            # ---- temperatures -----------------------------------------------
            temp = float(main.get("temp", 15))
            feels_like = float(main.get("feels_like", temp))

            # ---- wind: OWM gives m/s ----------------------------------------
            wind_ms = float(wind.get("speed", 0))
            wind_kmh = round(wind_ms * 3.6, 1)

            # ---- precipitation ----------------------------------------------
            precip_1h = float(rain.get("1h", 0) or snow.get("1h", 0))

            # ---- humidity ---------------------------------------------------
            humidity = float(main.get("humidity", 0)) or None

            # ---- derived city -----------------------------------------------
            resolved_city = city or raw.get("name")

            return WeatherData(
                temperature_celsius=temp,
                feels_like_celsius=feels_like,
                condition=condition,
                humidity=humidity,
                wind_speed_kmh=wind_kmh,
                uv_index=None,  # not in /weather endpoint; use /onecall for UV
                precipitation_mm=precip_1h,
                wmo_code=None,
                lat=float(coord.get("lat", 0)),
                lon=float(coord.get("lon", 0)),
                city=resolved_city,
                provider="openweathermap",
                sunrise=sys.get("sunrise"),   # Unix timestamp
                sunset=sys.get("sunset"),
            )

        except Exception as exc:  # pragma: no cover
            logger.error("owm normalizer error: %s", exc)
            return WeatherData.unknown()
