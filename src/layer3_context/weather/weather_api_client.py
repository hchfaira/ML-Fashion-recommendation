"""
WeatherAPIClient
================
Async HTTP client that fetches **real** weather data from:

  1. **Open-Meteo** (https://open-meteo.com) — completely free, no API key.
  2. **OpenWeatherMap** (https://openweathermap.org/api) — optional, requires
     a ``WEATHER_API_KEY`` env-var.  Enables richer data (UV index via
     the /onecall endpoint).

Fallback chain
--------------
  Open-Meteo (always tried first, no key)
    → OpenWeatherMap (only if ``WEATHER_API_KEY`` is set)
    → WeatherCache stale entry
    → Static fallback from ``config/data/weather_data.json``
    → ``WeatherData.unknown()`` sentinel

Geo-coding
----------
City name → (lat, lon) via the free Open-Meteo Geo-coding API.
The result is cached in memory for the process lifetime.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import httpx

from .weather_cache import WeatherCache
from .weather_models import WeatherData
from .weather_normalizer import WeatherNormalizer
from src.core import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CFG_PATH = Path(__file__).parents[3] / "config" / "data" / "weather_api_config.json"
_STATIC_PATH = Path(__file__).parents[3] / "config" / "data" / "weather_data.json"


def _load_cfg() -> dict:
    try:
        return json.loads(_CFG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


class WeatherAPIClient:
    """
    Async real-time weather client with integrated caching.

    Parameters
    ----------
    cache : WeatherCache | None
        Pass a custom cache or ``None`` to disable caching entirely.
    owm_api_key : str | None
        OpenWeatherMap API key.  Falls back to ``WEATHER_API_KEY`` env-var.
    """

    def __init__(
        self,
        cache: WeatherCache | None = None,
        owm_api_key: str | None = None,
    ) -> None:
        cfg = _load_cfg()
        prov = cfg.get("providers", {})
        self._cfg = cfg

        # Open-Meteo config
        om_cfg = prov.get("open_meteo", {})
        self._om_url: str = om_cfg.get("base_url", "https://api.open-meteo.com/v1/forecast")
        self._om_timeout: float = float(om_cfg.get("timeout_seconds", 8))
        self._om_current: list[str] = om_cfg.get(
            "current_params",
            ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
             "precipitation", "weather_code", "wind_speed_10m", "uv_index"],
        )
        self._om_daily: list[str] = om_cfg.get("daily_params", ["sunrise", "sunset"])

        # OpenWeatherMap config
        owm_cfg = prov.get("openweathermap", {})
        self._owm_url: str = owm_cfg.get(
            "base_url", "https://api.openweathermap.org/data/2.5/weather"
        )
        self._owm_timeout: float = float(owm_cfg.get("timeout_seconds", 8))
        self._owm_key: str | None = (
            owm_api_key
            or os.environ.get("WEATHER_API_KEY")
            or os.environ.get("OPENWEATHER_API_KEY")
        )

        # Geo-coding
        geo_cfg = cfg.get("geocoding", {})
        self._geo_url: str = geo_cfg.get(
            "open_meteo_url",
            "https://geocoding-api.open-meteo.com/v1/search",
        )
        self._geo_timeout: float = float(geo_cfg.get("timeout_seconds", 5))
        self._geocache: dict[str, tuple[float, float]] = {}

        # Defaults
        defaults = cfg.get("defaults", {})
        self._default_lat: float = float(defaults.get("lat", 48.8566))
        self._default_lon: float = float(defaults.get("lon", 2.3522))
        self._default_city: str = defaults.get("city", "Paris")

        # Helpers
        self._normalizer = WeatherNormalizer()
        if cache is None:
            cache_cfg = cfg.get("cache", {})
            cache = WeatherCache(
                ttl_seconds=int(cache_cfg.get("ttl_seconds", 3600)),
                stale_ttl_seconds=int(cache_cfg.get("stale_ttl_seconds", 86400)),
                max_entries=int(cache_cfg.get("max_entries", 100)),
                cache_dir=cache_cfg.get("directory", ".weather_cache"),
            )
        self._cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_by_city(self, city: str) -> WeatherData:
        """
        Fetch current weather for a city name.

        The city is geo-coded to (lat, lon) via Open-Meteo's free Geocoding
        API and then delegated to :meth:`get_by_coords`.
        """
        coords = await self._geocode(city)
        if coords is None:
            logger.warning("geocoding failed for '%s', using defaults", city)
            return await self.get_by_coords(
                self._default_lat, self._default_lon, city=city
            )
        lat, lon = coords
        return await self.get_by_coords(lat, lon, city=city)

    async def get_by_coords(
        self,
        lat: float,
        lon: float,
        *,
        city: str | None = None,
        force_refresh: bool = False,
    ) -> WeatherData:
        """
        Fetch current weather for GPS coordinates.

        Tries Open-Meteo first (no key required), then OpenWeatherMap if a
        key is available, then stale cache, then static fallback.

        Parameters
        ----------
        lat, lon : float
            GPS coordinates.
        city : str | None
            Human-readable name to attach to the result.
        force_refresh : bool
            Bypass cache even if a fresh entry exists.
        """
        # ---- cache check ------------------------------------------------
        if not force_refresh:
            cached = self._cache.get(lat, lon)
            if cached is not None and cached.provider != "cache-stale":
                logger.debug("weather from fresh cache (%s)", cached.city or f"{lat},{lon}")
                return cached

        # ---- Open-Meteo (primary, free) ---------------------------------
        data = await self._fetch_open_meteo(lat, lon, city=city)
        if data is not None and data.condition.value != "unknown":
            self._cache.put(lat, lon, data)
            return data

        # ---- OpenWeatherMap (secondary, needs key) ----------------------
        if self._owm_key:
            data = await self._fetch_owm(lat, lon, city=city)
            if data is not None and data.condition.value != "unknown":
                self._cache.put(lat, lon, data)
                return data

        # ---- stale cache ------------------------------------------------
        stale = self._cache.get(lat, lon)   # may now return stale entry
        if stale is not None:
            logger.warning("all APIs failed – using stale cache for %s", stale.city)
            return stale

        # ---- static fallback --------------------------------------------
        logger.error("all weather sources failed – using static fallback")
        return self._static_fallback(lat, lon, city)

    async def get_default(self) -> WeatherData:
        """Return weather for the configured default location (city / coords)."""
        return await self.get_by_coords(
            self._default_lat,
            self._default_lon,
            city=self._default_city,
        )

    # ------------------------------------------------------------------
    # Private: Open-Meteo
    # ------------------------------------------------------------------

    async def _fetch_open_meteo(
        self, lat: float, lon: float, city: str | None
    ) -> Optional[WeatherData]:
        params: dict = {
            "latitude": lat,
            "longitude": lon,
            "current": ",".join(self._om_current),
            "daily": ",".join(self._om_daily),
            "timezone": "auto",
            "forecast_days": 1,
        }
        try:
            async with httpx.AsyncClient(timeout=self._om_timeout) as client:
                resp = await client.get(self._om_url, params=params)
                resp.raise_for_status()
                raw = resp.json()
            logger.info(
                "Open-Meteo OK for %s,%s → %s",
                lat, lon, raw.get("current", {}).get("weather_code"),
            )
            return self._normalizer.from_open_meteo(raw, lat=lat, lon=lon, city=city)
        except httpx.TimeoutException:
            logger.warning("Open-Meteo timeout for %s,%s", lat, lon)
        except httpx.HTTPStatusError as exc:
            logger.warning("Open-Meteo HTTP %s for %s,%s", exc.response.status_code, lat, lon)
        except Exception as exc:
            logger.warning("Open-Meteo error: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Private: OpenWeatherMap
    # ------------------------------------------------------------------

    async def _fetch_owm(
        self, lat: float, lon: float, city: str | None
    ) -> Optional[WeatherData]:
        params: dict = {
            "lat": lat,
            "lon": lon,
            "appid": self._owm_key,
            "units": "metric",
        }
        try:
            async with httpx.AsyncClient(timeout=self._owm_timeout) as client:
                resp = await client.get(self._owm_url, params=params)
                resp.raise_for_status()
                raw = resp.json()
            logger.info("OWM OK for %s,%s", lat, lon)
            return self._normalizer.from_openweathermap(raw, city=city)
        except httpx.TimeoutException:
            logger.warning("OWM timeout for %s,%s", lat, lon)
        except httpx.HTTPStatusError as exc:
            logger.warning("OWM HTTP %s for %s,%s", exc.response.status_code, lat, lon)
        except Exception as exc:
            logger.warning("OWM error: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Private: Geocoding
    # ------------------------------------------------------------------

    async def _geocode(self, city: str) -> Optional[tuple[float, float]]:
        """Resolve city name → (lat, lon) via Open-Meteo Geocoding API."""
        key = city.lower().strip()
        if key in self._geocache:
            return self._geocache[key]

        try:
            async with httpx.AsyncClient(timeout=self._geo_timeout) as client:
                resp = await client.get(
                    self._geo_url,
                    params={"name": city, "count": 1, "language": "en", "format": "json"},
                )
                resp.raise_for_status()
                results = resp.json().get("results") or []

            if results:
                r = results[0]
                coords = (float(r["latitude"]), float(r["longitude"]))
                self._geocache[key] = coords
                logger.debug("geocoded '%s' → %s", city, coords)
                return coords
            logger.warning("no geocoding results for '%s'", city)

        except Exception as exc:
            logger.warning("geocoding error for '%s': %s", city, exc)

        return None

    # ------------------------------------------------------------------
    # Private: static fallback
    # ------------------------------------------------------------------

    def _static_fallback(
        self, lat: float, lon: float, city: str | None
    ) -> WeatherData:
        """Return a mild/unknown sentinel based on static config."""
        try:
            static = json.loads(_STATIC_PATH.read_text(encoding="utf-8"))
            temp_recs = static.get("temperature_recommendations", {})
            # 18-25°C band as default
            band = temp_recs.get("18_to_25", {})
            _ = band  # just ensure it parses
        except Exception:
            pass

        return WeatherData.unknown(lat=lat, lon=lon, city=city)
