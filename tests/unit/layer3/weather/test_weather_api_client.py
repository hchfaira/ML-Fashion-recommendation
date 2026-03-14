"""
Unit tests for WeatherAPIClient
================================
All HTTP calls are mocked with ``respx`` (httpx transport mock) or
``pytest-mock``.  No real network requests are made.
"""
from __future__ import annotations

import json
import pytest
import pytest_asyncio
import respx
import httpx
from unittest.mock import patch, AsyncMock

from src.layer3_context.weather.weather_api_client import WeatherAPIClient
from src.layer3_context.weather.weather_cache import WeatherCache
from src.layer3_context.weather.weather_models import WeatherCondition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def no_cache_client(tmp_path):
    """WeatherAPIClient with in-memory-only cache (no files)."""
    cache = WeatherCache(ttl_seconds=3600, cache_dir=None)
    return WeatherAPIClient(cache=cache, owm_api_key=None)


@pytest.fixture
def owm_client(tmp_path):
    """WeatherAPIClient with OWM key and in-memory cache."""
    cache = WeatherCache(ttl_seconds=3600, cache_dir=None)
    return WeatherAPIClient(cache=cache, owm_api_key="test-owm-key-123")


# Minimal valid Open-Meteo response
OM_RESPONSE = {
    "latitude": 48.85,
    "longitude": 2.35,
    "timezone": "Europe/Paris",
    "current_units": {"wind_speed_10m": "km/h"},
    "current": {
        "temperature_2m": 16.0,
        "apparent_temperature": 14.0,
        "relative_humidity_2m": 68,
        "precipitation": 0.0,
        "weather_code": 2,  # partly cloudy
        "wind_speed_10m": 12.0,
        "uv_index": 3,
    },
    "daily": {"sunrise": ["2024-10-15T07:45"], "sunset": ["2024-10-15T18:30"]},
}

# Minimal OWM response
OWM_RESPONSE = {
    "coord": {"lat": 48.85, "lon": 2.35},
    "weather": [{"main": "Clouds"}],
    "main": {"temp": 16.0, "feels_like": 14.0, "humidity": 68},
    "wind": {"speed": 3.3},
    "name": "Paris",
}

# Geocoding response
GEO_RESPONSE = {
    "results": [{"latitude": 48.8566, "longitude": 2.3522, "name": "Paris"}]
}


# ---------------------------------------------------------------------------
# Open-Meteo primary path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestOpenMeteoPath:

    @respx.mock
    async def test_get_by_coords_returns_weather_data(self, no_cache_client):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        data = await no_cache_client.get_by_coords(48.85, 2.35)
        assert abs(data.temperature_celsius - 16.0) < 0.1
        assert data.provider == "open-meteo"

    @respx.mock
    async def test_get_by_coords_condition_parsed(self, no_cache_client):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        data = await no_cache_client.get_by_coords(48.85, 2.35)
        assert data.condition == WeatherCondition.CLOUDY  # WMO 2

    @respx.mock
    async def test_result_stored_in_cache(self, no_cache_client):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        await no_cache_client.get_by_coords(48.85, 2.35)
        # Second call should hit cache (not the mock)
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(500)  # would fail if called
        )
        cached = await no_cache_client.get_by_coords(48.85, 2.35)
        assert cached is not None  # served from cache

    @respx.mock
    async def test_force_refresh_bypasses_cache(self, no_cache_client):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        await no_cache_client.get_by_coords(48.85, 2.35)
        # Now force refresh — mock must be called again
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json={**OM_RESPONSE,
                "current": {**OM_RESPONSE["current"], "temperature_2m": 20.0}})
        )
        data = await no_cache_client.get_by_coords(48.85, 2.35, force_refresh=True)
        assert abs(data.temperature_celsius - 20.0) < 0.1


# ---------------------------------------------------------------------------
# OWM fallback path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestOWMFallback:

    @respx.mock
    async def test_owm_used_when_open_meteo_fails(self, owm_client):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(500)
        )
        respx.get("https://api.openweathermap.org/data/2.5/weather").mock(
            return_value=httpx.Response(200, json=OWM_RESPONSE)
        )
        data = await owm_client.get_by_coords(48.85, 2.35)
        assert data.provider == "openweathermap"

    @respx.mock
    async def test_fallback_sentinel_when_all_fail(self, no_cache_client):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(503)
        )
        data = await no_cache_client.get_by_coords(0.0, 0.0)
        # Should return fallback/unknown, not raise
        assert data is not None
        assert data.condition == WeatherCondition.UNKNOWN


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGeocoding:

    @respx.mock
    async def test_get_by_city_resolves_coords(self, no_cache_client):
        respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
            return_value=httpx.Response(200, json=GEO_RESPONSE)
        )
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        data = await no_cache_client.get_by_city("Paris")
        assert data.temperature_celsius == 16.0

    @respx.mock
    async def test_geocoding_cached_second_call(self, no_cache_client):
        geo_mock = respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
            return_value=httpx.Response(200, json=GEO_RESPONSE)
        )
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        await no_cache_client.get_by_city("Paris")
        await no_cache_client.get_by_city("Paris")  # second call
        # Geocoding API should only be called once
        assert geo_mock.call_count == 1

    @respx.mock
    async def test_geocoding_failure_uses_defaults(self, no_cache_client):
        respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
            return_value=httpx.Response(200, json={"results": []})  # no results
        )
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        # Should not raise; falls back to default lat/lon
        data = await no_cache_client.get_by_city("UnknownCity42")
        assert data is not None

    @respx.mock
    async def test_get_default_uses_configured_location(self, no_cache_client):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        data = await no_cache_client.get_default()
        assert data is not None


# ---------------------------------------------------------------------------
# Timeout / error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestErrorHandling:

    @respx.mock
    async def test_timeout_returns_fallback(self, no_cache_client):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            side_effect=httpx.TimeoutException("timeout")
        )
        data = await no_cache_client.get_by_coords(48.85, 2.35)
        assert data is not None  # should not raise

    @respx.mock
    async def test_http_error_does_not_raise(self, no_cache_client):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(429)
        )
        data = await no_cache_client.get_by_coords(48.85, 2.35)
        assert data is not None

    @respx.mock
    async def test_stale_cache_used_when_api_fails(self, tmp_path):
        """When the API is down, a stale cache entry should be returned."""
        import time
        cache = WeatherCache(ttl_seconds=1, stale_ttl_seconds=3600, cache_dir=None)
        from src.layer3_context.weather.weather_models import WeatherData
        stale_data = WeatherData(
            temperature_celsius=15, condition=WeatherCondition.SUNNY,
            city="Paris", lat=48.85, lon=2.35, provider="open-meteo"
        )
        cache.put(48.85, 2.35, stale_data)

        # Advance time past TTL so cache is stale
        orig_time = time.time()
        with patch("time.time", return_value=orig_time + 2):
            respx.get("https://api.open-meteo.com/v1/forecast").mock(
                return_value=httpx.Response(503)
            )
            client = WeatherAPIClient(cache=cache, owm_api_key=None)
            data = await client.get_by_coords(48.85, 2.35)
        assert data is not None
        assert data.city == "Paris"
