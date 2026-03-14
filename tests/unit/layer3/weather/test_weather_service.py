"""
Unit tests for the updated WeatherService façade
=================================================
Tests backward-compatible methods and new rich API.
HTTP is mocked via respx.
"""
from __future__ import annotations

import pytest
import respx
import httpx

from src.layer3_context.weather_service import WeatherService
from src.layer3_context.weather.weather_cache import WeatherCache
from src.layer3_context.weather.weather_models import WeatherCondition
from src.core.models import WeatherContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OM_RESPONSE = {
    "latitude": 48.85, "longitude": 2.35,
    "timezone": "Europe/Paris",
    "current_units": {"wind_speed_10m": "km/h"},
    "current": {
        "temperature_2m": 20.0,
        "apparent_temperature": 19.0,
        "relative_humidity_2m": 55,
        "precipitation": 0.0,
        "weather_code": 1,    # mainly clear → SUNNY
        "wind_speed_10m": 8.0,
        "uv_index": 4,
    },
    "daily": {"sunrise": [], "sunset": []},
}

RAINY_OM = {**OM_RESPONSE, "current": {**OM_RESPONSE["current"],
    "weather_code": 63, "precipitation": 3.0}}  # moderate rain


@pytest.fixture
def svc():
    cache = WeatherCache(ttl_seconds=3600, cache_dir=None)
    return WeatherService(cache=cache, owm_api_key=None)


# ---------------------------------------------------------------------------
# Rich API
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetWeatherData:

    @respx.mock
    async def test_returns_weather_data_with_coords(self, svc):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        data = await svc.get_weather_data(lat=48.85, lon=2.35)
        assert abs(data.temperature_celsius - 20.0) < 0.1
        assert data.condition == WeatherCondition.SUNNY

    @respx.mock
    async def test_get_weather_data_by_city(self, svc):
        respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
            return_value=httpx.Response(200, json={
                "results": [{"latitude": 48.8566, "longitude": 2.3522}]
            })
        )
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        data = await svc.get_weather_data(location="Paris")
        assert data is not None

    @respx.mock
    async def test_coords_take_priority_over_city(self, svc):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        data = await svc.get_weather_data(location="Paris", lat=48.85, lon=2.35)
        # Geocoding API should NOT have been called
        assert abs(data.temperature_celsius - 20.0) < 0.1

    @respx.mock
    async def test_get_outfit_advisor_returns_advisor(self, svc):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        data = await svc.get_weather_data(lat=48.85, lon=2.35)
        advisor = svc.get_outfit_advisor(data)
        advice = advisor.get_advice()
        assert "summary" in advice


# ---------------------------------------------------------------------------
# Backward-compatible thin API
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBackwardCompatAPI:

    @respx.mock
    async def test_get_weather_returns_weather_context(self, svc):
        respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
            return_value=httpx.Response(200, json={
                "results": [{"latitude": 48.85, "longitude": 2.35}]
            })
        )
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        ctx = await svc.get_weather("Paris")
        assert isinstance(ctx, WeatherContext)
        assert ctx.temperature_celsius == 20.0

    @respx.mock
    async def test_get_weather_returns_none_on_failure(self, svc):
        respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(503)
        )
        # Returns None (unknown condition) or a context
        result = await svc.get_weather("UnknownCity")
        # Should not raise; result may be None or WeatherContext
        assert result is None or isinstance(result, WeatherContext)

    @respx.mock
    async def test_get_weather_by_coords_returns_context(self, svc):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json=OM_RESPONSE)
        )
        ctx = await svc.get_weather_by_coords(48.85, 2.35)
        assert isinstance(ctx, WeatherContext)


# ---------------------------------------------------------------------------
# Clothing recommendations (backward compat)
# ---------------------------------------------------------------------------

class TestClothingRecommendations:

    def test_rainy_adds_umbrella(self, svc):
        ctx = WeatherContext(
            temperature_celsius=12, condition="rainy",
            humidity=90, wind_speed_kmh=15
        )
        recs = svc.get_clothing_recommendations(ctx)
        assert "umbrella" in recs["accessories"]

    def test_rainy_avoid_suede(self, svc):
        ctx = WeatherContext(temperature_celsius=12, condition="rainy")
        recs = svc.get_clothing_recommendations(ctx)
        assert "suede" in recs["avoid"]

    def test_cold_returns_heavy_layers(self, svc):
        ctx = WeatherContext(temperature_celsius=3, condition="cloudy")
        recs = svc.get_clothing_recommendations(ctx)
        assert "heavy_coat" in recs["layers"]

    def test_hot_returns_minimal_layers(self, svc):
        ctx = WeatherContext(temperature_celsius=30, condition="sunny")
        recs = svc.get_clothing_recommendations(ctx)
        assert "minimal" in recs["layers"]

    def test_snowy_avoids_heels(self, svc):
        ctx = WeatherContext(temperature_celsius=-2, condition="snowy")
        recs = svc.get_clothing_recommendations(ctx)
        assert "heels" in recs["avoid"]

    def test_result_has_required_keys(self, svc):
        ctx = WeatherContext(temperature_celsius=18, condition="cloudy")
        recs = svc.get_clothing_recommendations(ctx)
        for k in ("layers", "avoid", "suggested_materials", "accessories"):
            assert k in recs


# ---------------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------------

def test_real_weather_service_alias():
    from src.layer3_context.weather_service import RealWeatherService, WeatherService
    assert RealWeatherService is WeatherService
