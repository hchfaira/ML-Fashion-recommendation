"""
Unit tests for WeatherModels
=============================
Tests for WeatherData, WeatherCondition and their helper methods.
No HTTP calls, no file I/O.
"""
import pytest
from datetime import datetime, timezone

from src.layer3_context.weather.weather_models import WeatherData, WeatherCondition


# ---------------------------------------------------------------------------
# WeatherCondition
# ---------------------------------------------------------------------------

class TestWeatherCondition:
    def test_string_equality(self):
        """WeatherCondition should compare equal to its string value."""
        assert WeatherCondition.SUNNY == "sunny"
        assert WeatherCondition.RAINY == "rainy"

    def test_is_wet_rainy(self):
        assert WeatherCondition.RAINY.is_wet is True

    def test_is_wet_drizzle(self):
        assert WeatherCondition.DRIZZLE.is_wet is True

    def test_is_wet_stormy(self):
        assert WeatherCondition.STORMY.is_wet is True

    def test_is_wet_sunny_false(self):
        assert WeatherCondition.SUNNY.is_wet is False

    def test_is_cold_condition_snowy(self):
        assert WeatherCondition.SNOWY.is_cold_condition is True

    def test_is_cold_condition_sunny_false(self):
        assert WeatherCondition.SUNNY.is_cold_condition is False


# ---------------------------------------------------------------------------
# WeatherData helpers
# ---------------------------------------------------------------------------

class TestWeatherDataProperties:

    def _data(self, **kw) -> WeatherData:
        defaults = dict(
            temperature_celsius=15.0,
            feels_like_celsius=15.0,
            condition=WeatherCondition.SUNNY,
        )
        defaults.update(kw)
        return WeatherData(**defaults)

    def test_is_cold_below_10(self):
        d = self._data(feels_like_celsius=8)
        assert d.is_cold is True

    def test_is_not_cold_above_10(self):
        d = self._data(feels_like_celsius=12)
        assert d.is_cold is False

    def test_is_hot_above_27(self):
        d = self._data(feels_like_celsius=30)
        assert d.is_hot is True

    def test_is_not_hot_below_27(self):
        d = self._data(feels_like_celsius=20)
        assert d.is_hot is False

    def test_is_wet_via_condition(self):
        d = self._data(condition=WeatherCondition.RAINY)
        assert d.is_wet is True

    def test_is_wet_via_precipitation(self):
        d = self._data(condition=WeatherCondition.CLOUDY, precipitation_mm=1.0)
        assert d.is_wet is True

    def test_is_not_wet_dry_cloudy(self):
        d = self._data(condition=WeatherCondition.CLOUDY, precipitation_mm=0.0)
        assert d.is_wet is False

    def test_is_windy_above_30(self):
        d = self._data(wind_speed_kmh=35)
        assert d.is_windy is True

    def test_is_not_windy_below_30(self):
        d = self._data(wind_speed_kmh=20)
        assert d.is_windy is False

    def test_is_high_uv(self):
        d = self._data(uv_index=7)
        assert d.is_high_uv is True

    def test_is_not_high_uv(self):
        d = self._data(uv_index=3)
        assert d.is_high_uv is False

    def test_is_not_high_uv_none(self):
        d = self._data(uv_index=None)
        assert d.is_high_uv is False

    @pytest.mark.parametrize("feels_like,expected", [
        (2, "freezing"),
        (8, "cold"),
        (15, "mild"),
        (22, "warm"),
        (30, "hot"),
    ])
    def test_temp_band(self, feels_like, expected):
        d = self._data(feels_like_celsius=feels_like)
        assert d.temp_band == expected

    def test_summary_contains_temp(self):
        d = self._data(temperature_celsius=18, city="Paris")
        assert "18" in d.summary
        assert "Paris" in d.summary

    def test_summary_contains_rain(self):
        d = self._data(condition=WeatherCondition.RAINY, precipitation_mm=2.5)
        assert "rain" in d.summary.lower() or "2.5" in d.summary


class TestWeatherDataSerialization:

    def test_to_dict_condition_is_string(self):
        d = WeatherData(condition=WeatherCondition.CLOUDY)
        result = d.to_dict()
        assert result["condition"] == "cloudy"
        assert isinstance(result["condition"], str)

    def test_to_dict_fetched_at_is_string(self):
        d = WeatherData()
        result = d.to_dict()
        assert isinstance(result["fetched_at"], str)
        assert "T" in result["fetched_at"]  # ISO 8601

    def test_roundtrip_model_validate(self):
        d = WeatherData(
            temperature_celsius=22.5,
            condition=WeatherCondition.SUNNY,
            city="London",
            provider="open-meteo",
        )
        raw = d.to_dict()
        restored = WeatherData.model_validate(raw)
        assert restored.temperature_celsius == 22.5
        assert restored.city == "London"
        assert restored.condition == WeatherCondition.SUNNY


class TestWeatherDataFactories:

    def test_unknown_factory(self):
        d = WeatherData.unknown(lat=48.85, lon=2.35, city="Paris")
        assert d.condition == WeatherCondition.UNKNOWN
        assert d.provider == "fallback"
        assert d.city == "Paris"
        assert d.lat == 48.85

    def test_to_weather_context_conversion(self):
        d = WeatherData(
            temperature_celsius=20.0,
            feels_like_celsius=18.0,
            condition=WeatherCondition.CLOUDY,
            humidity=65.0,
            wind_speed_kmh=12.0,
        )
        ctx = d.to_weather_context()
        assert ctx.temperature_celsius == 20.0
        assert ctx.condition == "cloudy"
        assert ctx.humidity == 65.0
        assert ctx.wind_speed_kmh == 12.0

    def test_to_weather_context_uv_forwarded(self):
        d = WeatherData(uv_index=8, condition=WeatherCondition.SUNNY,
                        temperature_celsius=28)
        ctx = d.to_weather_context()
        assert ctx.uv_index == 8
