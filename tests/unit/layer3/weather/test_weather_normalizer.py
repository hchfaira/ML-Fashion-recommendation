"""
Unit tests for WeatherNormalizer
================================
Exercises both Open-Meteo and OWM parsers with representative payloads.
No HTTP calls.
"""
import pytest
from src.layer3_context.weather.weather_normalizer import WeatherNormalizer
from src.layer3_context.weather.weather_models import WeatherCondition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normalizer():
    return WeatherNormalizer()


# Representative Open-Meteo response with `current` block (v2 API)
OPEN_METEO_CURRENT = {
    "latitude": 48.85,
    "longitude": 2.35,
    "timezone": "Europe/Paris",
    "current_units": {
        "temperature_2m": "°C",
        "apparent_temperature": "°C",
        "relative_humidity_2m": "%",
        "precipitation": "mm",
        "weather_code": "wmo code",
        "wind_speed_10m": "km/h",
        "uv_index": "",
    },
    "current": {
        "temperature_2m": 14.3,
        "apparent_temperature": 12.1,
        "relative_humidity_2m": 72,
        "precipitation": 0.0,
        "weather_code": 3,          # overcast
        "wind_speed_10m": 18.0,
        "uv_index": 2,
    },
    "daily": {
        "sunrise": ["2024-10-15T07:45"],
        "sunset": ["2024-10-15T18:30"],
    },
}

# Older Open-Meteo response with `current_weather` block
OPEN_METEO_CW = {
    "latitude": 48.85,
    "longitude": 2.35,
    "current_weather": {
        "temperature": 11.0,
        "windspeed": 25.0,
        "weathercode": 61,   # slight rain
    },
}

# OpenWeatherMap /data/2.5/weather response
OWM_RESPONSE = {
    "coord": {"lat": 51.51, "lon": -0.13},
    "weather": [{"id": 500, "main": "Rain", "description": "light rain"}],
    "main": {
        "temp": 10.5,
        "feels_like": 8.3,
        "humidity": 85,
    },
    "wind": {"speed": 5.0, "deg": 200},   # m/s
    "rain": {"1h": 0.4},
    "sys": {"sunrise": 1697358600, "sunset": 1697397600},
    "name": "London",
}


# ---------------------------------------------------------------------------
# Open-Meteo normalizer
# ---------------------------------------------------------------------------

class TestOpenMeteoNormalizer:

    def test_condition_from_wmo_3_overcast(self, normalizer):
        data = normalizer.from_open_meteo(OPEN_METEO_CURRENT, lat=48.85, lon=2.35)
        assert data.condition == WeatherCondition.CLOUDY

    def test_temperature_parsed(self, normalizer):
        data = normalizer.from_open_meteo(OPEN_METEO_CURRENT, lat=48.85, lon=2.35)
        assert abs(data.temperature_celsius - 14.3) < 0.1

    def test_feels_like_parsed(self, normalizer):
        data = normalizer.from_open_meteo(OPEN_METEO_CURRENT, lat=48.85, lon=2.35)
        assert abs(data.feels_like_celsius - 12.1) < 0.1

    def test_humidity_parsed(self, normalizer):
        data = normalizer.from_open_meteo(OPEN_METEO_CURRENT, lat=48.85, lon=2.35)
        assert data.humidity == 72.0

    def test_wind_speed_parsed_kmh(self, normalizer):
        """When unit is km/h (default for current_weather block), no conversion."""
        data = normalizer.from_open_meteo(OPEN_METEO_CURRENT, lat=48.85, lon=2.35)
        assert data.wind_speed_kmh == 18.0

    def test_uv_index_parsed(self, normalizer):
        data = normalizer.from_open_meteo(OPEN_METEO_CURRENT, lat=48.85, lon=2.35)
        assert data.uv_index == 2

    def test_uv_capped_at_11(self, normalizer):
        raw = {**OPEN_METEO_CURRENT, "current": {**OPEN_METEO_CURRENT["current"], "uv_index": 15}}
        data = normalizer.from_open_meteo(raw, lat=0, lon=0)
        assert data.uv_index == 11

    def test_sunrise_captured(self, normalizer):
        data = normalizer.from_open_meteo(OPEN_METEO_CURRENT, lat=48.85, lon=2.35)
        assert data.sunrise == "2024-10-15T07:45"

    def test_provider_is_open_meteo(self, normalizer):
        data = normalizer.from_open_meteo(OPEN_METEO_CURRENT, lat=48.85, lon=2.35)
        assert data.provider == "open-meteo"

    def test_city_attached(self, normalizer):
        data = normalizer.from_open_meteo(OPEN_METEO_CURRENT, lat=48.85, lon=2.35, city="Paris")
        assert data.city == "Paris"

    def test_coords_stored(self, normalizer):
        data = normalizer.from_open_meteo(OPEN_METEO_CURRENT, lat=48.85, lon=2.35)
        assert data.lat == 48.85
        assert data.lon == 2.35

    def test_wmo_code_61_rain(self, normalizer):
        data = normalizer.from_open_meteo(OPEN_METEO_CW, lat=48.85, lon=2.35)
        assert data.condition == WeatherCondition.RAINY

    def test_legacy_current_weather_block(self, normalizer):
        data = normalizer.from_open_meteo(OPEN_METEO_CW, lat=48.85, lon=2.35)
        assert abs(data.temperature_celsius - 11.0) < 0.1

    @pytest.mark.parametrize("wmo,expected", [
        (0, WeatherCondition.SUNNY),
        (1, WeatherCondition.SUNNY),
        (2, WeatherCondition.CLOUDY),
        (45, WeatherCondition.FOGGY),
        (51, WeatherCondition.DRIZZLE),
        (63, WeatherCondition.RAINY),
        (71, WeatherCondition.SNOWY),
        (95, WeatherCondition.STORMY),
    ])
    def test_wmo_mapping(self, normalizer, wmo, expected):
        raw = {"current": {"weather_code": wmo, "temperature_2m": 15},
               "current_units": {}}
        data = normalizer.from_open_meteo(raw, lat=0, lon=0)
        assert data.condition == expected

    def test_unknown_wmo_defaults_to_cloudy(self, normalizer):
        raw = {"current": {"weather_code": 999, "temperature_2m": 15},
               "current_units": {}}
        data = normalizer.from_open_meteo(raw, lat=0, lon=0)
        assert data.condition == WeatherCondition.CLOUDY


# ---------------------------------------------------------------------------
# OWM normalizer
# ---------------------------------------------------------------------------

class TestOWMNormalizer:

    def test_condition_rain(self, normalizer):
        data = normalizer.from_openweathermap(OWM_RESPONSE)
        assert data.condition == WeatherCondition.RAINY

    def test_temperature_parsed(self, normalizer):
        data = normalizer.from_openweathermap(OWM_RESPONSE)
        assert abs(data.temperature_celsius - 10.5) < 0.1

    def test_feels_like_parsed(self, normalizer):
        data = normalizer.from_openweathermap(OWM_RESPONSE)
        assert abs(data.feels_like_celsius - 8.3) < 0.1

    def test_wind_converted_ms_to_kmh(self, normalizer):
        """OWM gives wind in m/s; normalizer converts to km/h."""
        data = normalizer.from_openweathermap(OWM_RESPONSE)
        assert abs(data.wind_speed_kmh - 18.0) < 0.5   # 5 m/s * 3.6 = 18

    def test_humidity_parsed(self, normalizer):
        data = normalizer.from_openweathermap(OWM_RESPONSE)
        assert data.humidity == 85.0

    def test_precipitation_from_rain_block(self, normalizer):
        data = normalizer.from_openweathermap(OWM_RESPONSE)
        assert abs(data.precipitation_mm - 0.4) < 0.01

    def test_city_from_name_field(self, normalizer):
        data = normalizer.from_openweathermap(OWM_RESPONSE)
        assert data.city == "London"

    def test_city_override(self, normalizer):
        data = normalizer.from_openweathermap(OWM_RESPONSE, city="Greater London")
        assert data.city == "Greater London"

    def test_coords_stored(self, normalizer):
        data = normalizer.from_openweathermap(OWM_RESPONSE)
        assert abs(data.lat - 51.51) < 0.01
        assert abs(data.lon - (-0.13)) < 0.01

    def test_provider_is_owm(self, normalizer):
        data = normalizer.from_openweathermap(OWM_RESPONSE)
        assert data.provider == "openweathermap"

    def test_sunrise_stored(self, normalizer):
        data = normalizer.from_openweathermap(OWM_RESPONSE)
        assert data.sunrise == 1697358600  # Unix timestamp

    @pytest.mark.parametrize("main,expected", [
        ("Clear", WeatherCondition.SUNNY),
        ("Clouds", WeatherCondition.CLOUDY),
        ("Rain", WeatherCondition.RAINY),
        ("Drizzle", WeatherCondition.DRIZZLE),
        ("Snow", WeatherCondition.SNOWY),
        ("Thunderstorm", WeatherCondition.STORMY),
        ("Mist", WeatherCondition.FOGGY),
        ("Fog", WeatherCondition.FOGGY),
    ])
    def test_owm_condition_mapping(self, normalizer, main, expected):
        raw = {"weather": [{"main": main}], "main": {"temp": 15},
               "wind": {}, "coord": {}}
        data = normalizer.from_openweathermap(raw)
        assert data.condition == expected

    def test_unknown_condition_defaults_cloudy(self, normalizer):
        raw = {"weather": [{"main": "Alien"}], "main": {"temp": 15},
               "wind": {}, "coord": {}}
        data = normalizer.from_openweathermap(raw)
        assert data.condition == WeatherCondition.CLOUDY
