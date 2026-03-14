"""
Unit tests for WeatherOutfitAdvisor
=====================================
Tests rule activation, penalty/boost scoring, and advice dict structure.
No HTTP calls, no file I/O.
"""
import pytest

from src.layer3_context.weather.weather_models import WeatherData, WeatherCondition
from src.layer3_context.weather.weather_outfit_advisor import WeatherOutfitAdvisor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def advisor_for(**kw) -> WeatherOutfitAdvisor:
    defaults = dict(temperature_celsius=15.0, feels_like_celsius=15.0,
                    condition=WeatherCondition.CLOUDY)
    defaults.update(kw)
    return WeatherOutfitAdvisor(WeatherData(**defaults))


# ---------------------------------------------------------------------------
# Rule activation
# ---------------------------------------------------------------------------

class TestRuleActivation:

    def test_rain_activates_rain_rule(self):
        advisor = advisor_for(condition=WeatherCondition.RAINY)
        assert "rain" in advisor._active_rules

    def test_drizzle_activates_drizzle_rule(self):
        advisor = advisor_for(condition=WeatherCondition.DRIZZLE)
        assert "drizzle" in advisor._active_rules

    def test_snow_activates_snow_rule(self):
        advisor = advisor_for(condition=WeatherCondition.SNOWY)
        assert "snow" in advisor._active_rules

    def test_stormy_activates_stormy_rule(self):
        advisor = advisor_for(condition=WeatherCondition.STORMY)
        assert "stormy" in advisor._active_rules

    def test_wind_above_30_activates_wind_rule(self):
        advisor = advisor_for(wind_speed_kmh=35)
        assert "wind_above_30kmh" in advisor._active_rules

    def test_wind_below_30_no_wind_rule(self):
        advisor = advisor_for(wind_speed_kmh=20)
        assert "wind_above_30kmh" not in advisor._active_rules

    def test_uv_above_6_activates_uv_rule(self):
        advisor = advisor_for(uv_index=7)
        assert "uv_above_6" in advisor._active_rules

    def test_uv_below_6_no_uv_rule(self):
        advisor = advisor_for(uv_index=3)
        assert "uv_above_6" not in advisor._active_rules

    def test_hot_temp_activates_hot_rule(self):
        advisor = advisor_for(feels_like_celsius=30)
        assert "hot_above_27" in advisor._active_rules

    def test_cold_temp_activates_cold_rule(self):
        advisor = advisor_for(feels_like_celsius=5)
        assert "cold_below_10" in advisor._active_rules

    def test_mild_weather_no_rules(self):
        advisor = advisor_for(condition=WeatherCondition.CLOUDY, feels_like_celsius=18,
                              wind_speed_kmh=10, uv_index=2)
        assert advisor._active_rules == []


# ---------------------------------------------------------------------------
# Penalty scoring
# ---------------------------------------------------------------------------

class TestPenaltyScoring:

    def test_suede_penalised_in_rain(self):
        advisor = advisor_for(condition=WeatherCondition.RAINY)
        penalty = advisor.penalise_item({"material": "suede"})
        assert penalty > 0

    def test_wool_not_penalised_in_rain(self):
        advisor = advisor_for(condition=WeatherCondition.RAINY)
        penalty = advisor.penalise_item({"material": "wool"})
        assert penalty == 0.0

    def test_open_toe_penalised_in_rain(self):
        advisor = advisor_for(condition=WeatherCondition.RAINY)
        penalty = advisor.penalise_item({"footwear_style": "open_toe"})
        assert penalty > 0

    def test_penalty_capped_at_1(self):
        # Worst-case: rain + snow + stormy all at once (simulate via heavy rain)
        advisor = advisor_for(condition=WeatherCondition.STORMY,
                              wind_speed_kmh=40, feels_like_celsius=2)
        penalty = advisor.penalise_item({
            "material": "suede silk linen",
            "footwear_style": "open_toe canvas",
            "silhouette": "full_skirt",
        })
        assert 0.0 <= penalty <= 1.0

    def test_no_active_rules_zero_penalty(self):
        advisor = advisor_for(condition=WeatherCondition.CLOUDY, feels_like_celsius=18)
        penalty = advisor.penalise_item({"material": "suede"})
        assert penalty == 0.0

    def test_heels_penalised_in_snow(self):
        advisor = advisor_for(condition=WeatherCondition.SNOWY)
        penalty = advisor.penalise_item({"footwear_style": "heels"})
        assert penalty > 0

    def test_linen_penalised_when_cold(self):
        advisor = advisor_for(feels_like_celsius=5)
        penalty = advisor.penalise_item({"material": "linen"})
        assert penalty > 0


# ---------------------------------------------------------------------------
# Boost scoring
# ---------------------------------------------------------------------------

class TestBoostScoring:

    def test_wool_boosted_when_cold(self):
        advisor = advisor_for(feels_like_celsius=5)
        boost = advisor.boost_item({"material": "wool"})
        assert boost > 0

    def test_linen_boosted_when_hot(self):
        advisor = advisor_for(feels_like_celsius=30)
        boost = advisor.boost_item({"material": "linen"})
        assert boost > 0

    def test_umbrella_boosted_in_rain(self):
        advisor = advisor_for(condition=WeatherCondition.RAINY)
        boost = advisor.boost_item({"accessory_type": "umbrella"})
        assert boost > 0

    def test_no_active_rules_zero_boost(self):
        advisor = advisor_for(condition=WeatherCondition.CLOUDY, feels_like_celsius=18)
        boost = advisor.boost_item({"material": "wool"})
        assert boost == 0.0

    def test_boost_capped_at_1(self):
        advisor = advisor_for(condition=WeatherCondition.RAINY,
                              feels_like_celsius=5, wind_speed_kmh=35)
        boost = advisor.boost_item({
            "material": "wool",
            "accessory_type": "umbrella scarf gloves",
            "silhouette": "fitted",
        })
        assert 0.0 <= boost <= 1.0


# ---------------------------------------------------------------------------
# get_advice() dict
# ---------------------------------------------------------------------------

class TestGetAdvice:

    def test_returns_dict_with_expected_keys(self):
        advisor = advisor_for()
        advice = advisor.get_advice()
        for key in [
            "summary", "condition", "temp_band", "active_rules",
            "penalise_materials", "penalise_footwear", "penalise_silhouettes",
            "avoid_categories", "boost_materials", "boost_accessories",
            "boost_silhouettes", "weather_summary", "feels_like",
            "temperature_celsius", "is_wet", "is_windy", "is_high_uv",
        ]:
            assert key in advice, f"Missing key: {key}"

    def test_no_rules_summary(self):
        advisor = advisor_for(condition=WeatherCondition.CLOUDY, feels_like_celsius=18)
        advice = advisor.get_advice()
        assert "no special" in advice["summary"].lower()

    def test_rain_penalise_materials_contains_suede(self):
        advisor = advisor_for(condition=WeatherCondition.RAINY)
        advice = advisor.get_advice()
        assert "suede" in advice["penalise_materials"]

    def test_rain_boost_accessories_contains_umbrella(self):
        advisor = advisor_for(condition=WeatherCondition.RAINY)
        advice = advisor.get_advice()
        assert "umbrella" in advice["boost_accessories"]

    def test_cold_boost_materials_contains_wool(self):
        advisor = advisor_for(feels_like_celsius=5)
        advice = advisor.get_advice()
        assert "wool" in advice["boost_materials"]

    def test_is_wet_true_in_rainy(self):
        advisor = advisor_for(condition=WeatherCondition.RAINY, precipitation_mm=2.0)
        advice = advisor.get_advice()
        assert advice["is_wet"] is True

    def test_is_windy_true_above_30(self):
        advisor = advisor_for(wind_speed_kmh=40)
        advice = advisor.get_advice()
        assert advice["is_windy"] is True

    def test_condition_string_in_advice(self):
        advisor = advisor_for(condition=WeatherCondition.SUNNY)
        advice = advisor.get_advice()
        assert advice["condition"] == "sunny"

    def test_temp_band_warm(self):
        advisor = advisor_for(feels_like_celsius=22)
        advice = advisor.get_advice()
        assert advice["temp_band"] == "warm"

    def test_multiple_rules_union_of_lists(self):
        """Rain + cold: both rule sets should be merged in penalise_materials."""
        advisor = advisor_for(condition=WeatherCondition.RAINY, feels_like_celsius=5)
        advice = advisor.get_advice()
        # "rain" penalises suede; "cold_below_10" penalises linen
        assert "suede" in advice["penalise_materials"]
        assert "linen" in advice["penalise_materials"]
