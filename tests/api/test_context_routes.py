"""
Tests for Context API Routes
================================

Covers:
- GET  /context/weather
- POST /context/weather/outfit-filter
- GET  /context/occasions
- POST /context/occasions/analyze
- POST /context/morphology/advice
- POST /context/morphology/filter
- POST /context/filter
- POST /context/schedule/analyze
- POST /context/rotation/analyze
- POST /context/rotation/record-wear
- POST /context/activity/filter
- GET  /context/activity/comfort-factors
- POST /context/history/record
- GET  /context/history/{user_id}
- GET  /context/history/{user_id}/favorites
- GET  /context/history/{user_id}/patterns
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, date
from fastapi import HTTPException

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory, ColorProfile,
    Occasion, UserContext, WeatherContext,
)
from src.api.routes.context import (
    get_weather,
    filter_by_weather,
    list_occasions,
    analyze_occasion,
    get_morphology_advice,
    filter_by_morphology,
    filter_wardrobe_by_context,
    analyze_schedule,
    analyze_wardrobe_rotation,
    record_item_worn,
    filter_by_activity,
    list_comfort_factors,
    record_outfit_history,
    get_user_history,
    get_favorite_outfits,
    analyze_user_patterns,
    WeatherRequest,
    ContextFilterRequest,
    MorphologyRequest,
    ScheduleEvent,
    ScheduleRequest,
    RotationRequest,
    ActivityRequest,
    UserHistoryEntry,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_garment(gid="g1", category=GarmentCategory.TOP, color="white"):
    return Garment(
        id=gid,
        attributes=GarmentAttributes(
            category=category,
            color=ColorProfile(primary=color, hex_codes=[]),
        ),
    )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_weather_service():
    weather = MagicMock()
    weather.temperature_celsius = 22
    weather.feels_like = 20
    weather.humidity = 55
    weather.wind_speed = 10
    weather.conditions = "sunny"
    weather.precipitation_chance = 0.1
    weather.uv_index = 5
    weather.clothing_recommendations = ["t-shirt", "shorts"]

    mock = MagicMock()
    mock.get_weather_by_location = AsyncMock(return_value=weather)
    mock.get_weather_by_coords = AsyncMock(return_value=weather)
    mock.filter_by_weather = MagicMock(
        side_effect=lambda wardrobe, _: wardrobe[:1]
    )
    with patch(
        "src.api.routes.context.get_weather_service", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_occasion_analyzer():
    mock = MagicMock()
    mock.get_formality_level = MagicMock(return_value="smart_casual")
    mock.get_dress_code_hints = MagicMock(return_value=["blazer", "chinos"])
    mock.rank_items_for_occasion = MagicMock(
        return_value=[(_make_garment(), 0.9, "nice color")]
    )
    with patch(
        "src.api.routes.context.get_occasion_analyzer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_morphology_advisor():
    advice = MagicMock()
    advice.recommendations = ["wear V-neck"]
    advice.flattering_silhouettes = ["A-line"]
    advice.avoid = ["boxy cuts"]
    advice.accentuate = ["waist"]
    advice.minimize = ["hips"]
    advice.best_proportions = "2/3 top, 1/3 bottom"

    mock = MagicMock()
    mock.get_advice = MagicMock(return_value=advice)
    mock.filter_flattering_items = MagicMock(
        return_value=[(_make_garment(), 0.85)]
    )
    with patch(
        "src.api.routes.context.get_morphology_advisor", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_context_engine():
    mock = MagicMock()
    mock.filter_wardrobe_by_context = AsyncMock(
        side_effect=lambda wardrobe, ctx: wardrobe
    )
    with patch(
        "src.api.routes.context.get_context_engine", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_schedule_analyzer():
    analysis = MagicMock()
    analysis.formality_range = (3, 7)
    analysis.transitions_needed = 1
    analysis.versatile_recommendations = ["blazer"]
    analysis.outfit_changes = 2
    analysis.strategy = MagicMock(value="layering")

    mock = MagicMock()
    mock.analyze_day = MagicMock(return_value=analysis)
    with patch(
        "src.api.routes.context.get_schedule_analyzer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_wardrobe_rotation():
    analysis = MagicMock()
    analysis.overused = [_make_garment("g_over")]
    analysis.neglected = [_make_garment("g_neg")]
    analysis.suggestions = [_make_garment("g_sug")]
    analysis.rotation_score = 0.6
    analysis.utilization_percent = 45.0

    mock = MagicMock()
    mock.analyze_rotation = MagicMock(return_value=analysis)
    mock.record_wear = MagicMock()
    with patch(
        "src.api.routes.context.get_wardrobe_rotation", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_activity_analyzer():
    mock = MagicMock()
    mock.filter_by_activities = MagicMock(
        return_value=[(_make_garment(), 0.9)]
    )
    with patch(
        "src.api.routes.context.get_activity_analyzer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_user_history_manager():
    patterns = MagicMock()
    patterns.preferred_colors = ["navy", "white"]
    patterns.preferred_styles = ["casual"]
    patterns.occasion_frequency = {"casual": 10}
    patterns.average_rating = 4.2
    patterns.insights = ["prefers navy"]

    mock = MagicMock()
    mock.record_outfit = MagicMock()
    mock.get_history = MagicMock(return_value=[{"outfit": "A"}])
    mock.get_favorites = MagicMock(return_value=[{"outfit": "fav"}])
    mock.analyze_patterns = MagicMock(return_value=patterns)
    with patch(
        "src.api.routes.context.get_user_history_manager", return_value=mock
    ):
        yield mock


# ============================================================================
# Weather
# ============================================================================

class TestWeather:

    @pytest.mark.asyncio
    async def test_get_weather_by_location(self, mock_weather_service):
        result = await get_weather(location="Paris")
        assert result["temperature_celsius"] == 22
        assert result["conditions"] == "sunny"

    @pytest.mark.asyncio
    async def test_get_weather_by_coords(self, mock_weather_service):
        result = await get_weather(latitude=48.8, longitude=2.3)
        assert result["temperature_celsius"] == 22

    @pytest.mark.asyncio
    async def test_get_weather_no_input_raises_400(self, mock_weather_service):
        with pytest.raises(HTTPException) as exc_info:
            await get_weather()
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_get_weather_service_error(self, mock_weather_service):
        mock_weather_service.get_weather_by_location = AsyncMock(
            side_effect=RuntimeError("api down")
        )
        with pytest.raises(HTTPException) as exc_info:
            await get_weather(location="Mars")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_filter_by_weather(self, mock_weather_service):
        wardrobe = [_make_garment("g1"), _make_garment("g2")]
        weather = WeatherContext(temperature_celsius=22, condition="sunny")
        result = await filter_by_weather(wardrobe, weather)
        assert "suitable_items" in result
        assert result["filtered_count"] >= 0


# ============================================================================
# Occasions
# ============================================================================

class TestOccasions:

    @pytest.mark.asyncio
    async def test_list_occasions(self, mock_occasion_analyzer):
        result = await list_occasions()
        assert "occasions" in result
        assert len(result["occasions"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_occasion(self, mock_occasion_analyzer):
        wardrobe = [_make_garment()]
        result = await analyze_occasion("casual", wardrobe)
        assert result["occasion"] == "casual"
        assert "suitable_items" in result

    @pytest.mark.asyncio
    async def test_analyze_occasion_invalid_raises_400(
        self, mock_occasion_analyzer
    ):
        with pytest.raises(HTTPException) as exc_info:
            await analyze_occasion("nonexistent_occasion", [])
        assert exc_info.value.status_code == 400


# ============================================================================
# Morphology
# ============================================================================

class TestMorphology:

    @pytest.mark.asyncio
    async def test_morphology_advice(self, mock_morphology_advisor):
        req = MorphologyRequest(
            body_shape="pear", height_cm=165, preferred_fit="fitted"
        )
        result = await get_morphology_advice(req)
        assert result["body_shape"] == "pear"
        assert "recommendations" in result
        assert "flattering_silhouettes" in result

    @pytest.mark.asyncio
    async def test_morphology_filter(self, mock_morphology_advisor):
        wardrobe = [_make_garment()]
        result = await filter_by_morphology(wardrobe, "pear", strictness=0.7)
        assert "suitable_items" in result


# ============================================================================
# Context filter
# ============================================================================

class TestContextFilter:

    @pytest.mark.asyncio
    async def test_filter_by_context(self, mock_context_engine):
        req = ContextFilterRequest(
            wardrobe=[_make_garment()],
            occasion="casual",
        )
        result = await filter_wardrobe_by_context(req)
        assert "filtered_items" in result
        assert result["original_count"] == 1


# ============================================================================
# Schedule
# ============================================================================

class TestSchedule:

    @pytest.mark.asyncio
    async def test_analyze_schedule(self, mock_schedule_analyzer):
        events = [
            ScheduleEvent(
                title="Morning meeting",
                start_time=datetime(2024, 6, 1, 9, 0),
                end_time=datetime(2024, 6, 1, 10, 0),
                event_type="meeting",
            )
        ]
        req = ScheduleRequest(
            events=events,
            wardrobe=[_make_garment()],
            date=date(2024, 6, 1),
        )
        result = await analyze_schedule(req)
        assert "formality_range" in result
        assert "transitions_needed" in result
        assert result["events_count"] == 1


# ============================================================================
# Wardrobe rotation
# ============================================================================

class TestWardrobeRotation:

    @pytest.mark.asyncio
    async def test_analyze_rotation(self, mock_wardrobe_rotation):
        req = RotationRequest(
            user_id="u1",
            wardrobe=[_make_garment()],
            days_to_consider=30,
        )
        result = await analyze_wardrobe_rotation(req)
        assert "overused_items" in result
        assert "neglected_items" in result
        assert "rotation_score" in result

    @pytest.mark.asyncio
    async def test_record_item_worn(self, mock_wardrobe_rotation):
        result = await record_item_worn(user_id="u1", garment_id="g1")
        assert result["status"] == "recorded"


# ============================================================================
# Activity
# ============================================================================

class TestActivity:

    @pytest.mark.asyncio
    async def test_filter_by_activity(self, mock_activity_analyzer):
        req = ActivityRequest(
            activities=["walking", "outdoor"],
            wardrobe=[_make_garment()],
            comfort_priority=0.8,
        )
        result = await filter_by_activity(req)
        assert "suitable_items" in result

    @pytest.mark.asyncio
    async def test_list_comfort_factors(self):
        result = await list_comfort_factors()
        assert "comfort_factors" in result
        assert "activity_types" in result
        assert len(result["comfort_factors"]) > 0


# ============================================================================
# User history
# ============================================================================

class TestUserHistory:

    @pytest.mark.asyncio
    async def test_record_outfit_history(self, mock_user_history_manager):
        entry = UserHistoryEntry(
            user_id="u1",
            outfit_garment_ids=["g1", "g2"],
            occasion="casual",
            date=datetime.now(),
            user_rating=4,
        )
        result = await record_outfit_history(entry)
        assert result["status"] == "recorded"

    @pytest.mark.asyncio
    async def test_get_user_history(self, mock_user_history_manager):
        result = await get_user_history(user_id="u1", limit=10)
        assert result["user_id"] == "u1"
        assert len(result["outfits"]) == 1

    @pytest.mark.asyncio
    async def test_get_favorites(self, mock_user_history_manager):
        result = await get_favorite_outfits(user_id="u1", limit=5)
        assert result["user_id"] == "u1"
        assert "favorites" in result

    @pytest.mark.asyncio
    async def test_analyze_patterns(self, mock_user_history_manager):
        result = await analyze_user_patterns(user_id="u1")
        assert result["user_id"] == "u1"
        assert "preferred_colors" in result
        assert "insights" in result
