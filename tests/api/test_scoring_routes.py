"""
Tests for Scoring API Routes
================================

Covers:
- GET  /scoring/profiles
- GET  /scoring/profiles/{profile_name}
- GET  /scoring/profiles/{profile_name}/weights
- POST /scoring/profiles/reload
- POST /scoring/scorecard
- POST /scoring/build-outfits
- POST /scoring/total-style
- POST /scoring/seven-point-rule
- POST /scoring/season-color-harmony
- POST /scoring/proportion-analysis
- POST /scoring/volume-balance
- POST /scoring/pattern-mixing
- POST /scoring/design-principles
- POST /scoring/creativity
- POST /scoring/three-color-rule
- POST /scoring/sandwich-rule
- POST /scoring/occasion-fit
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory, ColorProfile, Occasion,
)

from src.api.routes.scoring import (
    list_scoring_profiles,
    get_scoring_profile,
    get_profile_weights,
    reload_profiles,
    get_outfit_scorecard,
    build_outfits,
    get_total_style_score,
    check_seven_point_rule,
    check_season_color_harmony,
    analyze_proportions,
    check_volume_balance,
    analyze_pattern_mixing,
    analyze_design_principles,
    score_creativity,
    check_three_color_rule,
    check_sandwich_rule,
    check_occasion_fit,
    ScorecardRequest,
    OutfitBuildRequest,
    SevenPointRequest,
    SeasonColorRequest,
    PatternMixingRequest,
    DesignPrinciplesRequest,
    CreativityRequest,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_garments(n=3):
    cats = [GarmentCategory.TOP, GarmentCategory.BOTTOM, GarmentCategory.SHOES]
    colors = ["white", "navy", "brown"]
    garments = []
    for i in range(n):
        garments.append(
            Garment(
                id=f"g{i}",
                attributes=GarmentAttributes(
                    category=cats[i % len(cats)],
                    color=ColorProfile(primary=colors[i % len(colors)], hex_codes=[]),
                ),
            )
        )
    return garments


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_scoring_config_service():
    """Mock get_scoring_config_service and reload_scoring_config."""
    profile = MagicMock()
    profile.description = "Default profile"
    profile.criteria = {
        "color_harmony": MagicMock(enabled=True, weight=1.0,
                                    display_name="Color Harmony",
                                    description="Color compatibility",
                                    show_in_summary=True, show_details=True),
        "formality": MagicMock(enabled=True, weight=0.8,
                               display_name="Formality",
                               description="Formality match",
                               show_in_summary=True, show_details=False),
        "creativity": MagicMock(enabled=False, weight=0.5,
                                display_name="Creativity",
                                description="Creative scoring",
                                show_in_summary=False, show_details=False),
    }

    mock = MagicMock()
    mock.list_profiles.return_value = ["default", "business", "creative"]
    mock.get_profile.return_value = profile
    mock.normalize_weights.return_value = {
        "color_harmony": 0.556,
        "formality": 0.444,
    }

    with patch(
        "src.api.routes.scoring.get_scoring_config_service", return_value=mock
    ):
        with patch(
            "src.api.routes.scoring.reload_scoring_config", return_value=mock
        ):
            yield mock


@pytest.fixture
def mock_scorecard():
    """Mock OutfitScorecard."""
    result = MagicMock()
    result.to_json = MagicMock(return_value={"overall": 0.8, "scores": {}})

    mock = MagicMock(return_value=result)
    result.calculate_all_scores.return_value = result
    with patch("src.api.routes.scoring.OutfitScorecard", mock):
        yield mock


@pytest.fixture
def mock_outfit_builder():
    candidate = MagicMock()
    candidate.garments = _make_garments(2)
    candidate.total_score = 0.82
    candidate.score_breakdown = {"color": 0.9}

    mock = MagicMock()
    mock_instance = MagicMock()
    mock_instance.build_outfits.return_value = [candidate]
    mock.return_value = mock_instance
    with patch("src.api.routes.scoring.OutfitBuilder", mock):
        yield mock_instance


@pytest.fixture
def mock_total_style_scorer():
    report = MagicMock()
    report.grade = "A"
    report.total_score = 0.88
    report.breakdown = {"color": 0.9}
    report.strengths = ["great color harmony"]
    report.improvements = ["try accessorizing"]
    report.summary = "Excellent outfit"

    mock = MagicMock()
    mock.score_outfit.return_value = report
    with patch(
        "src.api.routes.scoring.get_total_style_scorer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_seven_point_scorer():
    result = MagicMock()
    result.score = 0.85
    result.current_points = 6
    result.target_points = 7
    result.is_balanced = True
    result.suggestions = ["add a watch"]

    mock = MagicMock()
    mock.score.return_value = result
    with patch(
        "src.api.routes.scoring.get_seven_point_scorer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_season_color_scorer():
    result = MagicMock()
    result.score = 0.9
    result.harmony_level = "high"
    result.matching_colors = ["navy"]
    result.clashing_colors = []
    result.recommendations = ["add warm accents"]

    mock = MagicMock()
    mock.score.return_value = result
    with patch(
        "src.api.routes.scoring.get_season_color_scorer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_proportion_scorer():
    result = MagicMock()
    result.score = 0.75
    result.ratio = MagicMock(value="2:1")
    result.is_balanced = True
    result.weight_distribution = {"top": 0.5, "bottom": 0.5}
    result.suggestions = []

    mock = MagicMock()
    mock.score.return_value = result
    with patch(
        "src.api.routes.scoring.get_proportion_scorer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_volume_balance_scorer():
    result = MagicMock()
    result.score = 0.8
    result.top_volume = MagicMock(value="medium")
    result.bottom_volume = MagicMock(value="medium")
    result.is_balanced = True
    result.recommendations = ["well balanced"]

    mock = MagicMock()
    mock.score.return_value = result
    with patch(
        "src.api.routes.scoring.get_volume_balance_scorer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_pattern_mixing_scorer():
    result = MagicMock()
    result.score = 0.7
    result.pattern_count = 2
    result.is_compatible = True
    result.patterns = [MagicMock(model_dump=MagicMock(return_value={"type": "stripes"}))]
    result.rules_followed = ["scale contrast"]
    result.suggestions = []

    mock = MagicMock()
    mock.score.return_value = result
    with patch(
        "src.api.routes.scoring.get_pattern_mixing_scorer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_design_principles_scorer():
    result = MagicMock()
    result.score = 0.85
    result.grade = "A"
    result.balance = MagicMock(model_dump=MagicMock(return_value={"symmetry": "balanced"}))
    result.proportion = MagicMock(model_dump=MagicMock(return_value={}))
    result.rhythm = MagicMock(model_dump=MagicMock(return_value={}))
    result.emphasis = MagicMock(model_dump=MagicMock(return_value={}))
    result.harmony = MagicMock(model_dump=MagicMock(return_value={}))
    result.recommendations = ["add focal point"]

    mock = MagicMock()
    mock.score.return_value = result
    with patch(
        "src.api.routes.scoring.get_design_principles_scorer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_creativity_scorer():
    result = MagicMock()
    result.score = 0.6
    result.level = MagicMock(value="moderate")
    result.rule_breaks = []
    result.is_intentional = True
    result.fashion_forward_elements = ["asymmetric hem"]
    result.summary = "Decent creativity"

    mock = MagicMock()
    mock.score.return_value = result
    with patch(
        "src.api.routes.scoring.get_creativity_scorer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_three_color_scorer():
    result = MagicMock()
    result.score = 0.95
    result.color_count = 3
    result.colors = ["white", "navy", "brown"]
    result.follows_rule = True
    result.suggestions = []

    mock = MagicMock()
    mock.score.return_value = result
    with patch(
        "src.api.routes.scoring.get_three_color_scorer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_sandwich_rule_scorer():
    result = MagicMock()
    result.score = 0.8
    result.has_sandwich = True
    result.sandwich_color = "white"
    result.connecting_elements = ["top", "shoes"]
    result.suggestions = []

    mock = MagicMock()
    mock.score.return_value = result
    with patch(
        "src.api.routes.scoring.get_sandwich_rule_scorer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_occasion_scorer():
    result = MagicMock()
    result.score = 0.85
    result.is_appropriate = True
    result.outfit_formality = "smart_casual"
    result.occasion_formality = "casual"
    result.formality_gap = 0
    result.suggestions = []

    mock = MagicMock()
    mock.score.return_value = result
    with patch(
        "src.api.routes.scoring.get_occasion_scorer", return_value=mock
    ):
        yield mock


# ============================================================================
# Profile management
# ============================================================================

class TestListProfiles:

    @pytest.mark.asyncio
    async def test_list_profiles(self, mock_scoring_config_service):
        result = await list_scoring_profiles()
        assert "profiles" in result
        assert result["total"] == 3

    @pytest.mark.asyncio
    async def test_list_profiles_service_error(self, mock_scoring_config_service):
        mock_scoring_config_service.list_profiles.side_effect = RuntimeError("err")
        with pytest.raises(HTTPException) as exc_info:
            await list_scoring_profiles()
        assert exc_info.value.status_code == 500


class TestGetProfile:

    @pytest.mark.asyncio
    async def test_get_existing_profile(self, mock_scoring_config_service):
        result = await get_scoring_profile("default")
        assert result["name"] == "default"
        assert "criteria" in result
        assert "enabled_count" in result

    @pytest.mark.asyncio
    async def test_get_missing_profile_raises_404(
        self, mock_scoring_config_service
    ):
        mock_scoring_config_service.get_profile.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            await get_scoring_profile("nonexistent")
        assert exc_info.value.status_code == 404


class TestGetProfileWeights:

    @pytest.mark.asyncio
    async def test_normalized_weights(self, mock_scoring_config_service):
        result = await get_profile_weights("default", normalized=True)
        assert result["normalized"] is True
        assert "weights" in result

    @pytest.mark.asyncio
    async def test_raw_weights(self, mock_scoring_config_service):
        result = await get_profile_weights("default", normalized=False)
        assert result["normalized"] is False

    @pytest.mark.asyncio
    async def test_missing_profile_raises_404(
        self, mock_scoring_config_service
    ):
        mock_scoring_config_service.get_profile.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            await get_profile_weights("nope")
        assert exc_info.value.status_code == 404


class TestReloadProfiles:

    @pytest.mark.asyncio
    async def test_reload_success(self, mock_scoring_config_service):
        result = await reload_profiles()
        assert result["status"] == "success"
        assert "available_profiles" in result


# ============================================================================
# Scorecard
# ============================================================================

class TestScorecard:

    @pytest.mark.asyncio
    async def test_scorecard_success(self, mock_scorecard):
        garments = _make_garments(2)
        req = ScorecardRequest(garments=garments)
        result = await get_outfit_scorecard(req)
        assert "overall" in result

    @pytest.mark.asyncio
    async def test_scorecard_needs_at_least_2(self):
        garments = _make_garments(1)
        req = ScorecardRequest(garments=garments)
        with pytest.raises(HTTPException) as exc_info:
            await get_outfit_scorecard(req)
        assert exc_info.value.status_code == 400


# ============================================================================
# Build outfits
# ============================================================================

class TestBuildOutfits:

    @pytest.mark.asyncio
    async def test_build_outfits_success(self, mock_outfit_builder):
        garments = _make_garments(3)
        req = OutfitBuildRequest(wardrobe=garments, max_outfits=3)
        result = await build_outfits(req)
        assert "outfits" in result
        assert result["total_generated"] == 1

    @pytest.mark.asyncio
    async def test_build_outfits_needs_at_least_2(self):
        req = OutfitBuildRequest(wardrobe=_make_garments(1), max_outfits=1)
        with pytest.raises(HTTPException) as exc_info:
            await build_outfits(req)
        assert exc_info.value.status_code == 400


# ============================================================================
# Individual scorers
# ============================================================================

class TestTotalStyleScore:

    @pytest.mark.asyncio
    async def test_success(self, mock_total_style_scorer):
        result = await get_total_style_score(_make_garments(2))
        assert result["grade"] == "A"
        assert result["total_score"] == 0.88

    @pytest.mark.asyncio
    async def test_needs_at_least_2(self):
        with pytest.raises(HTTPException) as exc_info:
            await get_total_style_score(_make_garments(1))
        assert exc_info.value.status_code == 400


class TestSevenPointRule:

    @pytest.mark.asyncio
    async def test_success(self, mock_seven_point_scorer):
        req = SevenPointRequest(garments=_make_garments(3))
        result = await check_seven_point_rule(req)
        assert result["score"] == 0.85
        assert result["is_balanced"] is True


class TestSeasonColorHarmony:

    @pytest.mark.asyncio
    async def test_success(self, mock_season_color_scorer):
        req = SeasonColorRequest(
            garments=_make_garments(2),
            skin_undertone="warm",
            color_season="autumn",
        )
        result = await check_season_color_harmony(req)
        assert result["score"] == 0.9
        assert result["harmony_level"] == "high"


class TestProportionAnalysis:

    @pytest.mark.asyncio
    async def test_success(self, mock_proportion_scorer):
        result = await analyze_proportions(_make_garments(2))
        assert result["score"] == 0.75
        assert result["is_balanced"] is True


class TestVolumeBalance:

    @pytest.mark.asyncio
    async def test_success(self, mock_volume_balance_scorer):
        result = await check_volume_balance(_make_garments(2))
        assert result["score"] == 0.8
        assert result["is_balanced"] is True


class TestPatternMixing:

    @pytest.mark.asyncio
    async def test_success(self, mock_pattern_mixing_scorer):
        req = PatternMixingRequest(garments=_make_garments(2))
        result = await analyze_pattern_mixing(req)
        assert result["score"] == 0.7
        assert result["is_compatible"] is True


class TestDesignPrinciples:

    @pytest.mark.asyncio
    async def test_success(self, mock_design_principles_scorer):
        req = DesignPrinciplesRequest(garments=_make_garments(2))
        result = await analyze_design_principles(req)
        assert result["score"] == 0.85
        assert result["grade"] == "A"


class TestCreativity:

    @pytest.mark.asyncio
    async def test_success(self, mock_creativity_scorer):
        req = CreativityRequest(garments=_make_garments(2))
        result = await score_creativity(req)
        assert result["score"] == 0.6
        assert result["is_intentional"] is True


class TestThreeColorRule:

    @pytest.mark.asyncio
    async def test_success(self, mock_three_color_scorer):
        result = await check_three_color_rule(_make_garments(3))
        assert result["score"] == 0.95
        assert result["follows_rule"] is True


class TestSandwichRule:

    @pytest.mark.asyncio
    async def test_success(self, mock_sandwich_rule_scorer):
        result = await check_sandwich_rule(_make_garments(3))
        assert result["score"] == 0.8
        assert result["has_sandwich"] is True


class TestOccasionFit:

    @pytest.mark.asyncio
    async def test_success(self, mock_occasion_scorer):
        result = await check_occasion_fit(_make_garments(2), "casual")
        assert result["score"] == 0.85
        assert result["is_appropriate"] is True

    @pytest.mark.asyncio
    async def test_invalid_occasion_raises_400(self, mock_occasion_scorer):
        with pytest.raises(HTTPException) as exc_info:
            await check_occasion_fit(_make_garments(2), "invalid_occasion")
        assert exc_info.value.status_code == 400
