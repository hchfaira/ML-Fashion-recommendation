"""
Tests for Wardrobe Analysis API Routes
========================================

Covers:
- POST /wardrobe-analysis/analyze           (full wardrobe analysis)
- POST /wardrobe-analysis/improve           (outfit improvement)
- POST /wardrobe-analysis/impact-removal    (removal impact analysis)
- POST /wardrobe-analysis/simulate-addition (what-if simulation)
"""

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.core.models import (
    Garment,
    GarmentAttributes,
    GarmentCategory,
    ColorProfile,
    FormalityLevel,
    Season,
    Occasion,
    UserContext,
    WardrobeDistribution,
    WardrobeGap,
    OccasionCoverage,
    GarmentVersatility,
    PurchaseSuggestion,
    WardrobeAnalysisResult,
    OutfitDiagnosis,
    AdditionSuggestion,
    ReplacementSuggestion,
    PurchaseTargeted,
    OutfitImprovementResult,
    RemovalImpact,
    CategoryDistribution,
)

from src.api.routes.wardrobe_analysis import (
    analyze_wardrobe,
    improve_outfit,
    analyze_removal_impact,
    simulate_addition,
    WardrobeAnalysisRequest,
    OutfitImprovementRequest,
    RemovalImpactRequest,
    SimulateAdditionRequest,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_garment(
    garment_id=None,
    category=GarmentCategory.TOP,
    subcategory="t-shirt",
    color="white",
    formality=FormalityLevel.CASUAL,
    seasons=None,
) -> Garment:
    """Create a minimal test garment."""
    return Garment(
        id=garment_id or f"g_{uuid4().hex[:8]}",
        attributes=GarmentAttributes(
            category=category,
            subcategory=subcategory,
            color=ColorProfile(primary=color, hex_codes=[]),
            formality_level=formality,
            season_suitable=seasons or [Season.SPRING, Season.SUMMER],
        ),
    )


def _make_wardrobe():
    """Create a realistic test wardrobe."""
    return [
        _make_garment(garment_id="top1", category=GarmentCategory.TOP, color="white"),
        _make_garment(garment_id="top2", category=GarmentCategory.TOP, color="navy"),
        _make_garment(garment_id="top3", category=GarmentCategory.TOP, color="black", formality=FormalityLevel.BUSINESS),
        _make_garment(garment_id="bottom1", category=GarmentCategory.BOTTOM, subcategory="jeans", color="blue"),
        _make_garment(garment_id="bottom2", category=GarmentCategory.BOTTOM, subcategory="trousers", color="grey", formality=FormalityLevel.BUSINESS),
        _make_garment(garment_id="shoes1", category=GarmentCategory.SHOES, subcategory="sneakers", color="white"),
        _make_garment(garment_id="shoes2", category=GarmentCategory.SHOES, subcategory="loafers", color="brown", formality=FormalityLevel.SMART_CASUAL),
        _make_garment(garment_id="jacket1", category=GarmentCategory.OUTERWEAR, subcategory="blazer", color="black", formality=FormalityLevel.BUSINESS),
        _make_garment(garment_id="acc1", category=GarmentCategory.ACCESSORY, subcategory="watch", color="silver"),
    ]


def _make_analysis_result():
    """Create a mock WardrobeAnalysisResult."""
    return WardrobeAnalysisResult(
        distribution=WardrobeDistribution(
            by_category=[CategoryDistribution(category="top", count=3, percentage=33.3)],
            total_items=9,
        ),
        gaps=[
            WardrobeGap(
                gap_type="category_missing",
                severity="medium",
                description="No dress items",
                recommendation="Add a dress",
            )
        ],
        occasion_coverage=[
            OccasionCoverage(
                occasion="daily_wear",
                coverage_score=0.8,
                suitable_items_count=5,
            )
        ],
        top_versatile_items=[
            GarmentVersatility(
                garment_id="top1",
                garment_description="white t-shirt",
                versatility_score=0.9,
                compatible_outfit_count=10,
            )
        ],
        purchase_suggestions=[
            PurchaseSuggestion(
                priority=1,
                category="dress",
                description="Add a dress",
                reason="No dresses in wardrobe",
            )
        ],
        overall_score=0.72,
        summary="Good wardrobe with room for improvement.",
    )


def _make_improvement_result():
    """Create a mock OutfitImprovementResult."""
    return OutfitImprovementResult(
        diagnosis=OutfitDiagnosis(
            overall_score=0.65,
            grade="B-",
            weak_dimensions=[{"dimension": "color_harmony", "score": 0.4}],
            strong_dimensions=[{"dimension": "proportion", "score": 0.8}],
            improvement_potential=0.35,
        ),
        additions=[
            AdditionSuggestion(
                garment_id="acc1",
                garment_description="silver watch",
                expected_score_change=0.05,
                reason="Adding an accessory improves score",
            )
        ],
        replacements=[
            ReplacementSuggestion(
                original_garment_id="top1",
                original_description="white t-shirt",
                replacement_garment_id="top2",
                replacement_description="navy t-shirt",
                expected_score_change=0.08,
                reason="Better color harmony",
            )
        ],
        purchase_suggestions=[
            PurchaseTargeted(
                category="accessory",
                description="A complementary accessory",
                reason="Color harmony is low",
                expected_score_change=0.1,
            )
        ],
        summary="Score 65%. Weak: Color Harmony.",
    )


def _make_removal_impact():
    """Create a mock RemovalImpact."""
    return RemovalImpact(
        garment_id="top1",
        garment_description="white t-shirt",
        outfits_affected=5,
        versatility_score=0.85,
        replacement_available=True,
        replacement_suggestions=["navy t-shirt"],
        impact_level="medium",
        summary="Removing 'white t-shirt' would affect ~5 outfits.",
    )


# ============================================================================
# /wardrobe-analysis/analyze
# ============================================================================

class TestAnalyzeWardrobe:
    """Tests for POST /wardrobe-analysis/analyze."""

    @pytest.mark.asyncio
    async def test_analyze_success(self):
        """Full analysis with a valid wardrobe."""
        mock_result = _make_analysis_result()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_wardrobe_analyzer",
            return_value=mock_analyzer,
        ):
            request = WardrobeAnalysisRequest(wardrobe=_make_wardrobe())
            result = await analyze_wardrobe(request)

        assert result.overall_score == 0.72
        assert result.distribution.total_items == 9
        assert len(result.gaps) == 1
        mock_analyzer.analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_with_occasions(self):
        """Analysis with specific occasions."""
        mock_result = _make_analysis_result()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_wardrobe_analyzer",
            return_value=mock_analyzer,
        ):
            request = WardrobeAnalysisRequest(
                wardrobe=_make_wardrobe(),
                occasions=["daily_wear", "business"],
            )
            result = await analyze_wardrobe(request)

        call_args = mock_analyzer.analyze.call_args
        assert call_args.kwargs.get("occasions") is not None

    @pytest.mark.asyncio
    async def test_analyze_with_context(self):
        """Analysis with user context."""
        mock_result = _make_analysis_result()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_wardrobe_analyzer",
            return_value=mock_analyzer,
        ):
            ctx = UserContext(
                occasion=Occasion.BUSINESS,
                style_preferences=["classic"],
            )
            request = WardrobeAnalysisRequest(
                wardrobe=_make_wardrobe(),
                context=ctx,
            )
            result = await analyze_wardrobe(request)

        call_args = mock_analyzer.analyze.call_args
        assert call_args.kwargs.get("context") is not None

    @pytest.mark.asyncio
    async def test_analyze_empty_wardrobe(self):
        """Analysis of empty wardrobe."""
        mock_result = WardrobeAnalysisResult(
            distribution=WardrobeDistribution(),
            overall_score=0.0,
            summary="Empty wardrobe.",
        )
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_wardrobe_analyzer",
            return_value=mock_analyzer,
        ):
            request = WardrobeAnalysisRequest(wardrobe=[])
            result = await analyze_wardrobe(request)

        assert result.overall_score == 0.0

    @pytest.mark.asyncio
    async def test_analyze_invalid_occasion_ignored(self):
        """Invalid occasion strings are silently skipped."""
        mock_result = _make_analysis_result()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_wardrobe_analyzer",
            return_value=mock_analyzer,
        ):
            request = WardrobeAnalysisRequest(
                wardrobe=_make_wardrobe(),
                occasions=["daily_wear", "INVALID_OCCASION"],
            )
            result = await analyze_wardrobe(request)

        # Should still succeed; invalid occasion is just skipped
        assert result.overall_score == 0.72

    @pytest.mark.asyncio
    async def test_analyze_custom_top_k(self):
        """Custom top_k_versatile parameter."""
        mock_result = _make_analysis_result()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_wardrobe_analyzer",
            return_value=mock_analyzer,
        ):
            request = WardrobeAnalysisRequest(
                wardrobe=_make_wardrobe(),
                top_k_versatile=10,
            )
            result = await analyze_wardrobe(request)

        call_args = mock_analyzer.analyze.call_args
        assert call_args.kwargs.get("top_k_versatile") == 10

    @pytest.mark.asyncio
    async def test_analyze_internal_error(self):
        """Internal error returns 500."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = RuntimeError("boom")

        with patch(
            "src.api.routes.wardrobe_analysis.get_wardrobe_analyzer",
            return_value=mock_analyzer,
        ):
            request = WardrobeAnalysisRequest(wardrobe=_make_wardrobe())
            with pytest.raises(Exception) as exc_info:
                await analyze_wardrobe(request)
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_analyze_value_error_returns_400(self):
        """ValueError returns 400."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = ValueError("bad input")

        with patch(
            "src.api.routes.wardrobe_analysis.get_wardrobe_analyzer",
            return_value=mock_analyzer,
        ):
            request = WardrobeAnalysisRequest(wardrobe=_make_wardrobe())
            with pytest.raises(Exception) as exc_info:
                await analyze_wardrobe(request)
            assert exc_info.value.status_code == 400


# ============================================================================
# /wardrobe-analysis/improve
# ============================================================================

class TestImproveOutfit:
    """Tests for POST /wardrobe-analysis/improve."""

    @pytest.mark.asyncio
    async def test_improve_success(self):
        """Successful outfit improvement."""
        mock_result = _make_improvement_result()
        mock_improver = MagicMock()
        mock_improver.improve.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_outfit_improver",
            return_value=mock_improver,
        ):
            wardrobe = _make_wardrobe()
            request = OutfitImprovementRequest(
                outfit_garments=wardrobe[:3],
                wardrobe=wardrobe,
            )
            result = await improve_outfit(request)

        assert result.diagnosis.overall_score == 0.65
        assert len(result.additions) == 1
        assert len(result.replacements) == 1
        mock_improver.improve.assert_called_once()

    @pytest.mark.asyncio
    async def test_improve_with_context(self):
        """Improvement with user context."""
        mock_result = _make_improvement_result()
        mock_improver = MagicMock()
        mock_improver.improve.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_outfit_improver",
            return_value=mock_improver,
        ):
            ctx = UserContext(occasion=Occasion.DATE)
            wardrobe = _make_wardrobe()
            request = OutfitImprovementRequest(
                outfit_garments=wardrobe[:2],
                wardrobe=wardrobe,
                context=ctx,
            )
            result = await improve_outfit(request)

        call_args = mock_improver.improve.call_args
        assert call_args.kwargs.get("context") is not None

    @pytest.mark.asyncio
    async def test_improve_with_profile(self):
        """Improvement with scoring profile."""
        mock_result = _make_improvement_result()
        mock_improver = MagicMock()
        mock_improver.improve.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_outfit_improver",
            return_value=mock_improver,
        ):
            wardrobe = _make_wardrobe()
            request = OutfitImprovementRequest(
                outfit_garments=wardrobe[:2],
                wardrobe=wardrobe,
                profile="creative",
            )
            result = await improve_outfit(request)

        call_args = mock_improver.improve.call_args
        assert call_args.kwargs.get("profile") == "creative"

    @pytest.mark.asyncio
    async def test_improve_empty_outfit(self):
        """Improvement of empty outfit."""
        mock_result = OutfitImprovementResult(
            diagnosis=OutfitDiagnosis(
                overall_score=0.0, grade="F", improvement_potential=1.0,
            ),
            summary="No garments provided.",
        )
        mock_improver = MagicMock()
        mock_improver.improve.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_outfit_improver",
            return_value=mock_improver,
        ):
            request = OutfitImprovementRequest(
                outfit_garments=[],
                wardrobe=_make_wardrobe(),
            )
            result = await improve_outfit(request)

        assert result.diagnosis.overall_score == 0.0

    @pytest.mark.asyncio
    async def test_improve_no_wardrobe(self):
        """Improvement without wardrobe (no replacements possible)."""
        mock_result = OutfitImprovementResult(
            diagnosis=OutfitDiagnosis(
                overall_score=0.5, grade="C-", improvement_potential=0.5,
            ),
            summary="Limited options.",
        )
        mock_improver = MagicMock()
        mock_improver.improve.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_outfit_improver",
            return_value=mock_improver,
        ):
            request = OutfitImprovementRequest(
                outfit_garments=_make_wardrobe()[:3],
            )
            result = await improve_outfit(request)

        assert result.diagnosis.grade == "C-"

    @pytest.mark.asyncio
    async def test_improve_internal_error(self):
        """Internal error returns 500."""
        mock_improver = MagicMock()
        mock_improver.improve.side_effect = RuntimeError("crash")

        with patch(
            "src.api.routes.wardrobe_analysis.get_outfit_improver",
            return_value=mock_improver,
        ):
            request = OutfitImprovementRequest(
                outfit_garments=_make_wardrobe()[:2],
                wardrobe=_make_wardrobe(),
            )
            with pytest.raises(Exception) as exc_info:
                await improve_outfit(request)
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_improve_value_error_returns_400(self):
        """ValueError returns 400."""
        mock_improver = MagicMock()
        mock_improver.improve.side_effect = ValueError("invalid")

        with patch(
            "src.api.routes.wardrobe_analysis.get_outfit_improver",
            return_value=mock_improver,
        ):
            request = OutfitImprovementRequest(
                outfit_garments=_make_wardrobe()[:2],
                wardrobe=_make_wardrobe(),
            )
            with pytest.raises(Exception) as exc_info:
                await improve_outfit(request)
            assert exc_info.value.status_code == 400


# ============================================================================
# /wardrobe-analysis/impact-removal
# ============================================================================

class TestRemovalImpact:
    """Tests for POST /wardrobe-analysis/impact-removal."""

    @pytest.mark.asyncio
    async def test_removal_impact_success(self):
        """Successful removal impact analysis."""
        mock_result = _make_removal_impact()
        mock_improver = MagicMock()
        mock_improver.analyze_removal_impact.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_outfit_improver",
            return_value=mock_improver,
        ):
            request = RemovalImpactRequest(
                garment_id="top1",
                wardrobe=_make_wardrobe(),
            )
            result = await analyze_removal_impact(request)

        assert result.garment_id == "top1"
        assert result.outfits_affected == 5
        assert result.impact_level == "medium"
        mock_improver.analyze_removal_impact.assert_called_once()

    @pytest.mark.asyncio
    async def test_removal_impact_nonexistent_garment(self):
        """Removal of non-existent garment."""
        mock_result = RemovalImpact(
            garment_id="nonexistent",
            garment_description="Unknown garment",
            outfits_affected=0,
            versatility_score=0.0,
            replacement_available=False,
            impact_level="low",
            summary="Garment nonexistent not found.",
        )
        mock_improver = MagicMock()
        mock_improver.analyze_removal_impact.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_outfit_improver",
            return_value=mock_improver,
        ):
            request = RemovalImpactRequest(
                garment_id="nonexistent",
                wardrobe=_make_wardrobe(),
            )
            result = await analyze_removal_impact(request)

        assert result.outfits_affected == 0
        assert result.impact_level == "low"

    @pytest.mark.asyncio
    async def test_removal_impact_with_context(self):
        """Removal impact with context."""
        mock_result = _make_removal_impact()
        mock_improver = MagicMock()
        mock_improver.analyze_removal_impact.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_outfit_improver",
            return_value=mock_improver,
        ):
            ctx = UserContext(occasion=Occasion.WORK)
            request = RemovalImpactRequest(
                garment_id="top1",
                wardrobe=_make_wardrobe(),
                context=ctx,
            )
            result = await analyze_removal_impact(request)

        call_args = mock_improver.analyze_removal_impact.call_args
        assert call_args.kwargs.get("context") is not None

    @pytest.mark.asyncio
    async def test_removal_impact_internal_error(self):
        """Internal error returns 500."""
        mock_improver = MagicMock()
        mock_improver.analyze_removal_impact.side_effect = RuntimeError("fail")

        with patch(
            "src.api.routes.wardrobe_analysis.get_outfit_improver",
            return_value=mock_improver,
        ):
            request = RemovalImpactRequest(
                garment_id="top1",
                wardrobe=_make_wardrobe(),
            )
            with pytest.raises(Exception) as exc_info:
                await analyze_removal_impact(request)
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_removal_impact_value_error(self):
        """ValueError returns 400."""
        mock_improver = MagicMock()
        mock_improver.analyze_removal_impact.side_effect = ValueError("bad")

        with patch(
            "src.api.routes.wardrobe_analysis.get_outfit_improver",
            return_value=mock_improver,
        ):
            request = RemovalImpactRequest(
                garment_id="top1",
                wardrobe=_make_wardrobe(),
            )
            with pytest.raises(Exception) as exc_info:
                await analyze_removal_impact(request)
            assert exc_info.value.status_code == 400


# ============================================================================
# /wardrobe-analysis/simulate-addition
# ============================================================================

class TestSimulateAddition:
    """Tests for POST /wardrobe-analysis/simulate-addition."""

    @pytest.mark.asyncio
    async def test_simulate_success(self):
        """Successful addition simulation."""
        mock_result = {
            "garment_description": "red dress",
            "versatility_score": 0.7,
            "new_outfit_combinations": 8,
            "gaps_before": 2,
            "gaps_after": 1,
            "gaps_resolved": 1,
            "recommendation": "Highly recommended",
        }
        mock_analyzer = MagicMock()
        mock_analyzer.simulate_addition.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_wardrobe_analyzer",
            return_value=mock_analyzer,
        ):
            new_garment = _make_garment(
                category=GarmentCategory.DRESS, subcategory="cocktail_dress", color="red",
            )
            request = SimulateAdditionRequest(
                wardrobe=_make_wardrobe(),
                virtual_garment=new_garment,
            )
            result = await simulate_addition(request)

        assert result["versatility_score"] == 0.7
        assert result["gaps_resolved"] == 1
        mock_analyzer.simulate_addition.assert_called_once()

    @pytest.mark.asyncio
    async def test_simulate_with_context(self):
        """Simulation with user context."""
        mock_result = {"recommendation": "Good addition"}
        mock_analyzer = MagicMock()
        mock_analyzer.simulate_addition.return_value = mock_result

        with patch(
            "src.api.routes.wardrobe_analysis.get_wardrobe_analyzer",
            return_value=mock_analyzer,
        ):
            ctx = UserContext(occasion=Occasion.FORMAL)
            new_garment = _make_garment(category=GarmentCategory.DRESS, color="black")
            request = SimulateAdditionRequest(
                wardrobe=_make_wardrobe(),
                virtual_garment=new_garment,
                context=ctx,
            )
            result = await simulate_addition(request)

        call_args = mock_analyzer.simulate_addition.call_args
        assert call_args.kwargs.get("context") is not None

    @pytest.mark.asyncio
    async def test_simulate_internal_error(self):
        """Internal error returns 500."""
        mock_analyzer = MagicMock()
        mock_analyzer.simulate_addition.side_effect = RuntimeError("oops")

        with patch(
            "src.api.routes.wardrobe_analysis.get_wardrobe_analyzer",
            return_value=mock_analyzer,
        ):
            new_garment = _make_garment(category=GarmentCategory.DRESS)
            request = SimulateAdditionRequest(
                wardrobe=_make_wardrobe(),
                virtual_garment=new_garment,
            )
            with pytest.raises(Exception) as exc_info:
                await simulate_addition(request)
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_simulate_value_error(self):
        """ValueError returns 400."""
        mock_analyzer = MagicMock()
        mock_analyzer.simulate_addition.side_effect = ValueError("bad")

        with patch(
            "src.api.routes.wardrobe_analysis.get_wardrobe_analyzer",
            return_value=mock_analyzer,
        ):
            new_garment = _make_garment(category=GarmentCategory.DRESS)
            request = SimulateAdditionRequest(
                wardrobe=_make_wardrobe(),
                virtual_garment=new_garment,
            )
            with pytest.raises(Exception) as exc_info:
                await simulate_addition(request)
            assert exc_info.value.status_code == 400
