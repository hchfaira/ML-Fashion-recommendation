"""
Unit Tests for OutfitImprover
================================

Tests for:
- Outfit diagnosis
- Addition suggestions
- Replacement suggestions
- Purchase suggestions
- Removal impact analysis
- Full improvement pipeline
"""

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from typing import List

from src.core.models import (
    Garment,
    GarmentAttributes,
    GarmentCategory,
    ColorProfile,
    FormalityLevel,
    Season,
    Occasion,
    UserContext,
    PatternInfo,
    MaterialProfile,
    SeasonalityInfo,
)
from src.layer2_style.outfit_improver import OutfitImprover
from src.layer2_style.outfit_scorecard import OutfitScorecard


# ============================================================================
# Fixtures
# ============================================================================

def _g(
    garment_id=None,
    category=GarmentCategory.TOP,
    subcategory="t-shirt",
    color="white",
    formality=FormalityLevel.CASUAL,
    seasons=None,
    pattern_type="solid",
    material="cotton",
) -> Garment:
    """Shortcut for creating a test garment."""
    season_list = seasons or [Season.SPRING, Season.SUMMER]
    return Garment(
        id=garment_id or f"g_{uuid4().hex[:8]}",
        attributes=GarmentAttributes(
            category=category,
            subcategory=subcategory,
            color=ColorProfile(primary=color, hex_codes=[]),
            formality_level=formality,
            season_suitable=season_list,
            seasonality=SeasonalityInfo(seasons=season_list),
            pattern=PatternInfo(type=pattern_type),
            material=MaterialProfile(primary=material),
        ),
    )


def _outfit() -> List[Garment]:
    """Basic 3-piece outfit."""
    return [
        _g(garment_id="top1", category=GarmentCategory.TOP, color="white"),
        _g(garment_id="bottom1", category=GarmentCategory.BOTTOM, subcategory="jeans", color="blue"),
        _g(garment_id="shoes1", category=GarmentCategory.SHOES, subcategory="sneakers", color="white"),
    ]


def _wardrobe() -> List[Garment]:
    """A wardrobe with alternatives."""
    return [
        _g(garment_id="top1", category=GarmentCategory.TOP, color="white"),
        _g(garment_id="top2", category=GarmentCategory.TOP, color="navy"),
        _g(garment_id="top3", category=GarmentCategory.TOP, color="black", formality=FormalityLevel.BUSINESS),
        _g(garment_id="bottom1", category=GarmentCategory.BOTTOM, subcategory="jeans", color="blue"),
        _g(garment_id="bottom2", category=GarmentCategory.BOTTOM, subcategory="chinos", color="khaki", formality=FormalityLevel.SMART_CASUAL),
        _g(garment_id="shoes1", category=GarmentCategory.SHOES, subcategory="sneakers", color="white"),
        _g(garment_id="shoes2", category=GarmentCategory.SHOES, subcategory="loafers", color="brown", formality=FormalityLevel.SMART_CASUAL),
        _g(garment_id="jacket1", category=GarmentCategory.OUTERWEAR, subcategory="denim_jacket", color="blue"),
        _g(garment_id="acc1", category=GarmentCategory.ACCESSORY, subcategory="watch", color="silver"),
        _g(garment_id="acc2", category=GarmentCategory.ACCESSORY, subcategory="belt", color="brown"),
    ]


def _mock_scorecard(scores=None, garments=None):
    """Create a mock OutfitScorecard."""
    sc = MagicMock(spec=OutfitScorecard)
    sc.scores = scores or {
        "seven_point": 0.6,
        "color_harmony": 0.4,
        "three_color": 0.8,
        "proportion": 0.7,
        "volume_balance": 0.5,
        "pattern_mixing": 0.9,
        "design_principles": 0.6,
        "creativity": 0.3,
        "overall": 0.6,
    }
    sc.garments = garments or _outfit()

    def get_grade(score=None):
        s = score if score is not None else sc.scores.get("overall", 0)
        if s >= 0.9:
            return "A+"
        elif s >= 0.7:
            return "B"
        elif s >= 0.5:
            return "C"
        else:
            return "D"

    sc.get_grade = get_grade
    return sc


@pytest.fixture
def improver():
    return OutfitImprover()


# ============================================================================
# Diagnosis
# ============================================================================

class TestDiagnosis:

    def test_diagnose_identifies_weak_dims(self, improver):
        sc = _mock_scorecard()
        diag = improver.diagnose_outfit(sc)
        weak_names = [d["dimension"] for d in diag.weak_dimensions]
        assert "color_harmony" in weak_names  # 0.4 < 0.5
        assert "creativity" in weak_names  # 0.3 < 0.5

    def test_diagnose_identifies_strong_dims(self, improver):
        sc = _mock_scorecard()
        diag = improver.diagnose_outfit(sc)
        strong_names = [d["dimension"] for d in diag.strong_dimensions]
        assert "three_color" in strong_names  # 0.8 >= 0.7
        assert "pattern_mixing" in strong_names  # 0.9 >= 0.7

    def test_diagnose_overall_score(self, improver):
        sc = _mock_scorecard()
        diag = improver.diagnose_outfit(sc)
        assert diag.overall_score == 0.6
        assert diag.grade == "C"

    def test_diagnose_improvement_potential(self, improver):
        sc = _mock_scorecard()
        diag = improver.diagnose_outfit(sc)
        assert 0 < diag.improvement_potential <= 1.0
        assert abs(diag.improvement_potential - 0.4) < 0.01

    def test_diagnose_perfect_outfit(self, improver):
        sc = _mock_scorecard(scores={
            "seven_point": 0.95,
            "color_harmony": 0.9,
            "three_color": 0.92,
            "proportion": 0.88,
            "volume_balance": 0.85,
            "pattern_mixing": 0.9,
            "design_principles": 0.87,
            "creativity": 0.8,
            "overall": 0.9,
        })
        diag = improver.diagnose_outfit(sc)
        assert len(diag.weak_dimensions) == 0
        assert len(diag.strong_dimensions) >= 5

    def test_diagnose_weak_sorted(self, improver):
        sc = _mock_scorecard()
        diag = improver.diagnose_outfit(sc)
        if len(diag.weak_dimensions) >= 2:
            scores = [d["score"] for d in diag.weak_dimensions]
            assert scores == sorted(scores)  # ascending


# ============================================================================
# Addition Suggestions
# ============================================================================

class TestAdditions:

    def test_suggest_additions_returns_list(self, improver):
        outfit = _outfit()
        wardrobe = _wardrobe()
        additions = improver.suggest_additions(outfit, wardrobe)
        assert isinstance(additions, list)

    def test_suggest_additions_excludes_outfit_items(self, improver):
        outfit = _outfit()
        wardrobe = _wardrobe()
        additions = improver.suggest_additions(outfit, wardrobe)
        outfit_ids = {g.id for g in outfit}
        for a in additions:
            assert a.garment_id not in outfit_ids

    def test_suggest_additions_limited_count(self, improver):
        outfit = _outfit()
        wardrobe = _wardrobe()
        additions = improver.suggest_additions(outfit, wardrobe, max_suggestions=2)
        assert len(additions) <= 2

    def test_suggest_additions_sorted_by_delta(self, improver):
        outfit = _outfit()
        wardrobe = _wardrobe()
        additions = improver.suggest_additions(outfit, wardrobe)
        if len(additions) >= 2:
            deltas = [a.expected_score_change for a in additions]
            assert deltas == sorted(deltas, reverse=True)

    def test_suggest_additions_have_description(self, improver):
        outfit = _outfit()
        wardrobe = _wardrobe()
        additions = improver.suggest_additions(outfit, wardrobe)
        for a in additions:
            assert a.garment_description != ""
            assert a.reason != ""

    def test_suggest_additions_empty_wardrobe(self, improver):
        outfit = _outfit()
        additions = improver.suggest_additions(outfit, [])
        assert additions == []


# ============================================================================
# Replacement Suggestions
# ============================================================================

class TestReplacements:

    def test_suggest_replacements_returns_list(self, improver):
        outfit = _outfit()
        wardrobe = _wardrobe()
        replacements = improver.suggest_replacements(outfit, wardrobe)
        assert isinstance(replacements, list)

    def test_replacements_same_category(self, improver):
        outfit = _outfit()
        wardrobe = _wardrobe()
        replacements = improver.suggest_replacements(outfit, wardrobe)
        wardrobe_dict = {g.id: g for g in wardrobe}
        outfit_dict = {g.id: g for g in outfit}
        for r in replacements:
            orig = outfit_dict.get(r.original_garment_id)
            repl = wardrobe_dict.get(r.replacement_garment_id)
            if orig and repl:
                assert orig.attributes.category == repl.attributes.category

    def test_replacements_positive_delta(self, improver):
        outfit = _outfit()
        wardrobe = _wardrobe()
        replacements = improver.suggest_replacements(outfit, wardrobe)
        for r in replacements:
            assert r.expected_score_change > 0

    def test_replacements_limited_count(self, improver):
        outfit = _outfit()
        wardrobe = _wardrobe()
        replacements = improver.suggest_replacements(outfit, wardrobe, max_suggestions=1)
        assert len(replacements) <= 1

    def test_replacements_empty_wardrobe(self, improver):
        outfit = _outfit()
        replacements = improver.suggest_replacements(outfit, [])
        assert replacements == []


# ============================================================================
# Purchase Suggestions
# ============================================================================

class TestPurchases:

    def test_purchase_for_weak_dims(self, improver):
        sc = _mock_scorecard()
        purchases = improver.suggest_purchases(_outfit(), sc)
        assert len(purchases) > 0
        # Should target weak dimensions
        categories = [p.category for p in purchases]
        assert len(categories) > 0

    def test_purchase_expected_change_positive(self, improver):
        sc = _mock_scorecard()
        purchases = improver.suggest_purchases(_outfit(), sc)
        for p in purchases:
            assert p.expected_score_change > 0

    def test_purchase_limited_count(self, improver):
        sc = _mock_scorecard()
        purchases = improver.suggest_purchases(_outfit(), sc, max_suggestions=1)
        assert len(purchases) <= 1

    def test_purchase_no_weak_dims(self, improver):
        sc = _mock_scorecard(scores={
            "seven_point": 0.8,
            "color_harmony": 0.8,
            "three_color": 0.8,
            "proportion": 0.8,
            "volume_balance": 0.8,
            "pattern_mixing": 0.8,
            "design_principles": 0.8,
            "creativity": 0.8,
            "overall": 0.8,
        })
        purchases = improver.suggest_purchases(_outfit(), sc)
        assert purchases == []

    def test_purchase_has_attributes(self, improver):
        sc = _mock_scorecard()
        purchases = improver.suggest_purchases(_outfit(), sc)
        for p in purchases:
            assert p.description != ""
            assert p.reason != ""


# ============================================================================
# Removal Impact
# ============================================================================

class TestRemovalImpact:

    def test_removal_impact_existing_garment(self, improver):
        wardrobe = _wardrobe()
        result = improver.analyze_removal_impact("top1", wardrobe)
        assert result.garment_id == "top1"
        assert result.outfits_affected >= 0
        assert 0 <= result.versatility_score <= 1
        assert result.impact_level in ["low", "medium", "high", "critical"]

    def test_removal_impact_nonexistent(self, improver):
        wardrobe = _wardrobe()
        result = improver.analyze_removal_impact("does_not_exist", wardrobe)
        assert result.outfits_affected == 0
        assert result.impact_level == "low"

    def test_removal_replacement_available(self, improver):
        """top1 should have replacement since top2 and top3 exist."""
        wardrobe = _wardrobe()
        result = improver.analyze_removal_impact("top1", wardrobe)
        assert result.replacement_available is True
        assert len(result.replacement_suggestions) >= 1

    def test_removal_no_replacement(self, improver):
        """Only one outerwear — should have no replacement."""
        wardrobe = _wardrobe()
        result = improver.analyze_removal_impact("jacket1", wardrobe)
        assert result.replacement_available is False

    def test_removal_summary_not_empty(self, improver):
        wardrobe = _wardrobe()
        result = improver.analyze_removal_impact("top1", wardrobe)
        assert result.summary != ""

    def test_removal_critical_for_unique_versatile(self, improver):
        """Removing the only item of a versatile category."""
        # Only one outerwear and it's versatile
        wardrobe = [
            _g(garment_id="jacket_only", category=GarmentCategory.OUTERWEAR, color="black",
               formality=FormalityLevel.SMART_CASUAL, seasons=[Season.FALL, Season.WINTER]),
            _g(category=GarmentCategory.TOP, color="white"),
            _g(category=GarmentCategory.BOTTOM, color="blue"),
            _g(category=GarmentCategory.SHOES, color="black"),
        ]
        result = improver.analyze_removal_impact("jacket_only", wardrobe)
        assert result.replacement_available is False
        assert result.impact_level in ["high", "critical"]


# ============================================================================
# Full Improvement Pipeline
# ============================================================================

class TestFullImprovement:

    def test_improve_returns_complete_result(self, improver):
        outfit = _outfit()
        wardrobe = _wardrobe()
        result = improver.improve(outfit, wardrobe)
        assert result.diagnosis is not None
        assert 0 <= result.diagnosis.overall_score <= 1
        assert isinstance(result.additions, list)
        assert isinstance(result.replacements, list)
        assert isinstance(result.purchase_suggestions, list)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_improve_empty_outfit(self, improver):
        result = improver.improve([], _wardrobe())
        assert result.diagnosis.overall_score == 0.0
        assert result.diagnosis.grade == "F"

    def test_improve_with_context(self, improver):
        outfit = _outfit()
        wardrobe = _wardrobe()
        ctx = UserContext(
            occasion=Occasion.CASUAL,
            body_shape="rectangle",
        )
        result = improver.improve(outfit, wardrobe, context=ctx)
        assert result.diagnosis is not None

    def test_improve_with_profile(self, improver):
        outfit = _outfit()
        wardrobe = _wardrobe()
        result = improver.improve(outfit, wardrobe, profile="casual")
        assert result.diagnosis is not None

    def test_improve_summary_mentions_score(self, improver):
        outfit = _outfit()
        wardrobe = _wardrobe()
        result = improver.improve(outfit, wardrobe)
        assert "%" in result.summary or "score" in result.summary.lower() or "Score" in result.summary
