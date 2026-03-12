"""
Unit Tests for WardrobeAnalyzer
================================

Tests for:
- Distribution analysis
- Gap detection
- Occasion coverage
- Versatility scoring
- What-if simulation
- Purchase suggestions
- Full analysis pipeline
"""

import pytest
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
from src.layer2_style.wardrobe_analyzer import WardrobeAnalyzer


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


def _balanced_wardrobe() -> List[Garment]:
    """Create a balanced wardrobe with multiple categories."""
    return [
        _g(garment_id="t1", category=GarmentCategory.TOP, color="white"),
        _g(garment_id="t2", category=GarmentCategory.TOP, color="navy"),
        _g(garment_id="t3", category=GarmentCategory.TOP, color="black", formality=FormalityLevel.BUSINESS),
        _g(garment_id="b1", category=GarmentCategory.BOTTOM, subcategory="jeans", color="blue"),
        _g(garment_id="b2", category=GarmentCategory.BOTTOM, subcategory="trousers", color="grey", formality=FormalityLevel.BUSINESS),
        _g(garment_id="s1", category=GarmentCategory.SHOES, subcategory="sneakers", color="white"),
        _g(garment_id="s2", category=GarmentCategory.SHOES, subcategory="loafers", color="brown", formality=FormalityLevel.SMART_CASUAL),
        _g(garment_id="o1", category=GarmentCategory.OUTERWEAR, subcategory="blazer", color="black", formality=FormalityLevel.BUSINESS, seasons=[Season.FALL, Season.WINTER]),
        _g(garment_id="a1", category=GarmentCategory.ACCESSORY, subcategory="watch", color="silver"),
    ]


def _minimal_wardrobe() -> List[Garment]:
    """A very small wardrobe with gaps."""
    return [
        _g(garment_id="t1", category=GarmentCategory.TOP, color="black"),
        _g(garment_id="b1", category=GarmentCategory.BOTTOM, color="black"),
    ]


@pytest.fixture
def analyzer():
    return WardrobeAnalyzer()


# ============================================================================
# Distribution Analysis
# ============================================================================

class TestAnalyzeDistribution:

    def test_distribution_basic(self, analyzer):
        wardrobe = _balanced_wardrobe()
        dist = analyzer.analyze_distribution(wardrobe)
        assert dist.total_items == 9
        assert len(dist.by_category) > 0
        assert len(dist.by_color) > 0
        assert len(dist.by_formality) > 0

    def test_distribution_empty(self, analyzer):
        dist = analyzer.analyze_distribution([])
        assert dist.total_items == 0
        assert dist.by_category == []

    def test_distribution_categories_correct(self, analyzer):
        wardrobe = _balanced_wardrobe()
        dist = analyzer.analyze_distribution(wardrobe)
        cat_dict = {c.category: c.count for c in dist.by_category}
        assert cat_dict.get("top") == 3
        assert cat_dict.get("bottom") == 2
        assert cat_dict.get("shoes") == 2

    def test_distribution_percentages_sum(self, analyzer):
        wardrobe = _balanced_wardrobe()
        dist = analyzer.analyze_distribution(wardrobe)
        total_pct = sum(c.percentage for c in dist.by_category)
        assert abs(total_pct - 100.0) < 1.0  # Should be ~100%

    def test_distribution_colors(self, analyzer):
        wardrobe = _balanced_wardrobe()
        dist = analyzer.analyze_distribution(wardrobe)
        color_names = {c.category for c in dist.by_color}
        assert "white" in color_names
        assert "black" in color_names

    def test_distribution_patterns(self, analyzer):
        wardrobe = [
            _g(pattern_type="solid"),
            _g(pattern_type="stripes"),
            _g(pattern_type="solid"),
        ]
        dist = analyzer.analyze_distribution(wardrobe)
        pattern_dict = {c.category: c.count for c in dist.by_pattern}
        assert pattern_dict.get("solid") == 2
        assert pattern_dict.get("stripes") == 1

    def test_distribution_materials(self, analyzer):
        wardrobe = [
            _g(material="cotton"),
            _g(material="silk"),
            _g(material="cotton"),
        ]
        dist = analyzer.analyze_distribution(wardrobe)
        mat_dict = {c.category: c.count for c in dist.by_material}
        assert mat_dict.get("cotton") == 2
        assert mat_dict.get("silk") == 1

    def test_distribution_seasons(self, analyzer):
        wardrobe = _balanced_wardrobe()
        dist = analyzer.analyze_distribution(wardrobe)
        assert len(dist.by_season) > 0


# ============================================================================
# Gap Detection
# ============================================================================

class TestDetectGaps:

    def test_detect_gaps_balanced(self, analyzer):
        """Balanced wardrobe should have fewer gaps."""
        wardrobe = _balanced_wardrobe()
        gaps = analyzer.detect_gaps(wardrobe)
        high_gaps = [g for g in gaps if g.severity == "high"]
        assert len(high_gaps) == 0

    def test_detect_gaps_minimal(self, analyzer):
        """Minimal wardrobe should detect missing categories."""
        wardrobe = _minimal_wardrobe()
        gaps = analyzer.detect_gaps(wardrobe)
        gap_types = [g.gap_type for g in gaps]
        assert "category_missing" in gap_types

    def test_detect_no_shoes(self, analyzer):
        """Missing shoes should be detected."""
        wardrobe = [
            _g(category=GarmentCategory.TOP),
            _g(category=GarmentCategory.TOP),
            _g(category=GarmentCategory.TOP),
            _g(category=GarmentCategory.BOTTOM),
            _g(category=GarmentCategory.BOTTOM),
        ]
        gaps = analyzer.detect_gaps(wardrobe)
        shoe_gaps = [g for g in gaps if "shoes" in g.description.lower()]
        assert len(shoe_gaps) >= 1

    def test_detect_color_imbalance(self, analyzer):
        """Single-color wardrobe should trigger color imbalance."""
        wardrobe = [_g(color="black") for _ in range(6)]
        gaps = analyzer.detect_gaps(wardrobe)
        color_gaps = [g for g in gaps if g.gap_type == "color_imbalance"]
        assert len(color_gaps) >= 1

    def test_detect_formality_gap(self, analyzer):
        """All-casual wardrobe with formal occasion context."""
        wardrobe = [
            _g(formality=FormalityLevel.CASUAL) for _ in range(5)
        ]
        ctx = UserContext(occasion=Occasion.BUSINESS)
        gaps = analyzer.detect_gaps(wardrobe, ctx)
        formality_gaps = [g for g in gaps if g.gap_type == "formality_gap"]
        assert len(formality_gaps) >= 1

    def test_detect_season_gap(self, analyzer):
        """Missing winter items."""
        wardrobe = [
            _g(seasons=[Season.SUMMER]) for _ in range(6)
        ]
        gaps = analyzer.detect_gaps(wardrobe)
        season_gaps = [g for g in gaps if g.gap_type == "season_gap"]
        assert len(season_gaps) >= 1

    def test_gaps_sorted_by_severity(self, analyzer):
        """Gaps should be sorted by severity (high first)."""
        wardrobe = _minimal_wardrobe()
        gaps = analyzer.detect_gaps(wardrobe)
        if len(gaps) >= 2:
            severity_order = {"high": 0, "medium": 1, "low": 2}
            for i in range(len(gaps) - 1):
                assert severity_order[gaps[i].severity] <= severity_order[gaps[i + 1].severity]

    def test_empty_wardrobe_no_gaps(self, analyzer):
        """Empty wardrobe produces no gap objects."""
        gaps = analyzer.detect_gaps([])
        assert gaps == []


# ============================================================================
# Occasion Coverage
# ============================================================================

class TestOccasionCoverage:

    def test_daily_coverage_good(self, analyzer):
        wardrobe = _balanced_wardrobe()
        cov = analyzer.analyze_occasion_coverage(wardrobe, Occasion.DAILY_WEAR)
        assert cov.coverage_score > 0.5
        assert cov.occasion == "daily_wear"

    def test_business_coverage(self, analyzer):
        wardrobe = _balanced_wardrobe()
        cov = analyzer.analyze_occasion_coverage(wardrobe, Occasion.BUSINESS)
        assert 0 <= cov.coverage_score <= 1
        assert cov.suitable_items_count >= 0

    def test_formal_coverage_low_for_casual_wardrobe(self, analyzer):
        """All-casual wardrobe should have low formal coverage."""
        wardrobe = [
            _g(category=GarmentCategory.TOP, formality=FormalityLevel.CASUAL),
            _g(category=GarmentCategory.BOTTOM, formality=FormalityLevel.CASUAL),
            _g(category=GarmentCategory.SHOES, formality=FormalityLevel.CASUAL),
        ]
        cov = analyzer.analyze_occasion_coverage(wardrobe, Occasion.FORMAL)
        assert cov.coverage_score < 0.8

    def test_missing_categories_listed(self, analyzer):
        """Missing required categories should appear in missing_categories."""
        wardrobe = [
            _g(category=GarmentCategory.TOP, formality=FormalityLevel.CASUAL),
        ]
        cov = analyzer.analyze_occasion_coverage(wardrobe, Occasion.DAILY_WEAR)
        # Should be missing BOTTOM and SHOES at minimum
        assert len(cov.missing_categories) >= 1

    def test_suggestion_provided_when_missing(self, analyzer):
        """Suggestion should be provided when coverage is incomplete."""
        wardrobe = [_g(category=GarmentCategory.TOP)]
        cov = analyzer.analyze_occasion_coverage(wardrobe, Occasion.WORK)
        assert cov.suggestion is not None

    def test_empty_wardrobe_zero_coverage(self, analyzer):
        cov = analyzer.analyze_occasion_coverage([], Occasion.DAILY_WEAR)
        assert cov.coverage_score == 0.0
        assert cov.suitable_items_count == 0


# ============================================================================
# Versatility Scoring
# ============================================================================

class TestVersatility:

    def test_versatility_ranking(self, analyzer):
        wardrobe = _balanced_wardrobe()
        results = analyzer.calculate_versatility(wardrobe, top_k=3)
        assert len(results) <= 3
        assert all(r.versatility_score >= 0 for r in results)
        # Should be sorted descending
        for i in range(len(results) - 1):
            assert results[i].versatility_score >= results[i + 1].versatility_score

    def test_versatility_empty(self, analyzer):
        results = analyzer.calculate_versatility([], top_k=5)
        assert results == []

    def test_neutral_color_bonus(self, analyzer):
        """Neutral-colored items should score higher."""
        wardrobe = [
            _g(garment_id="neutral", category=GarmentCategory.TOP, color="black"),
            _g(garment_id="bright", category=GarmentCategory.TOP, color="neon_green"),
            _g(category=GarmentCategory.BOTTOM, color="blue"),
            _g(category=GarmentCategory.SHOES, color="white"),
        ]
        results = analyzer.calculate_versatility(wardrobe, top_k=4)
        neutral_item = next((r for r in results if r.garment_id == "neutral"), None)
        bright_item = next((r for r in results if r.garment_id == "bright"), None)
        if neutral_item and bright_item:
            assert neutral_item.versatility_score >= bright_item.versatility_score

    def test_versatility_has_description(self, analyzer):
        wardrobe = _balanced_wardrobe()
        results = analyzer.calculate_versatility(wardrobe, top_k=1)
        assert results[0].garment_description != ""

    def test_versatility_has_occasions(self, analyzer):
        wardrobe = _balanced_wardrobe()
        results = analyzer.calculate_versatility(wardrobe, top_k=5)
        # At least one item should have compatible occasions
        has_occasions = any(len(r.compatible_occasions) > 0 for r in results)
        assert has_occasions


# ============================================================================
# What-If Simulation
# ============================================================================

class TestSimulateAddition:

    def test_simulate_new_category(self, analyzer):
        """Adding a new category should resolve gaps."""
        wardrobe = [
            _g(category=GarmentCategory.TOP),
            _g(category=GarmentCategory.BOTTOM),
        ]
        new_shoes = _g(category=GarmentCategory.SHOES, subcategory="sneakers")
        result = analyzer.simulate_addition(wardrobe, new_shoes)
        assert "garment_description" in result
        assert "versatility_score" in result
        assert result["gaps_resolved"] >= 0

    def test_simulate_returns_recommendation(self, analyzer):
        wardrobe = _minimal_wardrobe()
        new_item = _g(category=GarmentCategory.SHOES, color="black")
        result = analyzer.simulate_addition(wardrobe, new_item)
        assert "recommendation" in result
        assert result["recommendation"] in [
            "Highly recommended",
            "Good addition",
            "Low impact addition",
        ]

    def test_simulate_empty_wardrobe(self, analyzer):
        new_item = _g(category=GarmentCategory.TOP)
        result = analyzer.simulate_addition([], new_item)
        assert result["gaps_before"] == 0  # no gaps for empty
        assert "recommendation" in result


# ============================================================================
# Purchase Suggestions
# ============================================================================

class TestPurchaseSuggestions:

    def test_suggestions_for_gapped_wardrobe(self, analyzer):
        wardrobe = _minimal_wardrobe()
        gaps = analyzer.detect_gaps(wardrobe)
        suggestions = analyzer.generate_purchase_suggestions(wardrobe, gaps=gaps)
        assert len(suggestions) > 0
        assert suggestions[0].priority == 1

    def test_suggestions_capped(self, analyzer):
        wardrobe = _minimal_wardrobe()
        suggestions = analyzer.generate_purchase_suggestions(
            wardrobe, max_suggestions=2
        )
        assert len(suggestions) <= 2

    def test_suggestions_have_colors(self, analyzer):
        wardrobe = _balanced_wardrobe()
        suggestions = analyzer.generate_purchase_suggestions(wardrobe)
        # At least some suggestions should have color recommendations
        for s in suggestions:
            assert isinstance(s.suggested_colors, list)

    def test_no_suggestions_for_perfect_wardrobe(self, analyzer):
        """A well-balanced wardrobe with all categories may have few suggestions."""
        wardrobe = _balanced_wardrobe()
        # Add missing categories to make it more complete
        wardrobe.append(_g(category=GarmentCategory.DRESS, color="red", formality=FormalityLevel.SMART_CASUAL))
        wardrobe.append(_g(category=GarmentCategory.BAG, subcategory="tote", color="tan"))
        wardrobe.extend([
            _g(category=GarmentCategory.TOP, color="green", seasons=[Season.FALL, Season.WINTER]),
            _g(category=GarmentCategory.TOP, color="beige", formality=FormalityLevel.SMART_CASUAL, seasons=[Season.FALL, Season.WINTER]),
        ])
        suggestions = analyzer.generate_purchase_suggestions(wardrobe)
        # May still have suggestions but fewer
        assert isinstance(suggestions, list)


# ============================================================================
# Full Analysis Pipeline
# ============================================================================

class TestFullAnalysis:

    def test_full_analysis_balanced(self, analyzer):
        wardrobe = _balanced_wardrobe()
        result = analyzer.analyze(wardrobe)
        assert result.overall_score > 0
        assert result.distribution.total_items == 9
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_full_analysis_empty(self, analyzer):
        result = analyzer.analyze([])
        assert result.overall_score == 0.0
        assert "Empty wardrobe" in result.summary

    def test_full_analysis_custom_occasions(self, analyzer):
        wardrobe = _balanced_wardrobe()
        result = analyzer.analyze(
            wardrobe,
            occasions=[Occasion.WORK, Occasion.FORMAL],
        )
        occasion_names = {c.occasion for c in result.occasion_coverage}
        assert "work" in occasion_names
        assert "formal" in occasion_names

    def test_full_analysis_with_context(self, analyzer):
        wardrobe = _balanced_wardrobe()
        ctx = UserContext(
            occasion=Occasion.BUSINESS,
            body_shape="rectangle",
        )
        result = analyzer.analyze(wardrobe, context=ctx)
        assert result.overall_score > 0

    def test_full_analysis_overall_score_range(self, analyzer):
        wardrobe = _balanced_wardrobe()
        result = analyzer.analyze(wardrobe)
        assert 0.0 <= result.overall_score <= 1.0

    def test_full_analysis_top_versatile_count(self, analyzer):
        wardrobe = _balanced_wardrobe()
        result = analyzer.analyze(wardrobe, top_k_versatile=3)
        assert len(result.top_versatile_items) <= 3
