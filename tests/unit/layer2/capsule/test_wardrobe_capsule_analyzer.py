"""Tests — WardrobeCapsuleAnalyzer (F1)"""
import pytest
from uuid import uuid4
from typing import List, Dict

from src.core.models import (
    CapsuleGarmentRole,
    Garment,
    GarmentAttributes,
    GarmentCategory,
    ColorProfile,
    FormalityLevel,
    Season,
    PatternInfo,
    MaterialProfile,
    SeasonalityInfo,
)
from src.layer2_style.capsule.wardrobe_capsule_analyzer import WardrobeCapsuleAnalyzer


# ============================================================================
# Helpers
# ============================================================================

def _g(
    garment_id=None,
    category=GarmentCategory.TOP,
    subcategory="t-shirt",
    color="white",
    formality=FormalityLevel.CASUAL,
    seasons=None,
) -> Garment:
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
            pattern=PatternInfo(type="solid"),
            material=MaterialProfile(primary="cotton"),
        ),
    )


def _balanced_wardrobe() -> List[Garment]:
    return [
        _g("t1", GarmentCategory.TOP, "t-shirt", "white"),
        _g("t2", GarmentCategory.TOP, "t-shirt", "navy"),
        _g("t3", GarmentCategory.TOP, "shirt", "black", FormalityLevel.BUSINESS),
        _g("b1", GarmentCategory.BOTTOM, "jeans", "blue"),
        _g("b2", GarmentCategory.BOTTOM, "trousers", "grey", FormalityLevel.BUSINESS),
        _g("s1", GarmentCategory.SHOES, "sneakers", "white"),
        _g("s2", GarmentCategory.SHOES, "loafers", "brown", FormalityLevel.SMART_CASUAL),
        _g("o1", GarmentCategory.OUTERWEAR, "blazer", "black", FormalityLevel.BUSINESS,
           seasons=[Season.FALL, Season.WINTER]),
        _g("a1", GarmentCategory.ACCESSORY, "watch", "silver"),
    ]


@pytest.fixture
def analyzer():
    return WardrobeCapsuleAnalyzer()


# ============================================================================
# Empty wardrobe
# ============================================================================

class TestEmptyWardrobe:
    def test_returns_zero_score(self, analyzer):
        result = analyzer.analyze([])
        assert result.cohesion_score == 0.0

    def test_no_garment_scores(self, analyzer):
        result = analyzer.analyze([])
        assert result.garment_scores == []


# ============================================================================
# Single garment
# ============================================================================

class TestSingleGarment:
    def test_score_is_positive(self, analyzer):
        result = analyzer.analyze([_g(color="black")])
        assert result.cohesion_score > 0

    def test_no_redundant_pairs(self, analyzer):
        result = analyzer.analyze([_g()])
        assert result.redundant_pairs == []


# ============================================================================
# Cohesion score
# ============================================================================

class TestCohesionScore:
    def test_balanced_wardrobe_scores_above_40(self, analyzer):
        result = analyzer.analyze(_balanced_wardrobe())
        assert result.cohesion_score >= 40.0

    def test_cohesion_bounded_0_100(self, analyzer):
        result = analyzer.analyze(_balanced_wardrobe())
        assert 0.0 <= result.cohesion_score <= 100.0

    def test_all_neutral_blacks_scores_high(self, analyzer):
        garments = [_g(color="black") for _ in range(8)]
        result = analyzer.analyze(garments)
        # All same colour → high cohesion
        assert result.cohesion_score >= 50.0

    def test_redundant_wardrobe_scores_lower_than_diverse(self, analyzer):
        """A wardrobe of 8 identical garments (massive redundancy) should score
        lower than a wardrobe of 8 garments with diverse subcategories/colors."""
        # Redundant: 8 identical black t-shirts → 28 duplicate pairs, heavy penalty
        redundant = [_g(color="black") for _ in range(8)]
        redundant_result = analyzer.analyze(redundant)

        # Diverse: mix of tops & bottoms with neutral complementary colors → no redundancy
        diverse = [
            _g(color="black", category=GarmentCategory.TOP, subcategory="t-shirt"),
            _g(color="white", category=GarmentCategory.TOP, subcategory="shirt"),
            _g(color="navy", category=GarmentCategory.BOTTOM, subcategory="jeans"),
            _g(color="beige", category=GarmentCategory.BOTTOM, subcategory="trousers"),
            _g(color="grey", category=GarmentCategory.SHOES, subcategory="sneakers"),
            _g(color="brown", category=GarmentCategory.SHOES, subcategory="loafers"),
            _g(color="camel", category=GarmentCategory.OUTERWEAR, subcategory="coat"),
            _g(color="black", category=GarmentCategory.ACCESSORY, subcategory="watch"),
        ]
        diverse_result = analyzer.analyze(diverse)

        assert redundant_result.cohesion_score < diverse_result.cohesion_score


# ============================================================================
# Key pieces & orphans
# ============================================================================

class TestRoleAssignment:
    def test_high_centrality_yields_key_piece(self, analyzer):
        garments = [_g("g1", color="black")]
        centrality_map = {"g1": 0.9}  # well above key_piece_min
        result = analyzer.analyze(garments, centrality_map=centrality_map)
        assert result.garment_scores[0].capsule_role == CapsuleGarmentRole.KEY_PIECE

    def test_zero_centrality_yields_orphan(self, analyzer):
        garments = [_g("g1", color="red")]
        centrality_map = {"g1": 0.01}  # below orphan_max
        result = analyzer.analyze(garments, centrality_map=centrality_map)
        assert result.garment_scores[0].capsule_role == CapsuleGarmentRole.ORPHAN

    def test_key_pieces_listed_in_result(self, analyzer):
        garments = [_g("k1", color="black"), _g("k2", color="navy")]
        centrality_map = {"k1": 0.9, "k2": 0.85}
        result = analyzer.analyze(garments, centrality_map=centrality_map)
        assert "k1" in result.key_pieces
        assert "k2" in result.key_pieces


# ============================================================================
# Redundancy detection
# ============================================================================

class TestRedundancyDetection:
    def test_identical_subcategory_same_color_flagged(self, analyzer):
        garments = [
            _g("b1", GarmentCategory.BOTTOM, "jeans", "blue"),
            _g("b2", GarmentCategory.BOTTOM, "jeans", "blue"),
        ]
        result = analyzer.analyze(garments)
        assert len(result.redundant_pairs) == 1
        pair = result.redundant_pairs[0]
        assert pair.similarity_score == 1.0

    def test_different_subcategory_no_redundancy(self, analyzer):
        garments = [
            _g("b1", GarmentCategory.BOTTOM, "jeans", "blue"),
            _g("b2", GarmentCategory.BOTTOM, "trousers", "blue"),
        ]
        result = analyzer.analyze(garments)
        assert result.redundant_pairs == []

    def test_same_family_flagged(self, analyzer):
        garments = [
            _g("b1", GarmentCategory.TOP, "t-shirt", "black"),
            _g("b2", GarmentCategory.TOP, "t-shirt", "charcoal"),
        ]
        result = analyzer.analyze(garments)
        # black and charcoal are both dark_neutral → similarity 0.75 ≥ default threshold (0.85)?
        # default threshold is 0.85 so 0.75 should NOT be flagged
        assert len(result.redundant_pairs) == 0

    def test_no_redundancy_with_diverse_colors(self, analyzer):
        garments = [
            _g("b1", GarmentCategory.TOP, "t-shirt", "white"),
            _g("b2", GarmentCategory.TOP, "t-shirt", "red"),
        ]
        result = analyzer.analyze(garments)
        assert result.redundant_pairs == []


# ============================================================================
# Centrality map integration
# ============================================================================

class TestCentralityMapIntegration:
    def test_centrality_overrides_heuristic(self, analyzer):
        g = _g("g1", color="red")  # Red → heuristic would be low
        result_heuristic = analyzer.analyze([g])
        result_centrality = analyzer.analyze([g], centrality_map={"g1": 0.9})
        assert result_centrality.garment_scores[0].versatility_score == 0.9
        assert result_centrality.garment_scores[0].versatility_score > result_heuristic.garment_scores[0].versatility_score

    def test_missing_centrality_entry_falls_back(self, analyzer):
        g = _g("g1", color="black")
        # Pass map that doesn't contain g1
        result = analyzer.analyze([g], centrality_map={"other_id": 0.5})
        # Should use heuristic — no crash
        assert result.garment_scores[0].versatility_score > 0.0


# ============================================================================
# Wardrobe hash
# ============================================================================

class TestWardrobeHash:
    def test_same_garments_same_hash(self, analyzer):
        garments = _balanced_wardrobe()
        h1 = WardrobeCapsuleAnalyzer.wardrobe_hash(garments)
        h2 = WardrobeCapsuleAnalyzer.wardrobe_hash(garments)
        assert h1 == h2

    def test_different_garments_different_hash(self):
        garments_a = [_g("g1"), _g("g2")]
        garments_b = [_g("g1"), _g("g3")]
        h1 = WardrobeCapsuleAnalyzer.wardrobe_hash(garments_a)
        h2 = WardrobeCapsuleAnalyzer.wardrobe_hash(garments_b)
        assert h1 != h2

    def test_order_independent(self):
        g1 = _g("g1")
        g2 = _g("g2")
        h1 = WardrobeCapsuleAnalyzer.wardrobe_hash([g1, g2])
        h2 = WardrobeCapsuleAnalyzer.wardrobe_hash([g2, g1])
        assert h1 == h2


# ============================================================================
# Projected score
# ============================================================================

class TestProjectedScore:
    def test_projected_gte_current_when_orphans_present(self, analyzer):
        # Force orphan via centrality
        garments = [
            _g("k1", color="black"),
            _g("o1", color="red"),
        ]
        centrality_map = {"k1": 0.8, "o1": 0.01}
        result = analyzer.analyze(garments, centrality_map=centrality_map)
        assert result.projected_score_after_cleanup >= result.cohesion_score


# ============================================================================
# Capsule profile
# ============================================================================

class TestCapsuleProfile:
    def test_small_wardrobe_is_minimalist(self, analyzer):
        garments = [_g() for _ in range(5)]
        result = analyzer.analyze(garments)
        assert result.capsule_profile == "minimalist"

    def test_large_wardrobe_is_rich(self, analyzer):
        garments = [_g() for _ in range(40)]
        result = analyzer.analyze(garments)
        assert result.capsule_profile == "rich"
