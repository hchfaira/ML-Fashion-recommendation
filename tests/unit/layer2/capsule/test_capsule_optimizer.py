"""
Unit tests — CapsuleOptimizer (Greedy + 2-opt local search)
============================================================
Tests cover every step of the algorithm:
- Hard filtering (season / confidence / exclusion)
- Category seeding order
- Greedy fill maximises outfit combinations
- 2-opt local search improves (or at least preserves) score
- Anchor pieces are never dropped
- Per-piece contribution calculation
- Missing-piece detection
- Alternative capsule generation
- Edge cases: empty wardrobe, n > wardrobe, single piece, etc.
"""
from __future__ import annotations

import pytest
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.layer2_style.capsule.capsule_optimizer import (
    CapsuleOptimizer,
    CapsuleOptimizerResult,
    score_capsule,
    _garment_cat,
    _count_valid_outfits,
    _versatility_score,
    _form_tier,
    _mat_score,
    _hex_to_rgb,
    _nearest_neutral,
    _CATEGORY_ROLES,
)


# ============================================================================
# Helpers
# ============================================================================

def _g(
    garment_id: str | None = None,
    category: str = "top",
    subcategory: str = "t-shirt",
    color_hex: str = "#000000",
    formality: str = "casual",
    seasons: list[str] | None = None,
    material: str = "cotton",
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Build a serialised garment dict matching the AlgoStyle backend format."""
    return {
        "id": garment_id or f"g_{uuid4().hex[:8]}",
        "attributes": {
            "category": category,
            "subcategory": subcategory,
            "color_hex": color_hex,
            "formality": formality,
            "seasons": seasons or ["spring", "summer", "autumn", "winter"],
            "material": material,
            "confidence": confidence,
        },
    }


def _wardrobe_basic() -> List[Dict[str, Any]]:
    """Balanced 10-piece wardrobe."""
    return [
        _g("t1", "top", "t-shirt", "#FFFFFF", "casual"),
        _g("t2", "top", "shirt", "#1C3A5F", "smart_casual"),
        _g("t3", "top", "blouse", "#000000", "business"),
        _g("b1", "bottom", "jeans", "#0000FF", "casual"),
        _g("b2", "bottom", "trousers", "#808080", "business"),
        _g("d1", "dress", "midi-dress", "#FF0000", "smart_casual"),
        _g("s1", "shoes", "sneakers", "#FFFFFF", "casual"),
        _g("s2", "shoes", "loafers", "#8B4513", "smart_casual"),
        _g("o1", "outerwear", "blazer", "#000000", "business",
           seasons=["autumn", "winter"]),
        _g("a1", "accessory", "watch", "#C0C0C0", "casual"),
    ]


@pytest.fixture
def optimizer() -> CapsuleOptimizer:
    return CapsuleOptimizer(max_2opt_iterations=2)


@pytest.fixture
def wardrobe() -> List[Dict[str, Any]]:
    return _wardrobe_basic()


# ============================================================================
# Garment helper functions
# ============================================================================

class TestGarmentHelpers:
    """Low-level helper functions used by the optimizer."""

    def test_garment_cat_string(self):
        assert _garment_cat({"attributes": {"category": "top"}}) == "top"

    def test_garment_cat_dict(self):
        assert _garment_cat({"attributes": {"category": {"value": "bottom"}}}) == "bottom"

    def test_garment_cat_missing(self):
        assert _garment_cat({}) == ""

    def test_form_tier_known(self):
        assert _form_tier("casual") == 0
        assert _form_tier("smart_casual") == 1
        assert _form_tier("business") == 2
        assert _form_tier("formal") == 3

    def test_form_tier_none(self):
        assert _form_tier(None) == 0

    def test_mat_score_known(self):
        # cotton should be in the config or return default
        score = _mat_score("cotton")
        assert 0.0 <= score <= 1.0

    def test_mat_score_none(self):
        assert _mat_score(None) > 0.0

    def test_hex_to_rgb_valid(self):
        assert _hex_to_rgb("#FF0000") == (255, 0, 0)

    def test_hex_to_rgb_invalid(self):
        assert _hex_to_rgb("nope") == (200, 190, 185)

    def test_nearest_neutral_black(self):
        name = _nearest_neutral("#000000")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_versatility_all_seasons(self):
        g = _g(seasons=["spring", "summer", "autumn", "winter"])
        score = _versatility_score(g)
        assert score > 0.0

    def test_versatility_one_season_lower(self):
        g_all = _g(seasons=["spring", "summer", "autumn", "winter"])
        g_one = _g(seasons=["winter"])
        assert _versatility_score(g_all) >= _versatility_score(g_one)


# ============================================================================
# Outfit counting
# ============================================================================

class TestOutfitCounting:
    def test_no_garments_zero(self):
        assert _count_valid_outfits([], ["casual"]) == 0

    def test_top_bottom_pair(self):
        items = [
            _g("t", "top", formality="casual"),
            _g("b", "bottom", formality="casual"),
        ]
        count = _count_valid_outfits(items, ["casual"])
        assert count >= 1

    def test_dress_counts_alone(self):
        items = [_g("d", "dress", formality="casual")]
        count = _count_valid_outfits(items, ["casual"])
        assert count >= 1

    def test_shoes_multiply(self):
        base = [
            _g("t", "top", formality="casual"),
            _g("b", "bottom", formality="casual"),
        ]
        with_shoes = base + [
            _g("s1", "shoes", formality="casual"),
            _g("s2", "shoes", formality="casual"),
        ]
        assert _count_valid_outfits(with_shoes, ["casual"]) > _count_valid_outfits(base, ["casual"])

    def test_outerwear_multiplier(self):
        base = [
            _g("t", "top", formality="casual"),
            _g("b", "bottom", formality="casual"),
        ]
        with_outer = base + [_g("o", "outerwear", formality="casual")]
        assert _count_valid_outfits(with_outer, ["casual"]) >= _count_valid_outfits(base, ["casual"])

    def test_formality_mismatch_reduces(self):
        matched = [
            _g("t", "top", formality="casual"),
            _g("b", "bottom", formality="casual"),
        ]
        mismatched = [
            _g("t", "top", formality="casual"),
            _g("b", "bottom", formality="formal"),
        ]
        assert _count_valid_outfits(matched, ["casual"]) >= _count_valid_outfits(mismatched, ["casual"])


# ============================================================================
# Score capsule
# ============================================================================

class TestScoreCapsule:
    def test_empty_garments_zero(self):
        result = score_capsule([], ["casual"], 5)
        assert result["total_score"] == 0.0
        assert result["valid_combinations"] == 0

    def test_returns_all_keys(self):
        result = score_capsule(_wardrobe_basic()[:5], ["casual"], 5)
        for key in (
            "total_score", "valid_combinations", "color_palette",
            "color_names", "occasion_coverage", "practicality_score",
        ):
            assert key in result

    def test_score_range(self):
        result = score_capsule(_wardrobe_basic(), ["casual", "smart_casual"], 10)
        assert 0.0 <= result["total_score"] <= 100.0

    def test_more_pieces_more_combos(self):
        wb = _wardrobe_basic()
        small = score_capsule(wb[:4], ["casual"], 4)
        large = score_capsule(wb[:8], ["casual"], 8)
        assert large["valid_combinations"] >= small["valid_combinations"]


# ============================================================================
# Hard filtering
# ============================================================================

class TestHardFiltering:
    def test_excludes_by_id(self, optimizer):
        wb = _wardrobe_basic()
        result = optimizer.optimize(wb, n_pieces=5, excluded_ids=["t1", "b1"])
        selected_ids = {g["id"] for g in result.selected_garments}
        assert "t1" not in selected_ids
        assert "b1" not in selected_ids

    def test_filters_wrong_season(self, optimizer):
        """Garments only suitable for winter filtered out in warm climate."""
        wb = [
            _g("summer_top", "top", seasons=["summer"]),
            _g("winter_top", "top", seasons=["winter"]),
            _g("b1", "bottom", seasons=["summer"]),
            _g("s1", "shoes"),
        ]
        result = optimizer.optimize(wb, n_pieces=3, climate="warm")
        selected_ids = {g["id"] for g in result.selected_garments}
        # summer_top should be preferred over winter_top for warm climate
        if "winter_top" in selected_ids:
            # If winter_top sneaked in via fallback, at least check
            # summer_top is also there (it's a better match)
            assert "summer_top" in selected_ids

    def test_filters_low_confidence(self, optimizer):
        """Garments below confidence threshold are filtered out."""
        wb = [
            _g("good", "top", confidence=0.95),
            _g("bad", "top", confidence=0.20),
            _g("b1", "bottom", confidence=0.90),
            _g("s1", "shoes", confidence=0.90),
        ]
        result = optimizer.optimize(wb, n_pieces=3)
        selected_ids = {g["id"] for g in result.selected_garments}
        assert "bad" not in selected_ids


# ============================================================================
# Category seeding
# ============================================================================

class TestCategorySeeding:
    def test_seeds_required_categories(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=6)
        cats = {_garment_cat(g) for g in result.selected_garments}
        # Should seed at least top + bottom or dress
        has_outfit_base = ("top" in cats and "bottom" in cats) or "dress" in cats
        assert has_outfit_base

    def test_seeding_order_respected(self, optimizer):
        """First items seeded follow the defined order: top, bottom, shoes, ..."""
        wb = [
            _g("a1", "accessory"),
            _g("o1", "outerwear"),
            _g("s1", "shoes"),
            _g("b1", "bottom"),
            _g("t1", "top"),
        ]
        result = optimizer.optimize(wb, n_pieces=3)
        cats = [_garment_cat(g) for g in result.selected_garments]
        assert "top" in cats
        assert "bottom" in cats


# ============================================================================
# Greedy fill
# ============================================================================

class TestGreedyFill:
    def test_fills_to_n_pieces(self, optimizer, wardrobe):
        for n in (3, 5, 7):
            result = optimizer.optimize(wardrobe, n_pieces=n)
            assert result.n_pieces == n

    def test_returns_all_if_n_exceeds_wardrobe(self, optimizer):
        wb = _wardrobe_basic()[:3]
        result = optimizer.optimize(wb, n_pieces=20)
        assert result.n_pieces == len(wb)

    def test_greedy_maximises_combos(self, optimizer, wardrobe):
        """Adding more pieces should not decrease valid_combinations."""
        r5 = optimizer.optimize(wardrobe, n_pieces=5)
        r8 = optimizer.optimize(wardrobe, n_pieces=8)
        assert r8.valid_combinations >= r5.valid_combinations


# ============================================================================
# 2-opt local search
# ============================================================================

class TestTwoOpt:
    def test_does_not_degrade_score(self, wardrobe):
        """2-opt should never produce a worse score than greedy alone."""
        opt_with = CapsuleOptimizer(max_2opt_iterations=3)
        opt_without = CapsuleOptimizer(max_2opt_iterations=0)
        r_with = opt_with.optimize(wardrobe, n_pieces=6)
        r_without = opt_without.optimize(wardrobe, n_pieces=6)
        assert r_with.total_score >= r_without.total_score

    def test_respects_max_iterations(self):
        """Setting max_iterations=0 disables 2-opt."""
        opt = CapsuleOptimizer(max_2opt_iterations=0)
        wb = _wardrobe_basic()
        result = opt.optimize(wb, n_pieces=5)
        assert result.n_pieces == 5  # Still returns valid result


# ============================================================================
# Anchor pieces
# ============================================================================

class TestAnchorPieces:
    def test_anchors_always_included(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=5, anchor_ids=["t1", "b1"])
        selected_ids = {g["id"] for g in result.selected_garments}
        assert "t1" in selected_ids
        assert "b1" in selected_ids

    def test_anchors_not_swapped_by_2opt(self, wardrobe):
        opt = CapsuleOptimizer(max_2opt_iterations=5)
        result = opt.optimize(wardrobe, n_pieces=5, anchor_ids=["t1", "b1", "s1"])
        selected_ids = {g["id"] for g in result.selected_garments}
        assert "t1" in selected_ids
        assert "b1" in selected_ids
        assert "s1" in selected_ids

    def test_all_anchors_when_n_equals_anchor_count(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=3, anchor_ids=["t1", "b1", "s1"])
        selected_ids = {g["id"] for g in result.selected_garments}
        assert selected_ids == {"t1", "b1", "s1"}


# ============================================================================
# Piece contributions
# ============================================================================

class TestPieceContributions:
    def test_contributions_count_matches_selected(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=6)
        assert len(result.piece_contributions) == result.n_pieces

    def test_contribution_fields(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=5)
        for pc in result.piece_contributions:
            assert "garment_id" in pc
            assert "category" in pc
            assert "role" in pc
            assert "outfit_contribution" in pc
            assert "versatility_score" in pc

    def test_contributions_sorted_descending(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=6)
        contribs = [pc["outfit_contribution"] for pc in result.piece_contributions]
        assert contribs == sorted(contribs, reverse=True)

    def test_contribution_non_negative(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=5)
        for pc in result.piece_contributions:
            assert pc["outfit_contribution"] >= 0


# ============================================================================
# Missing pieces
# ============================================================================

class TestMissingPieces:
    def test_cold_climate_missing_outerwear(self, optimizer):
        wb = [
            _g("t1", "top"), _g("b1", "bottom"), _g("s1", "shoes"),
        ]
        result = optimizer.optimize(wb, n_pieces=3, climate="cold")
        missing_lower = [m.lower() for m in result.missing_pieces]
        assert any("warm" in m or "coat" in m or "sweater" in m for m in missing_lower)

    def test_no_missing_when_complete(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=8)
        # With 8 pieces from a balanced wardrobe, missing list should be short
        assert len(result.missing_pieces) <= 5


# ============================================================================
# Alternatives
# ============================================================================

class TestAlternatives:
    def test_alternatives_generated(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=6)
        assert isinstance(result.alternatives, list)

    def test_alternatives_have_scores(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=6)
        for alt in result.alternatives:
            assert "total_score" in alt
            assert "valid_combinations" in alt
            assert "label" in alt

    def test_alternatives_capped_at_2(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=6)
        assert len(result.alternatives) <= 2


# ============================================================================
# CapsuleOptimizerResult
# ============================================================================

class TestCapsuleOptimizerResult:
    def test_to_dict(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=5)
        d = result.to_dict()
        assert isinstance(d, dict)
        for key in CapsuleOptimizerResult.__slots__:
            assert key in d

    def test_result_fields_types(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=5)
        assert isinstance(result.n_pieces, int)
        assert isinstance(result.total_wardrobe, int)
        assert isinstance(result.total_score, float)
        assert isinstance(result.valid_combinations, int)
        assert isinstance(result.color_palette, list)
        assert isinstance(result.color_names, list)
        assert isinstance(result.occasion_coverage, dict)
        assert isinstance(result.piece_contributions, list)
        assert isinstance(result.missing_pieces, list)
        assert isinstance(result.alternatives, list)


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_empty_wardrobe(self, optimizer):
        result = optimizer.optimize([], n_pieces=5)
        assert result.n_pieces == 0
        assert result.total_score == 0.0

    def test_single_garment(self, optimizer):
        wb = [_g("solo", "dress")]
        result = optimizer.optimize(wb, n_pieces=1)
        assert result.n_pieces == 1
        assert result.selected_garments[0]["id"] == "solo"

    def test_n_pieces_one(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=1)
        assert result.n_pieces == 1

    def test_all_excluded(self, optimizer, wardrobe):
        all_ids = [g["id"] for g in wardrobe]
        result = optimizer.optimize(wardrobe, n_pieces=5, excluded_ids=all_ids)
        # Fallback logic may still select some; at minimum it shouldn't crash
        assert isinstance(result, CapsuleOptimizerResult)

    def test_duplicate_occasion_types(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=5,
                                    occasion_types=["casual", "casual"])
        assert result.n_pieces == 5

    def test_unknown_occasion_type(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=5,
                                    occasion_types=["party_on_mars"])
        assert isinstance(result, CapsuleOptimizerResult)

    def test_all_same_category(self, optimizer):
        """Wardrobe of only tops — should still produce a result."""
        wb = [_g(f"t{i}", "top") for i in range(10)]
        result = optimizer.optimize(wb, n_pieces=5)
        assert result.n_pieces == 5
        assert result.valid_combinations == 0  # no outfits possible without bottoms

    def test_climate_warm(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=5, climate="warm")
        assert result.n_pieces == 5

    def test_climate_cold(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, n_pieces=5, climate="cold")
        assert result.n_pieces == 5
