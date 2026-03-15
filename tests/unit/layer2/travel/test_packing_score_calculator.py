"""Tests — PackingScoreCalculator"""
import pytest
from tests.unit.layer2.travel.fixtures import _g, make_travel_wardrobe
from src.core.models import GarmentCategory, FormalityLevel, Season

from src.layer2_style.travel.packing_score_calculator import PackingScoreCalculator


@pytest.fixture
def calc():
    return PackingScoreCalculator()


@pytest.fixture
def wardrobe():
    return make_travel_wardrobe()


class TestEmptyInput:
    def test_empty_garments_returns_zero(self, calc):
        assert calc([], {}) == 0.0

    def test_versatility_ratio_empty(self, calc):
        assert calc.versatility_ratio([]) == 0.0


class TestScoreRange:
    def test_score_in_0_100(self, calc, wardrobe):
        score = calc(wardrobe, {"casual": 5, "evening": 2}, colour_cohesion=0.8)
        assert 0.0 <= score <= 100.0

    def test_score_increases_with_more_categories(self, calc):
        tops_only = [_g(f"t{i}", GarmentCategory.TOP) for i in range(5)]
        balanced = make_travel_wardrobe()

        score_tops = calc(tops_only, {"casual": 3}, colour_cohesion=0.7)
        score_balanced = calc(balanced, {"casual": 5, "evening": 2}, colour_cohesion=0.8)
        # Balanced wardrobe should score higher
        assert score_balanced >= score_tops


class TestVersatilityRatio:
    def test_ratio_increases_with_combinations(self, calc, wardrobe):
        vr = calc.versatility_ratio(wardrobe)
        assert vr > 0.0

    def test_ratio_tops_only_lower(self, calc):
        tops_only = [_g(f"t{i}", GarmentCategory.TOP) for i in range(4)]
        vr = calc.versatility_ratio(tops_only)
        assert vr <= 2.0  # no combinations possible

    def test_ratio_with_full_outfit_categories(self, calc, wardrobe):
        vr = calc.versatility_ratio(wardrobe)
        assert vr > 1.0  # should have more outfits than pieces


class TestScoreLabel:
    def test_excellent(self):
        assert PackingScoreCalculator.score_label(85) == "excellent"

    def test_good(self):
        assert PackingScoreCalculator.score_label(65) == "good"

    def test_fair(self):
        assert PackingScoreCalculator.score_label(45) == "fair"

    def test_poor(self):
        assert PackingScoreCalculator.score_label(30) == "poor"

    def test_boundary_80(self):
        assert PackingScoreCalculator.score_label(80) == "excellent"


class TestOccasionCoverage:
    def test_no_occasions_returns_full_score(self, calc, wardrobe):
        score = calc(wardrobe, {}, colour_cohesion=1.0)
        # No occasion constraint — score should be positive
        assert score > 0.0

    def test_mismatched_occasion_lowers_score(self, calc):
        # Wardrobe of very casual items only
        casual_only = [
            _g(f"t{i}", GarmentCategory.TOP, formality=FormalityLevel.VERY_CASUAL)
            for i in range(4)
        ]
        # Occasion requires formal wear
        score = calc(casual_only, {"formal": 5}, colour_cohesion=0.8)
        # Should have lower occasion coverage
        assert score < 80.0
