"""Tests — TravelWardrobePlanner"""
import pytest
from tests.unit.layer2.travel.fixtures import _g, make_travel_wardrobe
from src.core.models import GarmentCategory, FormalityLevel

from src.layer2_style.travel.travel_wardrobe_planner import TravelWardrobePlanner
from src.core.travel_models import PackingPlan, TravelConstraints


@pytest.fixture
def planner():
    return TravelWardrobePlanner()


@pytest.fixture
def wardrobe():
    return make_travel_wardrobe()


@pytest.fixture
def constraints():
    return TravelConstraints(
        destination="rome",
        days=7,
        occasions={"casual": 4, "evening": 2, "tourism": 1},
        max_pieces=10,
        body_shape="pear",
        season="summer",
    )


class TestReturnTypes:
    def test_returns_packing_plan(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        assert isinstance(plan, PackingPlan)

    def test_destination_stored(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        assert plan.destination == "rome"

    def test_day_plans_count(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        assert len(plan.day_plans) == constraints.days


class TestPackingConstraints:
    def test_packed_count_lte_max_pieces(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        assert len(plan.packed_garments) <= constraints.max_pieces

    def test_packed_count_lte_wardrobe_size(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        assert len(plan.packed_garments) <= len(wardrobe)

    def test_packed_garment_ids_unique(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        ids = [pg.garment_id for pg in plan.packed_garments]
        assert len(ids) == len(set(ids))


class TestScoring:
    def test_packing_score_in_range(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        assert 0.0 <= plan.packing_score <= 100.0

    def test_score_label_valid(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        assert plan.score_label in {"excellent", "good", "fair", "poor"}

    def test_versatility_ratio_positive(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        assert plan.versatility_ratio >= 0.0


class TestEmptyWardrobe:
    def test_empty_wardrobe_returns_warnings(self, planner, constraints):
        plan = planner.plan([], constraints)
        assert len(plan.warnings) > 0
        assert plan.packing_score == 0.0
        assert len(plan.packed_garments) == 0


class TestDayPlans:
    def test_day_plans_have_garments(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        # At least most days should have garments
        filled = sum(1 for dp in plan.day_plans if dp.garment_ids)
        assert filled >= constraints.days * 0.5

    def test_day_numbers_sequential(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        days = [dp.day for dp in plan.day_plans]
        assert days == sorted(days)

    def test_occasions_present(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        occasions_used = {dp.occasion for dp in plan.day_plans}
        # Should cover at least one of the requested occasions
        assert occasions_used & set(constraints.occasions.keys())


class TestOccasionCoverage:
    def test_occasion_coverage_keys_match_input(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        for occ in constraints.occasions:
            assert occ in plan.occasion_coverage

    def test_occasion_coverage_values_in_range(self, planner, wardrobe, constraints):
        plan = planner.plan(wardrobe, constraints)
        for cov in plan.occasion_coverage.values():
            assert 0.0 <= cov <= 1.0


class TestSmallWardrobe:
    def test_small_wardrobe_generates_warnings(self, planner, constraints):
        tiny = [_g(f"t{i}", GarmentCategory.TOP) for i in range(2)]
        plan = planner.plan(tiny, constraints)
        assert len(plan.warnings) > 0

    def test_large_max_pieces_still_works(self, planner, wardrobe):
        c = TravelConstraints(
            destination="paris", days=14, occasions={"casual": 10, "business": 4}, max_pieces=50
        )
        plan = planner.plan(wardrobe, c)
        assert len(plan.packed_garments) <= len(wardrobe)
