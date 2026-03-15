"""Tests — WeeklyRotationPlanner"""
import pytest
from tests.unit.layer2.travel.fixtures import _g, make_travel_wardrobe
from src.core.models import GarmentCategory, FormalityLevel

from src.layer3_context.travel.weekly_rotation_planner import WeeklyRotationPlanner
from src.core.travel_models import WeeklyRotationPlan


@pytest.fixture
def planner():
    return WeeklyRotationPlanner(cooldown_days=2)


@pytest.fixture
def wardrobe():
    return make_travel_wardrobe()


@pytest.fixture
def schedule():
    return [
        {"day": "Monday", "session": "full_day", "occasion": "work"},
        {"day": "Tuesday", "session": "full_day", "occasion": "work"},
        {"day": "Wednesday", "session": "full_day", "occasion": "work"},
        {"day": "Thursday", "session": "full_day", "occasion": "casual"},
        {"day": "Friday", "session": "full_day", "occasion": "work"},
        {"day": "Saturday", "session": "full_day", "occasion": "casual"},
        {"day": "Sunday", "session": "full_day", "occasion": "casual"},
    ]


class TestReturnTypes:
    def test_returns_weekly_plan(self, planner, wardrobe, schedule):
        plan = planner.plan(wardrobe, schedule=schedule)
        assert isinstance(plan, WeeklyRotationPlan)

    def test_slot_count_matches_schedule(self, planner, wardrobe, schedule):
        plan = planner.plan(wardrobe, schedule=schedule)
        assert len(plan.slots) == len(schedule)

    def test_empty_wardrobe_returns_warning(self, planner, schedule):
        plan = planner.plan([], schedule=schedule)
        assert len(plan.warnings) > 0
        assert len(plan.slots) == 0


class TestDefaultSchedule:
    def test_default_schedule_generates_7_slots(self, planner, wardrobe):
        plan = planner.plan(wardrobe)
        assert len(plan.slots) == 7

    def test_custom_work_weekend_split(self, planner, wardrobe):
        plan = planner.plan(wardrobe, work_days=3, weekend_days=2)
        assert len(plan.slots) == 5

    def test_work_occasion_applied(self, planner, wardrobe):
        plan = planner.plan(wardrobe, work_days=5, weekend_days=2,
                            work_occasion="business", weekend_occasion="casual")
        business_days = [s for s in plan.slots if s.occasion == "business"]
        casual_days = [s for s in plan.slots if s.occasion == "casual"]
        assert len(business_days) == 5
        assert len(casual_days) == 2


class TestAntiRepetition:
    def test_repeat_rate_in_range(self, planner, wardrobe, schedule):
        plan = planner.plan(wardrobe, schedule=schedule)
        assert 0.0 <= plan.repeat_rate <= 1.0

    def test_large_wardrobe_low_repeat_rate(self, wardrobe, schedule):
        """12-piece wardrobe should have low repeat rate for 7 days."""
        planner = WeeklyRotationPlanner(cooldown_days=3)
        plan = planner.plan(wardrobe, schedule=schedule)
        assert plan.repeat_rate <= 0.5

    def test_tiny_wardrobe_has_repeats(self, schedule):
        """3-piece wardrobe forced to repeat over 7 days."""
        planner = WeeklyRotationPlanner(cooldown_days=3)
        tiny = [_g(f"t{i}", GarmentCategory.TOP) for i in range(3)]
        plan = planner.plan(tiny, schedule=schedule)
        # With only 3 items and 7 days with cooldown=3, repeats are expected
        assert plan.repeat_rate > 0.0


class TestUtilisation:
    def test_utilisation_keys_are_garment_ids(self, planner, wardrobe, schedule):
        plan = planner.plan(wardrobe, schedule=schedule)
        wardrobe_ids = {g.id for g in wardrobe}
        for gid in plan.garment_utilisation:
            assert gid in wardrobe_ids

    def test_utilisation_values_positive(self, planner, wardrobe, schedule):
        plan = planner.plan(wardrobe, schedule=schedule)
        for count in plan.garment_utilisation.values():
            assert count > 0


class TestCoverage:
    def test_coverage_pct_in_range(self, planner, wardrobe, schedule):
        plan = planner.plan(wardrobe, schedule=schedule)
        assert 0.0 <= plan.coverage_pct <= 1.0

    def test_full_coverage_with_enough_garments(self, planner, wardrobe, schedule):
        plan = planner.plan(wardrobe, schedule=schedule)
        # With 12 garments and 7 days, we expect high coverage
        assert plan.coverage_pct >= 0.8


class TestSlotContent:
    def test_slots_have_garment_ids(self, planner, wardrobe, schedule):
        plan = planner.plan(wardrobe, schedule=schedule)
        for slot in plan.slots:
            assert isinstance(slot.garment_ids, list)
            assert len(slot.garment_ids) > 0

    def test_day_names_match_schedule(self, planner, wardrobe, schedule):
        plan = planner.plan(wardrobe, schedule=schedule)
        plan_days = [s.day for s in plan.slots]
        expected_days = [s["day"] for s in schedule]
        assert plan_days == expected_days
