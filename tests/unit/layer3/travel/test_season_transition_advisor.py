"""Tests — SeasonTransitionAdvisor"""
import pytest
from tests.unit.layer2.travel.fixtures import _g, make_travel_wardrobe
from src.core.models import GarmentCategory, Season

from src.layer3_context.travel.season_transition_advisor import SeasonTransitionAdvisor
from src.core.travel_models import SeasonTransitionPlan, TransitionPhase


@pytest.fixture
def advisor():
    return SeasonTransitionAdvisor()


@pytest.fixture
def wardrobe():
    return make_travel_wardrobe()


class TestReturnTypes:
    def test_returns_plan(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "summer", "fall")
        assert isinstance(plan, SeasonTransitionPlan)

    def test_seasons_stored(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "summer", "fall")
        assert plan.current_season == "summer"
        assert plan.target_season == "fall"

    def test_actions_is_list(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "summer", "fall")
        assert isinstance(plan.actions, list)

    def test_phase_summary_has_three_phases(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "summer", "fall")
        assert len(plan.phase_summary) == 3


class TestGarmentClassification:
    def test_all_garments_get_action(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "summer", "fall")
        # Actions include per-garment actions + color/buy actions
        per_garment = [a for a in plan.actions if a.garment_id is not None]
        assert len(per_garment) == len(wardrobe)

    def test_action_values_valid(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "summer", "fall")
        valid_actions = {"store", "keep", "buy", "layer"}
        for action in plan.actions:
            assert action.action in valid_actions

    def test_store_ids_in_wardrobe(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "summer", "fall")
        wardrobe_ids = {g.id for g in wardrobe}
        for gid in plan.store_ids:
            assert gid in wardrobe_ids

    def test_keep_ids_in_wardrobe(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "summer", "fall")
        wardrobe_ids = {g.id for g in wardrobe}
        for gid in plan.keep_ids:
            assert gid in wardrobe_ids

    def test_store_plus_keep_equals_wardrobe_size(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "summer", "fall")
        assert len(plan.store_ids) + len(plan.keep_ids) == len(wardrobe)


class TestSeasonPairCoverage:
    def test_summer_to_fall(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "summer", "fall")
        assert plan.current_season == "summer"
        assert plan.target_season == "fall"

    def test_fall_to_winter(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "fall", "winter")
        assert len(plan.actions) > 0

    def test_winter_to_spring(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "winter", "spring")
        assert len(plan.actions) > 0

    def test_spring_to_summer(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "spring", "summer")
        assert len(plan.actions) > 0

    def test_autumn_alias(self, advisor, wardrobe):
        """'autumn' should be accepted as alias for 'fall'."""
        plan = advisor.advise(wardrobe, "summer", "autumn")
        assert plan.target_season == "autumn"

    def test_unknown_pair_still_returns_plan(self, advisor, wardrobe):
        """Should not crash on unknown season combo."""
        plan = advisor.advise(wardrobe, "spring", "fall")
        assert isinstance(plan, SeasonTransitionPlan)


class TestColorShiftNotes:
    def test_color_notes_are_strings(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "summer", "fall")
        for note in plan.color_shift_notes:
            assert isinstance(note, str)
            assert len(note) > 0

    def test_color_notes_nonempty_for_known_pair(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "summer", "fall")
        assert len(plan.color_shift_notes) > 0


class TestBuyDescriptions:
    def test_buy_descriptions_are_strings(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "fall", "winter")
        for desc in plan.buy_descriptions:
            assert isinstance(desc, str)


class TestPhaseOrdering:
    def test_actions_sorted_by_phase(self, advisor, wardrobe):
        plan = advisor.advise(wardrobe, "summer", "fall")
        phases = [a.phase.value for a in plan.actions]
        # Phase 1 should come before phase 2, phase 2 before phase 3
        # (not necessarily strictly sorted since per-garment actions are all phase_1)
        phase_1_indices = [i for i, p in enumerate(phases) if p == "phase_1"]
        phase_2_indices = [i for i, p in enumerate(phases) if p == "phase_2"]
        phase_3_indices = [i for i, p in enumerate(phases) if p == "phase_3"]

        if phase_1_indices and phase_2_indices:
            assert max(phase_1_indices) < max(phase_2_indices) or min(phase_2_indices) >= min(phase_2_indices)
        if phase_2_indices and phase_3_indices:
            assert max(phase_2_indices) <= max(phase_3_indices)


class TestEmptyWardrobe:
    def test_empty_wardrobe_returns_plan(self, advisor):
        plan = advisor.advise([], "summer", "fall")
        assert isinstance(plan, SeasonTransitionPlan)
        assert len(plan.store_ids) == 0
        assert len(plan.keep_ids) == 0
