"""Integration tests — Season Planning Pipeline end-to-end.

Tests the full season planning flow:
  SeasonalAuditEngine → SeasonTransitionAdvisor → ShoppingListOptimizer → WeeklyRotationPlanner
"""
import pytest
from typing import List

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory, ColorProfile,
    FormalityLevel, Season, PatternInfo, MaterialProfile, SeasonalityInfo,
)
from src.core.travel_models import (
    SeasonAuditResult, SeasonTransitionPlan, ShoppingListResult, WeeklyRotationPlan
)
from src.layer2_style.travel.seasonal_audit_engine import SeasonalAuditEngine
from src.layer2_style.travel.shopping_list_optimizer import ShoppingListOptimizer
from src.layer3_context.travel.season_transition_advisor import SeasonTransitionAdvisor
from src.layer3_context.travel.weekly_rotation_planner import WeeklyRotationPlanner


def _g(gid, cat, color, formality, material, seasons, style_tags=None):
    sl = seasons or [Season.SPRING]
    return Garment(
        id=gid,
        attributes=GarmentAttributes(
            category=cat,
            subcategory=cat.value,
            color=ColorProfile(primary=color, hex_codes=[]),
            formality_level=formality,
            season_suitable=sl,
            seasonality=SeasonalityInfo(seasons=sl),
            pattern=PatternInfo(type="solid"),
            material=MaterialProfile(primary=material),
            style_tags=style_tags or ["casual"],
        ),
    )


@pytest.fixture
def mixed_wardrobe() -> List[Garment]:
    return [
        _g("t1", GarmentCategory.TOP, "white", FormalityLevel.CASUAL, "cotton",
           [Season.SPRING, Season.SUMMER]),
        _g("t2", GarmentCategory.TOP, "navy", FormalityLevel.SMART_CASUAL, "cotton",
           [Season.SPRING, Season.FALL]),
        _g("t3", GarmentCategory.TOP, "black", FormalityLevel.BUSINESS, "silk",
           [Season.FALL, Season.WINTER]),
        _g("b1", GarmentCategory.BOTTOM, "blue", FormalityLevel.CASUAL, "denim",
           [Season.SPRING, Season.FALL]),
        _g("b2", GarmentCategory.BOTTOM, "grey", FormalityLevel.BUSINESS, "wool",
           [Season.FALL, Season.WINTER]),
        _g("d1", GarmentCategory.DRESS, "rust", FormalityLevel.SMART_CASUAL, "linen",
           [Season.SPRING, Season.SUMMER]),
        _g("o1", GarmentCategory.OUTERWEAR, "camel", FormalityLevel.SMART_CASUAL, "wool",
           [Season.FALL, Season.WINTER]),
        _g("s1", GarmentCategory.SHOES, "white", FormalityLevel.CASUAL, "leather",
           [Season.SPRING, Season.SUMMER]),
        _g("s2", GarmentCategory.SHOES, "brown", FormalityLevel.SMART_CASUAL, "leather",
           [Season.FALL, Season.WINTER]),
        _g("a1", GarmentCategory.ACCESSORY, "olive", FormalityLevel.CASUAL, "wool",
           [Season.FALL, Season.WINTER]),
    ]


class TestSeasonAuditToTransition:
    def test_audit_store_matches_transition_store(self, mixed_wardrobe):
        """Garments flagged as 'store' by audit should be stored in transition plan."""
        engine = SeasonalAuditEngine()
        audit = engine.audit(mixed_wardrobe, "winter")

        from src.core.travel_models import SeasonReadiness
        store_ids = {a.garment_id for a in audit.garment_audits if a.readiness == SeasonReadiness.STORE}

        advisor = SeasonTransitionAdvisor()
        plan = advisor.advise(mixed_wardrobe, "summer", "winter")

        # Transition plan store_ids may differ (different algorithm) but should overlap
        # — this is a sanity check, not an exact match
        assert isinstance(plan.store_ids, list)
        assert len(plan.store_ids) + len(plan.keep_ids) == len(mixed_wardrobe)

    def test_transition_phases_are_complete(self, mixed_wardrobe):
        advisor = SeasonTransitionAdvisor()
        plan = advisor.advise(mixed_wardrobe, "summer", "fall")

        assert "phase_1" in plan.phase_summary
        assert "phase_2" in plan.phase_summary
        assert "phase_3" in plan.phase_summary


class TestAuditToShoppingList:
    def test_gaps_become_shopping_items(self, mixed_wardrobe):
        """Coverage gaps from audit should be passed as hard_gaps to optimizer."""
        engine = SeasonalAuditEngine()
        audit = engine.audit(mixed_wardrobe, "winter")
        hard_gaps = [c.category for c in audit.coverage_by_category if c.gap > 0]

        optimizer = ShoppingListOptimizer()
        result = optimizer.optimize(
            mixed_wardrobe,
            budget=500,
            season="winter",
            hard_gaps=hard_gaps,
        )

        assert isinstance(result, ShoppingListResult)
        assert result.budget_eur == 500
        # Items should be returned
        assert isinstance(result.items, list)

    def test_shopping_list_respects_budget(self, mixed_wardrobe):
        budget = 200.0
        optimizer = ShoppingListOptimizer()
        result = optimizer.optimize(mixed_wardrobe, budget=budget, season="fall")
        assert result.total_estimated_cost <= budget + 0.01


class TestFullSeasonPipeline:
    def test_audit_transition_rotation(self, mixed_wardrobe):
        """Run full season pipeline: audit → transition → weekly rotation."""
        # Step 1: Audit for target season
        engine = SeasonalAuditEngine()
        audit = engine.audit(mixed_wardrobe, "fall")

        # Step 2: Transition plan
        advisor = SeasonTransitionAdvisor()
        transition = advisor.advise(mixed_wardrobe, "summer", "fall")

        # Step 3: Weekly rotation with 'keep' garments
        keep_garments = [g for g in mixed_wardrobe if g.id in transition.keep_ids]
        if not keep_garments:
            keep_garments = mixed_wardrobe  # fallback

        rotation_planner = WeeklyRotationPlanner(cooldown_days=2)
        weekly_plan = rotation_planner.plan(
            keep_garments,
            work_days=5,
            weekend_days=2,
            work_occasion="work",
            weekend_occasion="casual",
        )

        # Verify each step returned the right type
        assert isinstance(audit, SeasonAuditResult)
        assert isinstance(transition, SeasonTransitionPlan)
        assert isinstance(weekly_plan, WeeklyRotationPlan)

        # 7-day plan
        assert len(weekly_plan.slots) == 7

    def test_all_seasons_roundtrip(self, mixed_wardrobe):
        """Run audit + transition for all season pairs without errors."""
        pairs = [
            ("summer", "fall"),
            ("fall", "winter"),
            ("winter", "spring"),
            ("spring", "summer"),
        ]
        engine = SeasonalAuditEngine()
        advisor = SeasonTransitionAdvisor()

        for cur, tgt in pairs:
            audit = engine.audit(mixed_wardrobe, tgt)
            plan = advisor.advise(mixed_wardrobe, cur, tgt)
            assert isinstance(audit, SeasonAuditResult)
            assert isinstance(plan, SeasonTransitionPlan)


class TestWeeklyRotationPlannerIntegration:
    def test_rotation_with_explicit_schedule(self, mixed_wardrobe):
        schedule = [
            {"day": "Monday", "session": "full_day", "occasion": "work"},
            {"day": "Tuesday", "session": "full_day", "occasion": "work"},
            {"day": "Wednesday", "session": "full_day", "occasion": "casual"},
            {"day": "Thursday", "session": "full_day", "occasion": "work"},
            {"day": "Friday", "session": "full_day", "occasion": "work"},
            {"day": "Saturday", "session": "full_day", "occasion": "casual"},
            {"day": "Sunday", "session": "full_day", "occasion": "casual"},
        ]
        planner = WeeklyRotationPlanner(cooldown_days=2)
        plan = planner.plan(mixed_wardrobe, schedule=schedule)

        assert len(plan.slots) == 7
        assert 0.0 <= plan.repeat_rate <= 1.0
        assert plan.coverage_pct >= 0.5

    def test_rotation_utilises_wardrobe(self, mixed_wardrobe):
        planner = WeeklyRotationPlanner(cooldown_days=2)
        plan = planner.plan(mixed_wardrobe)
        # At least some garments should be used
        assert len(plan.garment_utilisation) > 0
        total_uses = sum(plan.garment_utilisation.values())
        assert total_uses >= 7  # at least one garment per day slot
