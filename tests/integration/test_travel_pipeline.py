"""Integration tests — Travel Pipeline (TravelWardrobePlanner end-to-end).

These tests verify that the full travel planning pipeline runs correctly
from raw wardrobe → constraints → packing plan → packing score.
"""
import pytest
from typing import List
from uuid import uuid4

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory, ColorProfile,
    FormalityLevel, Season, PatternInfo, MaterialProfile, SeasonalityInfo,
)
from src.core.travel_models import TravelConstraints, PackingPlan
from src.layer2_style.travel.travel_wardrobe_planner import TravelWardrobePlanner
from src.layer2_style.travel.packing_score_calculator import PackingScoreCalculator
from src.layer2_style.travel.seasonal_audit_engine import SeasonalAuditEngine
from src.layer3_context.travel.constraint_parser import ConstraintParser


def _make_garment(gid, cat, color, formality, material, seasons, style_tags=None):
    sl = seasons or [Season.SPRING, Season.SUMMER]
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
def rich_wardrobe() -> List[Garment]:
    return [
        _make_garment("t1", GarmentCategory.TOP, "white", FormalityLevel.CASUAL, "cotton",
                      [Season.SPRING, Season.SUMMER], ["casual"]),
        _make_garment("t2", GarmentCategory.TOP, "navy", FormalityLevel.SMART_CASUAL, "cotton",
                      [Season.SPRING, Season.FALL], ["smart_casual"]),
        _make_garment("t3", GarmentCategory.TOP, "black", FormalityLevel.BUSINESS, "silk",
                      [Season.FALL, Season.WINTER], ["professional"]),
        _make_garment("t4", GarmentCategory.TOP, "striped", FormalityLevel.CASUAL, "linen",
                      [Season.SPRING, Season.SUMMER], ["casual", "beach"]),
        _make_garment("b1", GarmentCategory.BOTTOM, "blue", FormalityLevel.CASUAL, "denim",
                      [Season.SPRING, Season.FALL], ["casual"]),
        _make_garment("b2", GarmentCategory.BOTTOM, "grey", FormalityLevel.BUSINESS, "wool",
                      [Season.FALL, Season.WINTER], ["professional"]),
        _make_garment("b3", GarmentCategory.BOTTOM, "beige", FormalityLevel.CASUAL, "cotton",
                      [Season.SPRING, Season.SUMMER], ["casual"]),
        _make_garment("d1", GarmentCategory.DRESS, "rust", FormalityLevel.SMART_CASUAL, "linen",
                      [Season.SPRING, Season.SUMMER], ["casual", "evening"]),
        _make_garment("o1", GarmentCategory.OUTERWEAR, "camel", FormalityLevel.SMART_CASUAL, "wool",
                      [Season.FALL, Season.WINTER], ["casual", "layering"]),
        _make_garment("o2", GarmentCategory.OUTERWEAR, "black", FormalityLevel.BUSINESS, "cotton",
                      [Season.FALL, Season.WINTER], ["professional"]),
        _make_garment("s1", GarmentCategory.SHOES, "white", FormalityLevel.CASUAL, "leather",
                      [Season.SPRING, Season.SUMMER], ["casual"]),
        _make_garment("s2", GarmentCategory.SHOES, "brown", FormalityLevel.SMART_CASUAL, "leather",
                      [Season.FALL, Season.WINTER], ["professional"]),
        _make_garment("a1", GarmentCategory.ACCESSORY, "olive", FormalityLevel.CASUAL, "wool",
                      [Season.FALL, Season.WINTER], ["casual"]),
    ]


class TestFullTravelPipeline:
    def test_constraint_parsing_to_plan(self, rich_wardrobe):
        """End-to-end: parse constraints → plan packing list."""
        parser = ConstraintParser()
        constraints = parser.parse(
            destination="Rome",
            days=7,
            occasions="casual:4,evening:2,tourism:1",
            max_pieces=10,
            body_shape="pear",
            season="summer",
        )
        planner = TravelWardrobePlanner()
        plan = planner.plan(rich_wardrobe, constraints)

        assert isinstance(plan, PackingPlan)
        assert plan.destination == "rome"
        assert len(plan.packed_garments) <= 10
        assert plan.days == 7
        assert 0.0 <= plan.packing_score <= 100.0

    def test_day_plans_cover_all_days(self, rich_wardrobe):
        parser = ConstraintParser()
        constraints = parser.parse(
            destination="paris",
            days=5,
            occasions="work:3,casual:2",
            max_pieces=12,
        )
        planner = TravelWardrobePlanner()
        plan = planner.plan(rich_wardrobe, constraints)

        assert len(plan.day_plans) == 5

    def test_score_label_is_set(self, rich_wardrobe):
        constraints = TravelConstraints(
            destination="london",
            days=7,
            occasions={"casual": 5, "business": 2},
            max_pieces=12,
        )
        planner = TravelWardrobePlanner()
        plan = planner.plan(rich_wardrobe, constraints)

        assert plan.score_label in {"excellent", "good", "fair", "poor"}

    def test_occasion_coverage_in_result(self, rich_wardrobe):
        constraints = TravelConstraints(
            destination="tokyo",
            days=7,
            occasions={"casual": 4, "business": 3},
            max_pieces=10,
        )
        planner = TravelWardrobePlanner()
        plan = planner.plan(rich_wardrobe, constraints)

        assert "casual" in plan.occasion_coverage
        assert "business" in plan.occasion_coverage


class TestPipelineWithSeasonalAudit:
    def test_audit_then_plan(self, rich_wardrobe):
        """Audit wardrobe for season, then plan travel from remaining 'ready' pieces."""
        engine = SeasonalAuditEngine()
        audit = engine.audit(rich_wardrobe, "fall")

        # Get only ready/adaptable garments
        from src.core.travel_models import SeasonReadiness
        good_ids = {
            a.garment_id
            for a in audit.garment_audits
            if a.readiness != SeasonReadiness.STORE
        }
        good_garments = [g for g in rich_wardrobe if g.id in good_ids]

        constraints = TravelConstraints(
            destination="paris",
            days=5,
            occasions={"casual": 3, "evening": 2},
            max_pieces=8,
            season="fall",
        )
        planner = TravelWardrobePlanner()
        plan = planner.plan(good_garments, constraints)

        assert isinstance(plan, PackingPlan)
        # All packed garments should be from the 'good' set
        packed_ids = {pg.garment_id for pg in plan.packed_garments}
        assert packed_ids <= good_ids

    def test_audit_readiness_percentage_positive(self, rich_wardrobe):
        engine = SeasonalAuditEngine()
        for season in ["spring", "summer", "fall", "winter"]:
            result = engine.audit(rich_wardrobe, season)
            assert result.overall_readiness_pct >= 0.0
            assert result.overall_readiness_pct <= 100.0


class TestPackingScoreIntegration:
    def test_packing_score_uses_real_garments(self, rich_wardrobe):
        calc = PackingScoreCalculator()
        score = calc(rich_wardrobe, {"casual": 5, "evening": 2}, colour_cohesion=0.75)
        assert 0.0 <= score <= 100.0

    def test_versatility_ratio_with_full_wardrobe(self, rich_wardrobe):
        calc = PackingScoreCalculator()
        vr = calc.versatility_ratio(rich_wardrobe)
        # With tops, bottoms, dresses, shoes — should have > 1 outfit per piece
        assert vr > 1.0
