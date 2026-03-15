"""Tests — SeasonalAuditEngine"""
import pytest
from tests.unit.layer2.travel.fixtures import _g, make_travel_wardrobe
from src.core.models import GarmentCategory, FormalityLevel, Season, MaterialProfile

from src.layer2_style.travel.seasonal_audit_engine import SeasonalAuditEngine
from src.core.travel_models import SeasonReadiness


@pytest.fixture
def engine():
    return SeasonalAuditEngine()


@pytest.fixture
def wardrobe():
    return make_travel_wardrobe()


class TestReturnTypes:
    def test_returns_season_audit_result(self, engine, wardrobe):
        from src.core.travel_models import SeasonAuditResult
        result = engine.audit(wardrobe, "fall")
        assert isinstance(result, SeasonAuditResult)

    def test_all_garments_audited(self, engine, wardrobe):
        result = engine.audit(wardrobe, "fall")
        assert len(result.garment_audits) == len(wardrobe)

    def test_counts_sum_to_total(self, engine, wardrobe):
        result = engine.audit(wardrobe, "fall")
        total = result.ready_count + result.adaptable_count + result.store_count
        assert total == len(wardrobe)


class TestReadinessValues:
    def test_readiness_is_valid_enum(self, engine, wardrobe):
        result = engine.audit(wardrobe, "summer")
        for audit in result.garment_audits:
            assert audit.readiness in SeasonReadiness

    def test_score_in_range(self, engine, wardrobe):
        result = engine.audit(wardrobe, "fall")
        for audit in result.garment_audits:
            assert 0.0 <= audit.score <= 1.0


class TestSeasonSpecifics:
    def test_linen_tops_ready_for_summer(self, engine):
        linen_top = _g("l1", GarmentCategory.TOP, material="linen",
                       seasons=[Season.SUMMER])
        result = engine.audit([linen_top], "summer")
        audit = result.garment_audits[0]
        # Linen should be READY or ADAPTABLE for summer (not store)
        assert audit.readiness != SeasonReadiness.STORE

    def test_heavy_wool_penalised_for_summer(self, engine):
        wool_coat = _g("w1", GarmentCategory.OUTERWEAR, material="heavy_wool",
                       seasons=[Season.WINTER])
        result = engine.audit([wool_coat], "summer")
        audit = result.garment_audits[0]
        # heavy_wool should score low for summer
        assert audit.score < 0.7

    def test_denim_neutral_for_fall(self, engine):
        jeans = _g("j1", GarmentCategory.BOTTOM, material="denim",
                   seasons=[Season.FALL, Season.SPRING])
        result = engine.audit([jeans], "fall")
        audit = result.garment_audits[0]
        assert audit.readiness in {SeasonReadiness.READY, SeasonReadiness.ADAPTABLE}


class TestOverallReadiness:
    def test_overall_readiness_is_percentage(self, engine, wardrobe):
        result = engine.audit(wardrobe, "fall")
        assert 0.0 <= result.overall_readiness_pct <= 100.0

    def test_empty_wardrobe_zero_readiness(self, engine):
        result = engine.audit([], "fall")
        assert result.overall_readiness_pct == 0.0
        assert result.ready_count == 0


class TestCoverage:
    def test_coverage_items_returned(self, engine, wardrobe):
        result = engine.audit(wardrobe, "fall")
        # Should have coverage for key categories
        assert isinstance(result.coverage_by_category, list)

    def test_gaps_are_strings(self, engine):
        # Wardrobe with only tops — should have gaps for bottom, shoes
        tops_only = [
            _g(f"t{i}", GarmentCategory.TOP, seasons=[Season.FALL])
            for i in range(2)
        ]
        result = engine.audit(tops_only, "fall")
        # Gaps should exist
        for gap in result.top_gaps:
            assert isinstance(gap, str)
            assert len(gap) > 0
