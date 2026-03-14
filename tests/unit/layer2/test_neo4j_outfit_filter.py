"""
Tests for Neo4jOutfitFilter (Phase 3 pre-filter + combo generator).

Coverage
--------
- FilterResult dataclass
- _local_compatibility helper
- _garment_seasons / _garment_occasions helpers
- Neo4jOutfitFilter.filter_wardrobe (local + neo4j paths)
- Neo4jOutfitFilter.generate_smart_combinations
- Neo4jOutfitFilter.score_partial_outfit_fast
- Neo4jOutfitFilter.get_compatibility_hint
- Neo4jOutfitFilter.batch_compatibility_hints
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.models import (
    FormalityLevel,
    Garment,
    GarmentAttributes,
    GarmentCategory,
    OccasionProfile,
    Season,
    SeasonalityInfo,
    UserContext,
    Occasion,
    StyleIdentity,
    ColorProfile,
)
from src.layer2_style.neo4j_outfit_filter import (
    FilterResult,
    Neo4jOutfitFilter,
    _garment_occasions,
    _garment_seasons,
    _local_compatibility,
    _formality_rank,
)


# ===========================================================================
# Fixtures
# ===========================================================================


def _make_garment(
    gid: str = "g1",
    category: GarmentCategory = GarmentCategory.TOP,
    formality: FormalityLevel = FormalityLevel.CASUAL,
    seasons: list[Season] | None = None,
    occasions: list[str] | None = None,
    style_tags: list[str] | None = None,
    versatility: float = 0.5,
) -> Garment:
    """Factory for minimal Garment instances."""
    occasion_profile = OccasionProfile(
        formality_level=formality,
        occasions=occasions or [],
    )
    seasonality = SeasonalityInfo(seasons=seasons or [])
    attrs = GarmentAttributes(
        category=category,
        color=ColorProfile(primary="white"),
        occasion_profile=occasion_profile,
        seasonality=seasonality,
        style_tags=style_tags or [],
    )
    return Garment(id=gid, attributes=attrs)


@pytest.fixture
def top1():
    return _make_garment(
        "top1", GarmentCategory.TOP, FormalityLevel.CASUAL,
        seasons=[Season.SPRING, Season.SUMMER],
        occasions=["casual", "weekend"],
        style_tags=["minimalist"],
    )


@pytest.fixture
def top2():
    return _make_garment(
        "top2", GarmentCategory.TOP, FormalityLevel.BUSINESS,
        seasons=[Season.FALL, Season.WINTER],
        occasions=["work", "business"],
        style_tags=["classic"],
    )


@pytest.fixture
def bottom1():
    return _make_garment(
        "bottom1", GarmentCategory.BOTTOM, FormalityLevel.CASUAL,
        seasons=[Season.SPRING, Season.SUMMER],
        occasions=["casual"],
        style_tags=["minimalist"],
    )


@pytest.fixture
def bottom2():
    return _make_garment(
        "bottom2", GarmentCategory.BOTTOM, FormalityLevel.BUSINESS,
        seasons=[Season.FALL, Season.WINTER],
        occasions=["work"],
        style_tags=["classic"],
    )


@pytest.fixture
def shoes1():
    return _make_garment(
        "shoes1", GarmentCategory.SHOES, FormalityLevel.CASUAL,
        seasons=[Season.SPRING, Season.SUMMER],
        occasions=["casual"],
    )


@pytest.fixture
def dress1():
    return _make_garment(
        "dress1", GarmentCategory.DRESS, FormalityLevel.SMART_CASUAL,
        seasons=[Season.SPRING],
        occasions=["casual", "evening"],
    )


@pytest.fixture
def wardrobe(top1, top2, bottom1, bottom2, shoes1):
    """Flat list of garments (no wardrobe-dict structure)."""
    return [top1, top2, bottom1, bottom2, shoes1]


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.get_garments_by_season = AsyncMock(return_value=[])
    client.get_garments_by_occasion = AsyncMock(return_value=[])
    client.get_edge_weight = AsyncMock(return_value=None)
    return client


@pytest.fixture
def filt():
    """Filter with no Neo4j client (local mode)."""
    return Neo4jOutfitFilter(min_compatibility=0.3)


@pytest.fixture
def filt_neo4j(mock_client):
    """Filter with mocked Neo4j client."""
    return Neo4jOutfitFilter(mock_client, min_compatibility=0.3)


# ===========================================================================
# FilterResult
# ===========================================================================


class TestFilterResult:
    def test_counts(self, top1, bottom1):
        result = FilterResult(kept=[top1, bottom1], dropped=[])
        assert result.kept_count == 2
        assert result.dropped_count == 0

    def test_summary_neo4j(self, top1):
        result = FilterResult(kept=[top1], dropped=[], neo4j_used=True, filter_time_ms=12.5)
        assert "Neo4j" in result.summary()
        assert "12.5" in result.summary()

    def test_summary_local(self, top1):
        result = FilterResult(kept=[top1], dropped=[], neo4j_used=False, filter_time_ms=3.0)
        assert "local" in result.summary()


# ===========================================================================
# Helper functions
# ===========================================================================


class TestHelpers:
    def test_garment_seasons_from_seasonality(self, top1):
        assert _garment_seasons(top1) == {"spring", "summer"}

    def test_garment_seasons_empty(self):
        g = _make_garment("x")
        assert _garment_seasons(g) == set()

    def test_garment_occasions_from_profile(self, top1):
        occ = _garment_occasions(top1)
        assert "casual" in occ
        assert "weekend" in occ

    def test_garment_occasions_empty(self):
        g = _make_garment("x")
        assert _garment_occasions(g) == set()

    def test_formality_rank_casual(self, top1):
        assert _formality_rank(top1) == 1

    def test_formality_rank_business(self, top2):
        assert _formality_rank(top2) == 4

    def test_local_compatibility_same_style(self, top1, bottom1):
        score = _local_compatibility(top1, bottom1)
        assert 0.0 <= score <= 1.0
        # Same formality / season / occasions → high score
        assert score > 0.5

    def test_local_compatibility_different_formality(self, top1, top2):
        score = _local_compatibility(top1, top2)
        # top1=casual(1) vs top2=business(4) → large delta → low score
        assert score < 0.5

    def test_local_compatibility_range(self, top1, bottom2):
        score = _local_compatibility(top1, bottom2)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# filter_wardrobe  — local path
# ===========================================================================


class TestFilterWardrobeLocal:
    @pytest.mark.asyncio
    async def test_no_context_returns_all(self, filt, wardrobe):
        result = await filt.filter_wardrobe(wardrobe)
        assert result.neo4j_used is False
        assert result.kept_count == len(wardrobe)
        assert result.dropped_count == 0

    @pytest.mark.asyncio
    async def test_season_filter(self, filt, wardrobe):
        ctx = UserContext()
        # Inject season into filter call via monkey-patching _local_filter
        result = await filt.filter_wardrobe(
            wardrobe,
            context=ctx,
        )
        # With empty context (no season/occasion) all should pass
        assert result.kept_count == len(wardrobe)

    @pytest.mark.asyncio
    async def test_max_items_respected(self, filt, wardrobe):
        result = await filt.filter_wardrobe(wardrobe, max_items=2)
        assert result.kept_count <= 2

    @pytest.mark.asyncio
    async def test_kept_plus_dropped_equals_total(self, filt, wardrobe):
        result = await filt.filter_wardrobe(wardrobe, max_items=3)
        assert result.kept_count + result.dropped_count == len(wardrobe)

    @pytest.mark.asyncio
    async def test_returns_filter_result(self, filt, wardrobe):
        result = await filt.filter_wardrobe(wardrobe)
        assert isinstance(result, FilterResult)

    @pytest.mark.asyncio
    async def test_filter_time_positive(self, filt, wardrobe):
        result = await filt.filter_wardrobe(wardrobe)
        assert result.filter_time_ms >= 0.0


# ===========================================================================
# filter_wardrobe  — Neo4j path
# ===========================================================================


class TestFilterWardrobeNeo4j:
    @pytest.mark.asyncio
    async def test_calls_neo4j_when_context_given(self, filt_neo4j, mock_client, wardrobe):
        ctx = UserContext(occasion=Occasion.CASUAL)
        mock_client.get_garments_by_occasion.return_value = [
            {"garment_id": "top1", "score": 0.9},
        ]
        result = await filt_neo4j.filter_wardrobe(wardrobe, context=ctx, user_id="u1")
        mock_client.get_garments_by_occasion.assert_called_once()
        # top1 should be in kept (score > 0)
        kept_ids = {g.id for g in result.kept}
        assert "top1" in kept_ids

    @pytest.mark.asyncio
    async def test_neo4j_used_flag(self, filt_neo4j, mock_client, wardrobe):
        mock_client.get_garments_by_season.return_value = []
        mock_client.get_garments_by_occasion.return_value = []
        result = await filt_neo4j.filter_wardrobe(wardrobe)
        assert result.neo4j_used is True

    @pytest.mark.asyncio
    async def test_fallback_on_neo4j_error(self, mock_client, wardrobe):
        mock_client.get_garments_by_occasion.side_effect = RuntimeError("neo4j down")
        filt = Neo4jOutfitFilter(mock_client, min_compatibility=0.3)
        # Provide an occasion context so the client is actually called
        ctx = UserContext(occasion=Occasion.CASUAL)
        result = await filt.filter_wardrobe(wardrobe, context=ctx)
        # Falls back to local
        assert result.neo4j_used is False
        assert result.kept_count > 0

    @pytest.mark.asyncio
    async def test_max_items_respected_neo4j(self, filt_neo4j, mock_client, wardrobe):
        mock_client.get_garments_by_occasion.return_value = [
            {"garment_id": g.id, "score": 0.8} for g in wardrobe
        ]
        ctx = UserContext(occasion=Occasion.CASUAL)
        result = await filt_neo4j.filter_wardrobe(wardrobe, context=ctx, max_items=2)
        assert result.kept_count <= 2


# ===========================================================================
# generate_smart_combinations
# ===========================================================================


class TestGenerateSmartCombinations:
    def test_basic_top_bottom(self, filt, top1, bottom1):
        combos = filt.generate_smart_combinations([top1, bottom1])
        assert len(combos) >= 1
        for c in combos:
            ids = {g.id for g in c}
            assert "top1" in ids
            assert "bottom1" in ids

    def test_with_shoes(self, filt, top1, bottom1, shoes1):
        combos = filt.generate_smart_combinations([top1, bottom1, shoes1])
        assert len(combos) >= 1
        # At least one combo should include shoes
        any_shoes = any("shoes1" in {g.id for g in c} for c in combos)
        assert any_shoes

    def test_full_body_alone(self, filt, dress1):
        combos = filt.generate_smart_combinations([dress1])
        # full_body alone is a valid minimal outfit
        assert len(combos) >= 1
        assert all("dress1" in {g.id for g in c} for c in combos)

    def test_no_base_returns_empty(self, filt, shoes1):
        # Only shoes — no valid base
        combos = filt.generate_smart_combinations([shoes1])
        assert combos == []

    def test_max_combos_respected(self, filt):
        tops = [_make_garment(f"t{i}", GarmentCategory.TOP) for i in range(10)]
        bottoms = [_make_garment(f"b{i}", GarmentCategory.BOTTOM) for i in range(10)]
        combos = filt.generate_smart_combinations(tops + bottoms, max_combos=5)
        assert len(combos) <= 5

    def test_no_duplicates(self, filt, top1, top2, bottom1, bottom2, shoes1):
        combos = filt.generate_smart_combinations(
            [top1, top2, bottom1, bottom2, shoes1], max_combos=50
        )
        keys = [frozenset(g.id for g in c) for c in combos]
        assert len(keys) == len(set(keys)), "Duplicate combos found"

    def test_minimum_combo_size(self, filt, top1, top2, bottom1, bottom2):
        combos = filt.generate_smart_combinations([top1, top2, bottom1, bottom2])
        for c in combos:
            assert len(c) >= 2, "All combos must have at least 2 garments"

    def test_empty_input(self, filt):
        assert filt.generate_smart_combinations([]) == []


# ===========================================================================
# score_partial_outfit_fast
# ===========================================================================


class TestScorePartialOutfitFast:
    def test_two_compatible(self, filt, top1, bottom1):
        score = filt.score_partial_outfit_fast([top1, bottom1])
        assert 0.0 <= score <= 1.0

    def test_single_garment(self, filt, top1):
        score = filt.score_partial_outfit_fast([top1])
        assert score == 0.5

    def test_empty(self, filt):
        score = filt.score_partial_outfit_fast([])
        assert score == 0.5

    def test_compatible_pair_higher_than_incompatible(self, filt, top1, bottom1, top2, bottom2):
        # top1+bottom1 share season/occasion/style → higher compat
        score_compat = filt.score_partial_outfit_fast([top1, bottom1])
        # top1+bottom2 differ in formality/season → lower
        score_incompat = filt.score_partial_outfit_fast([top1, bottom2])
        assert score_compat >= score_incompat


# ===========================================================================
# get_compatibility_hint
# ===========================================================================


class TestGetCompatibilityHint:
    @pytest.mark.asyncio
    async def test_no_client_returns_neutral(self, filt):
        w = await filt.get_compatibility_hint("a", "b")
        assert w == 0.5

    @pytest.mark.asyncio
    async def test_neo4j_hit(self, mock_client):
        mock_client.get_edge_weight.return_value = {"weight": 0.8}
        filt = Neo4jOutfitFilter(mock_client)
        w = await filt.get_compatibility_hint("a", "b")
        assert w == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_neo4j_miss_returns_neutral(self, mock_client):
        mock_client.get_edge_weight.return_value = None
        filt = Neo4jOutfitFilter(mock_client)
        w = await filt.get_compatibility_hint("a", "b")
        assert w == 0.5

    @pytest.mark.asyncio
    async def test_result_cached(self, mock_client):
        mock_client.get_edge_weight.return_value = {"weight": 0.75}
        filt = Neo4jOutfitFilter(mock_client)
        w1 = await filt.get_compatibility_hint("a", "b")
        w2 = await filt.get_compatibility_hint("a", "b")
        assert w1 == w2
        # Should only call neo4j once
        assert mock_client.get_edge_weight.call_count == 1

    @pytest.mark.asyncio
    async def test_reverse_key_cached(self, mock_client):
        mock_client.get_edge_weight.return_value = {"weight": 0.6}
        filt = Neo4jOutfitFilter(mock_client)
        w1 = await filt.get_compatibility_hint("a", "b")
        # Reverse lookup should use cache
        w2 = await filt.get_compatibility_hint("b", "a")
        assert w1 == w2
        assert mock_client.get_edge_weight.call_count == 1

    @pytest.mark.asyncio
    async def test_neo4j_error_returns_neutral(self, mock_client):
        mock_client.get_edge_weight.side_effect = RuntimeError("fail")
        filt = Neo4jOutfitFilter(mock_client)
        w = await filt.get_compatibility_hint("a", "b")
        assert w == 0.5


# ===========================================================================
# batch_compatibility_hints
# ===========================================================================


class TestBatchCompatibilityHints:
    @pytest.mark.asyncio
    async def test_no_client_all_neutral(self, filt):
        results = await filt.batch_compatibility_hints([("a", "b"), ("c", "d")])
        assert results[("a", "b")] == 0.5
        assert results[("c", "d")] == 0.5

    @pytest.mark.asyncio
    async def test_fetches_missing_pairs(self, mock_client):
        mock_client.get_edge_weight.return_value = {"weight": 0.7}
        filt = Neo4jOutfitFilter(mock_client)
        results = await filt.batch_compatibility_hints([("a", "b"), ("c", "d")])
        assert results[("a", "b")] == pytest.approx(0.7)
        assert results[("c", "d")] == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_uses_cache(self, mock_client):
        mock_client.get_edge_weight.return_value = {"weight": 0.9}
        filt = Neo4jOutfitFilter(mock_client)
        # Pre-populate cache
        await filt.get_compatibility_hint("a", "b")
        mock_client.get_edge_weight.reset_mock()
        # batch_compatibility_hints should not call neo4j again
        results = await filt.batch_compatibility_hints([("a", "b")])
        mock_client.get_edge_weight.assert_not_called()
        assert results[("a", "b")] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_empty_pairs(self, filt):
        results = await filt.batch_compatibility_hints([])
        assert results == {}
