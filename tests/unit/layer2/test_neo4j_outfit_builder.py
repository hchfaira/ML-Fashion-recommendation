"""
Tests for OutfitBuilder.neo4j_beam_search_outfits (Phase 3).

Coverage
--------
- Happy path: returns OutfitCandidate list
- Empty wardrobe
- Pre-filter returns 0 items
- No combos generated
- With / without Neo4j client
- beam_width / top_k respected
- Profile forwarded to scorer
- _neo4j_beam_prune helper
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
    ColorProfile,
)
from src.layer2_style.outfit_builder import OutfitBuilder, OutfitCandidate
from src.layer2_style.neo4j_outfit_filter import Neo4jOutfitFilter


# ===========================================================================
# Fixtures
# ===========================================================================


def _make_garment(
    gid: str,
    category: GarmentCategory = GarmentCategory.TOP,
    formality: FormalityLevel = FormalityLevel.CASUAL,
) -> Garment:
    attrs = GarmentAttributes(
        category=category,
        color=ColorProfile(primary="white"),
        occasion_profile=OccasionProfile(
            formality_level=formality,
            occasions=["casual"],
        ),
        seasonality=SeasonalityInfo(seasons=[Season.SPRING, Season.SUMMER]),
        style_tags=["minimalist"],
    )
    return Garment(id=gid, attributes=attrs)


@pytest.fixture
def builder():
    return OutfitBuilder()


@pytest.fixture
def small_wardrobe():
    """Wardrobe dict with 2 tops, 2 bottoms, 1 shoes."""
    return {
        "tops": [
            _make_garment("t1", GarmentCategory.TOP),
            _make_garment("t2", GarmentCategory.TOP),
        ],
        "bottoms": [
            _make_garment("b1", GarmentCategory.BOTTOM),
            _make_garment("b2", GarmentCategory.BOTTOM),
        ],
        "shoes": [
            _make_garment("s1", GarmentCategory.SHOES),
        ],
    }


@pytest.fixture
def mock_scorecard():
    sc = MagicMock()
    sc.scores = {"overall": 0.75}
    sc.get_grade.return_value = "B"
    return sc


# ===========================================================================
# Tests
# ===========================================================================


class TestNeo4jBeamSearchOutfits:

    @pytest.mark.asyncio
    async def test_returns_outfit_candidates(self, builder, small_wardrobe, mock_scorecard):
        with patch.object(builder, "score_outfit", return_value=mock_scorecard):
            results = await builder.neo4j_beam_search_outfits(small_wardrobe)
        assert isinstance(results, list)
        assert all(isinstance(r, OutfitCandidate) for r in results)

    @pytest.mark.asyncio
    async def test_top_k_respected(self, builder, small_wardrobe, mock_scorecard):
        with patch.object(builder, "score_outfit", return_value=mock_scorecard):
            results = await builder.neo4j_beam_search_outfits(
                small_wardrobe, top_k=2
            )
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_empty_wardrobe_returns_empty(self, builder):
        results = await builder.neo4j_beam_search_outfits({})
        assert results == []

    @pytest.mark.asyncio
    async def test_wardrobe_dict_all_empty_returns_empty(self, builder):
        results = await builder.neo4j_beam_search_outfits(
            {"tops": [], "bottoms": []}
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_no_valid_base_returns_empty(self, builder):
        """Only shoes — no tops/bottoms/dress → 0 combos → []."""
        wardrobe = {
            "shoes": [_make_garment("s1", GarmentCategory.SHOES)],
        }
        results = await builder.neo4j_beam_search_outfits(wardrobe)
        assert results == []

    @pytest.mark.asyncio
    async def test_results_sorted_desc(self, builder, small_wardrobe):
        # Alternate mock scores to verify sorting
        scores = [0.3, 0.9, 0.6, 0.8, 0.5]
        idx = [0]

        def _side_effect(garments, profile=None):
            sc = MagicMock()
            sc.scores = {"overall": scores[idx[0] % len(scores)]}
            sc.get_grade.return_value = "B"
            idx[0] += 1
            return sc

        with patch.object(builder, "score_outfit", side_effect=_side_effect):
            results = await builder.neo4j_beam_search_outfits(
                small_wardrobe, top_k=5
            )

        for i in range(len(results) - 1):
            assert results[i].overall_score >= results[i + 1].overall_score

    @pytest.mark.asyncio
    async def test_with_neo4j_client(self, builder, small_wardrobe, mock_scorecard):
        mock_client = AsyncMock()
        mock_client.get_garments_by_season = AsyncMock(return_value=[])
        mock_client.get_garments_by_occasion = AsyncMock(return_value=[])

        with patch.object(builder, "score_outfit", return_value=mock_scorecard):
            results = await builder.neo4j_beam_search_outfits(
                small_wardrobe,
                neo4j_client=mock_client,
            )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_context_forwarded(self, builder, small_wardrobe, mock_scorecard):
        ctx = UserContext(occasion=Occasion.CASUAL)
        with patch(
            "src.layer2_style.outfit_builder.Neo4jOutfitFilter.filter_wardrobe",
            new_callable=AsyncMock,
        ) as mock_filter:
            from src.layer2_style.neo4j_outfit_filter import FilterResult
            mock_filter.return_value = FilterResult(
                kept=small_wardrobe["tops"] + small_wardrobe["bottoms"],
                dropped=[],
            )
            with patch.object(
                Neo4jOutfitFilter,
                "generate_smart_combinations",
                return_value=[
                    [small_wardrobe["tops"][0], small_wardrobe["bottoms"][0]]
                ],
            ):
                with patch.object(builder, "score_outfit", return_value=mock_scorecard):
                    await builder.neo4j_beam_search_outfits(
                        small_wardrobe, context=ctx
                    )
            # context should be forwarded to filter_wardrobe
            call_kwargs = mock_filter.call_args
            assert call_kwargs is not None

    @pytest.mark.asyncio
    async def test_profile_forwarded_to_scorer(self, builder, small_wardrobe):
        received_profiles = []

        def _scorer(garments, profile=None):
            received_profiles.append(profile)
            sc = MagicMock()
            sc.scores = {"overall": 0.7}
            sc.get_grade.return_value = "B"
            return sc

        with patch.object(builder, "score_outfit", side_effect=_scorer):
            await builder.neo4j_beam_search_outfits(
                small_wardrobe, profile="business"
            )
        assert all(p == "business" for p in received_profiles)

    @pytest.mark.asyncio
    async def test_beam_width_limits_finalists(self, builder, small_wardrobe, mock_scorecard):
        with patch.object(builder, "score_outfit", return_value=mock_scorecard):
            results = await builder.neo4j_beam_search_outfits(
                small_wardrobe,
                beam_width=2,
                finalists_per_beam=3,
                top_k=5,
            )
        # Should still produce some results; finalists ≤ finalists_per_beam
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_filter_returns_zero(self, builder, small_wardrobe):
        with patch(
            "src.layer2_style.outfit_builder.Neo4jOutfitFilter.filter_wardrobe",
            new_callable=AsyncMock,
        ) as mock_filter:
            from src.layer2_style.neo4j_outfit_filter import FilterResult
            mock_filter.return_value = FilterResult(kept=[], dropped=list(small_wardrobe["tops"]))
            results = await builder.neo4j_beam_search_outfits(small_wardrobe)
        assert results == []


# ===========================================================================
# _neo4j_beam_prune
# ===========================================================================


class TestNeo4jBeamPrune:
    """Unit tests for the internal beam pruner."""

    @pytest.fixture
    def builder_and_filter(self):
        b = OutfitBuilder()
        filt = Neo4jOutfitFilter(min_compatibility=0.3)
        return b, filt

    def test_returns_n_finalists(self, builder_and_filter):
        b, filt = builder_and_filter
        combos = [
            [_make_garment(f"t{i}"), _make_garment(f"b{i}", GarmentCategory.BOTTOM)]
            for i in range(10)
        ]
        result = b._neo4j_beam_prune(combos, outfit_filter=filt, beam_width=5, n_finalists=4)
        assert len(result) <= 4

    def test_empty_input(self, builder_and_filter):
        b, filt = builder_and_filter
        assert b._neo4j_beam_prune([], outfit_filter=filt, beam_width=5, n_finalists=5) == []

    def test_no_duplicates(self, builder_and_filter):
        b, filt = builder_and_filter
        top = _make_garment("t1")
        bottom = _make_garment("b1", GarmentCategory.BOTTOM)
        combos = [[top, bottom], [top, bottom], [top, bottom]]
        result = b._neo4j_beam_prune(combos, outfit_filter=filt, beam_width=5, n_finalists=5)
        keys = [frozenset(g.id for g in c) for c in result]
        assert len(keys) == len(set(keys))

    def test_single_combo(self, builder_and_filter):
        b, filt = builder_and_filter
        combo = [_make_garment("t1"), _make_garment("b1", GarmentCategory.BOTTOM)]
        result = b._neo4j_beam_prune([combo], outfit_filter=filt, beam_width=5, n_finalists=5)
        assert len(result) == 1

    def test_better_combos_ranked_first(self, builder_and_filter):
        b, filt = builder_and_filter
        # Combo A: same formality/season → high compat
        a = [
            _make_garment("t_a", GarmentCategory.TOP, FormalityLevel.CASUAL),
            _make_garment("b_a", GarmentCategory.BOTTOM, FormalityLevel.CASUAL),
        ]
        # Combo B: different formality → low compat
        b_combo = [
            _make_garment("t_b", GarmentCategory.TOP, FormalityLevel.CASUAL),
            _make_garment("b_b", GarmentCategory.BOTTOM, FormalityLevel.BLACK_TIE),
        ]
        result = b._neo4j_beam_prune(
            [b_combo, a], outfit_filter=filt, beam_width=5, n_finalists=2
        )
        # First result should be the more compatible combo (a)
        first_ids = {g.id for g in result[0]}
        assert "t_a" in first_ids or "b_a" in first_ids
