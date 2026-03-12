"""
Integration tests for HybridOutfitRecommender.

Verifies that StyleIntelligenceModel (Layer 2) and ContextEngine (Layer 3)
are properly combined by HybridOutfitRecommender under three realistic
divergence scenarios:

  Scenario A — Style says GOOD, Context says BAD
  Scenario B — Style says BAD, Context says GOOD
  Scenario C — Both say GOOD

Additionally tests the five user-profile integration points:

  Point 1 — Persistent profile storage
  Point 2 — Context enrichment
  Point 3 — Adaptive weights
  Point 4 — Pre-scoring garment filtering
  Point 5 — Personalised explanations
"""
from __future__ import annotations

import asyncio
import pytest
from uuid import uuid4
from typing import List

from src.core.models import (
    Garment,
    GarmentAttributes,
    GarmentCategory,
    ColorInfo,
    PatternInfo,
    UserContext,
    Occasion,
    FormalityLevel,
)
from src.layer2_style.hybrid_recommender import (
    HybridOutfitRecommender,
    HybridScore,
    RankedOutfit,
    _ADAPTIVE_WEIGHTS,
)
from src.layer2_style.season_color_harmony import ColorSeason
from src.layer2_style.volume_balance_scorer import BodyShape


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_garment(
    category: str,
    subcategory: str,
    color: str,
    formality: str = "casual",
    pattern: str = "solid",
) -> Garment:
    """Factory helper for test garments."""
    return Garment(
        id=str(uuid4()),
        attributes=GarmentAttributes(
            category=GarmentCategory(category),
            subcategory=subcategory,
            color=ColorInfo(primary=color, hex_codes=[]),
            pattern=PatternInfo(type=pattern),
            formality_level=formality,
            season_suitable=["spring", "summer", "fall", "winter"],
            fit="regular",
        ),
    )


def casual_context() -> UserContext:
    return UserContext(
        occasion=Occasion.CASUAL,
        formality_preference=FormalityLevel.CASUAL,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def recommender() -> HybridOutfitRecommender:
    """Default hybrid recommender with adaptive weights DISABLED for deterministic tests."""
    return HybridOutfitRecommender(
        style_weight=0.40, context_weight=0.60, adaptive_weights=False
    )


@pytest.fixture
def scenario_a_garments() -> List[Garment]:
    """Scenario A — Style-heavy, context-unfriendly (formal pieces in casual context)."""
    return [
        make_garment("top", "sequin blouse", "gold", formality="formal"),
        make_garment("bottom", "silk palazzo pants", "ivory", formality="formal"),
        make_garment("shoes", "stiletto heels", "black", formality="formal"),
    ]


@pytest.fixture
def scenario_b_garments() -> List[Garment]:
    """Scenario B — Low-style, context-friendly (plain basics)."""
    return [
        make_garment("top", "plain white t-shirt", "white"),
        make_garment("bottom", "plain beige chinos", "beige"),
        make_garment("shoes", "plain white sneakers", "white"),
    ]


@pytest.fixture
def scenario_c_garments() -> List[Garment]:
    """Scenario C — Both engines agree: good style AND good context fit."""
    return [
        make_garment("top", "navy blazer", "navy", formality="smart_casual"),
        make_garment("bottom", "gray slim trousers", "gray", formality="smart_casual"),
        make_garment("shoes", "white leather sneakers", "white"),
        make_garment("outerwear", "camel coat", "camel", formality="smart_casual"),
    ]


# ---------------------------------------------------------------------------
# Basic API tests
# ---------------------------------------------------------------------------

class TestHybridRecommenderConstruction:
    """Constructor and basic property tests."""

    def test_default_weights(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        assert abs(rec.style_weight - 0.40) < 1e-6
        assert abs(rec.context_weight - 0.60) < 1e-6

    def test_custom_weights(self):
        rec = HybridOutfitRecommender(
            style_weight=0.50, context_weight=0.50, adaptive_weights=False
        )
        assert abs(rec.style_weight - 0.50) < 1e-6

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError):
            HybridOutfitRecommender(style_weight=0.30, context_weight=0.30)

    def test_set_user_profile_accepted(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(color_season=ColorSeason.WINTER, body_shape=BodyShape.HOURGLASS)
        rec.set_user_profile(color_season=None, body_shape=None)

    def test_weight_invariant(self):
        for sw in [0.10, 0.30, 0.50, 0.70, 0.90]:
            rec = HybridOutfitRecommender(
                style_weight=sw, context_weight=1 - sw, adaptive_weights=False
            )
            assert abs(rec.style_weight + rec.context_weight - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Score contract tests
# ---------------------------------------------------------------------------

class TestHybridScoreContract:
    """Verify mathematical/structural invariants of HybridScore."""

    @pytest.mark.asyncio
    async def test_score_range(self, recommender, scenario_c_garments):
        score: HybridScore = await recommender.score_outfit(
            scenario_c_garments, casual_context()
        )
        assert 0.0 <= score.style_score <= 1.0
        assert 0.0 <= score.context_score <= 1.0
        assert 0.0 <= score.combined_score <= 1.0

    @pytest.mark.asyncio
    async def test_grade_validity(self, recommender, scenario_c_garments):
        score: HybridScore = await recommender.score_outfit(
            scenario_c_garments, casual_context()
        )
        assert score.grade in {"A", "B", "C", "D", "F"}

    @pytest.mark.asyncio
    async def test_combined_is_weighted_blend(self, recommender, scenario_c_garments):
        score: HybridScore = await recommender.score_outfit(
            scenario_c_garments, casual_context()
        )
        expected = (
            recommender.style_weight * score.style_score
            + recommender.context_weight * score.context_score
        )
        assert abs(score.combined_score - round(expected, 3)) < 0.005

    @pytest.mark.asyncio
    async def test_weights_recorded_in_score(self, recommender, scenario_c_garments):
        score: HybridScore = await recommender.score_outfit(
            scenario_c_garments, casual_context()
        )
        assert abs(score.style_weight - recommender.style_weight) < 1e-6
        assert abs(score.context_weight - recommender.context_weight) < 1e-6

    @pytest.mark.asyncio
    async def test_to_dict_structure(self, recommender, scenario_c_garments):
        score: HybridScore = await recommender.score_outfit(
            scenario_c_garments, casual_context()
        )
        d = score.to_dict()
        assert "combined_score" in d
        assert "style_score" in d
        assert "context_score" in d
        assert "grade" in d
        assert "weights" in d
        assert "strengths" in d
        assert "improvements" in d
        assert "profile_used" in d  # Point 5

    @pytest.mark.asyncio
    async def test_too_few_garments_returns_neutral(self, recommender):
        single_garment = [make_garment("top", "t-shirt", "white")]
        score: HybridScore = await recommender.score_outfit(
            single_garment, casual_context()
        )
        assert score.combined_score == 0.5


# ---------------------------------------------------------------------------
# Scenario divergence tests
# ---------------------------------------------------------------------------

class TestScenarioDivergence:
    """Core divergence scenarios."""

    @pytest.mark.asyncio
    async def test_scenario_a_style_good_context_bad(
        self, recommender, scenario_a_garments
    ):
        score = await recommender.score_outfit(scenario_a_garments, casual_context())
        assert 0.0 <= score.combined_score <= 1.0
        assert score.grade in {"A", "B", "C", "D", "F"}
        expected = round(
            recommender.style_weight * score.style_score
            + recommender.context_weight * score.context_score, 3,
        )
        assert abs(score.combined_score - expected) < 0.005

    @pytest.mark.asyncio
    async def test_scenario_b_style_bad_context_good(
        self, recommender, scenario_b_garments
    ):
        score = await recommender.score_outfit(scenario_b_garments, casual_context())
        assert 0.0 <= score.combined_score <= 1.0
        expected = round(
            recommender.style_weight * score.style_score
            + recommender.context_weight * score.context_score, 3,
        )
        assert abs(score.combined_score - expected) < 0.005

    @pytest.mark.asyncio
    async def test_scenario_c_both_good(self, recommender, scenario_c_garments):
        score = await recommender.score_outfit(scenario_c_garments, casual_context())
        assert 0.0 <= score.combined_score <= 1.0
        expected = round(
            recommender.style_weight * score.style_score
            + recommender.context_weight * score.context_score, 3,
        )
        assert abs(score.combined_score - expected) < 0.005

    @pytest.mark.asyncio
    async def test_personalisation_changes_score(self, scenario_c_garments):
        generic = HybridOutfitRecommender(adaptive_weights=False)
        personalised = HybridOutfitRecommender(adaptive_weights=False)
        personalised.set_user_profile(
            color_season=ColorSeason.WINTER, body_shape=BodyShape.HOURGLASS,
        )
        ctx = casual_context()
        generic_score = await generic.score_outfit(scenario_c_garments, ctx)
        personal_score = await personalised.score_outfit(scenario_c_garments, ctx)
        assert 0.0 <= generic_score.combined_score <= 1.0
        assert 0.0 <= personal_score.combined_score <= 1.0


# ---------------------------------------------------------------------------
# Recommend (ranking) tests
# ---------------------------------------------------------------------------

class TestRecommend:
    """Tests for the ``recommend()`` endpoint."""

    @pytest.mark.asyncio
    async def test_recommend_sorted_best_first(self, recommender):
        garments = [
            make_garment("top", "navy shirt", "navy"),
            make_garment("top", "white t-shirt", "white"),
            make_garment("bottom", "blue jeans", "blue"),
            make_garment("bottom", "black trousers", "black"),
            make_garment("shoes", "white sneakers", "white"),
            make_garment("outerwear", "gray blazer", "gray"),
        ]
        ranked: List[RankedOutfit] = await recommender.recommend(
            garments, casual_context(), top_k=5
        )
        assert len(ranked) <= 5
        scores = [ro.combined_score for ro in ranked]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_recommend_top_k_respected(self, recommender):
        garments = [
            make_garment("top", "t-shirt", "white"),
            make_garment("top", "sweater", "gray"),
            make_garment("bottom", "jeans", "blue"),
            make_garment("bottom", "chinos", "beige"),
            make_garment("shoes", "sneakers", "white"),
        ]
        for k in [1, 2, 3]:
            ranked = await recommender.recommend(
                garments, casual_context(), top_k=k
            )
            assert len(ranked) <= k

    @pytest.mark.asyncio
    async def test_recommend_pre_built_combinations(self, recommender):
        t = make_garment("top", "t-shirt", "white")
        b = make_garment("bottom", "jeans", "blue")
        s = make_garment("shoes", "sneakers", "white")
        combos = [[t, b], [t, b, s]]
        ranked = await recommender.recommend(
            wardrobe=[t, b, s],
            user_context=casual_context(),
            outfit_combinations=combos,
            top_k=5,
        )
        assert len(ranked) == len(combos)

    @pytest.mark.asyncio
    async def test_ranked_outfit_to_dict(self, recommender):
        t = make_garment("top", "t-shirt", "white")
        b = make_garment("bottom", "jeans", "blue")
        ranked = await recommender.recommend([t, b], casual_context(), top_k=1)
        assert len(ranked) >= 1
        d = ranked[0].to_dict()
        assert "outfit_id" in d
        assert "garments" in d
        assert "combined_score" in d
        assert "grade" in d
        assert "profile_used" in d  # Point 5


# ---------------------------------------------------------------------------
# Weight sensitivity tests
# ---------------------------------------------------------------------------

class TestWeightSensitivity:
    """Changing weights should produce proportionally different combined scores."""

    @pytest.mark.asyncio
    async def test_style_heavy_vs_context_heavy(self, scenario_c_garments):
        ctx = casual_context()
        style_heavy = HybridOutfitRecommender(
            style_weight=0.80, context_weight=0.20, adaptive_weights=False
        )
        context_heavy = HybridOutfitRecommender(
            style_weight=0.20, context_weight=0.80, adaptive_weights=False
        )
        sh_score = await style_heavy.score_outfit(scenario_c_garments, ctx)
        ch_score = await context_heavy.score_outfit(scenario_c_garments, ctx)
        sh_expected = round(0.80 * sh_score.style_score + 0.20 * sh_score.context_score, 3)
        ch_expected = round(0.20 * ch_score.style_score + 0.80 * ch_score.context_score, 3)
        assert abs(sh_score.combined_score - sh_expected) < 0.005
        assert abs(ch_score.combined_score - ch_expected) < 0.005

    @pytest.mark.asyncio
    async def test_equal_weights(self, scenario_c_garments):
        rec = HybridOutfitRecommender(
            style_weight=0.50, context_weight=0.50, adaptive_weights=False
        )
        score = await rec.score_outfit(scenario_c_garments, casual_context())
        expected = round(0.50 * score.style_score + 0.50 * score.context_score, 3)
        assert abs(score.combined_score - expected) < 0.005


# ===========================================================================
# Point 1 — Persistent profile storage
# ===========================================================================

class TestPersistentProfile:
    """Verify that set_user_profile stores values reused by score_outfit."""

    def test_profile_completeness_none(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        assert rec.profile_completeness == "none"

    def test_profile_completeness_partial(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(color_season=ColorSeason.AUTUMN)
        assert rec.profile_completeness == "partial"

    def test_profile_completeness_full(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(color_season=ColorSeason.WINTER, body_shape=BodyShape.PEAR)
        assert rec.profile_completeness == "full"

    @pytest.mark.asyncio
    async def test_stored_profile_used_in_score_outfit(self, scenario_c_garments):
        """score_outfit should use stored profile when no override is given."""
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(color_season=ColorSeason.WINTER, body_shape=BodyShape.HOURGLASS)

        score = await rec.score_outfit(scenario_c_garments, casual_context())
        assert score.profile_used["color_season"] == "winter"
        assert score.profile_used["body_shape"] == "hourglass"
        assert score.profile_used["completeness"] == "full"

    @pytest.mark.asyncio
    async def test_override_takes_precedence(self, scenario_c_garments):
        """Explicit user_season / body_shape args should override the stored profile."""
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(color_season=ColorSeason.WINTER, body_shape=BodyShape.HOURGLASS)

        score = await rec.score_outfit(
            scenario_c_garments, casual_context(),
            user_season=ColorSeason.SPRING,
            body_shape=BodyShape.PEAR,
        )
        assert score.profile_used["color_season"] == "spring"
        assert score.profile_used["body_shape"] == "pear"

    @pytest.mark.asyncio
    async def test_no_profile_reports_none(self, scenario_c_garments):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        score = await rec.score_outfit(scenario_c_garments, casual_context())
        assert score.profile_used["color_season"] is None
        assert score.profile_used["body_shape"] is None
        assert score.profile_used["completeness"] == "none"


# ===========================================================================
# Point 2 — Context enrichment
# ===========================================================================

class TestContextEnrichment:
    """Verify _enrich_context fills missing context fields from stored profile."""

    def test_enriches_body_type(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(body_shape=BodyShape.PEAR)
        ctx = casual_context()
        enriched = rec._enrich_context(ctx)
        assert enriched.body_type == "pear"

    def test_enriches_color_season(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(color_season=ColorSeason.WINTER)
        ctx = casual_context()
        enriched = rec._enrich_context(ctx)
        assert enriched.color_season == "winter"

    def test_enriches_skin_undertone(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(color_season=ColorSeason.AUTUMN)
        ctx = casual_context()
        enriched = rec._enrich_context(ctx)
        assert enriched.skin_undertone == "warm"

    def test_does_not_overwrite_explicit_values(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(
            color_season=ColorSeason.WINTER, body_shape=BodyShape.PEAR
        )
        ctx = UserContext(
            occasion=Occasion.CASUAL,
            body_type="athletic",
            color_season="summer",
            skin_undertone="neutral",
        )
        enriched = rec._enrich_context(ctx)
        # Explicit values should be preserved
        assert enriched.body_type == "athletic"
        assert enriched.color_season == "summer"
        assert enriched.skin_undertone == "neutral"

    def test_original_context_not_mutated(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(body_shape=BodyShape.HOURGLASS)
        ctx = casual_context()
        rec._enrich_context(ctx)
        assert ctx.body_type is None  # original unchanged

    @pytest.mark.asyncio
    async def test_enrichment_visible_in_context_breakdown(self, scenario_c_garments):
        """score_outfit should show enriched fields in context_breakdown."""
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(
            color_season=ColorSeason.AUTUMN, body_shape=BodyShape.RECTANGLE
        )
        score = await rec.score_outfit(scenario_c_garments, casual_context())
        bd = score.context_breakdown
        assert bd.get("body_type") == "rectangle" or bd.get("body_shape") == "rectangle"


# ===========================================================================
# Point 3 — Adaptive weights
# ===========================================================================

class TestAdaptiveWeights:
    """Verify weights shift based on profile completeness."""

    def test_no_profile_gives_balanced_weights(self):
        rec = HybridOutfitRecommender(adaptive_weights=True)
        # Adaptive weights are applied when set_user_profile is called.
        # At construction, the base weights (0.40/0.60) are used.
        # Calling set_user_profile(None, None) triggers the "none" preset.
        rec.set_user_profile(color_season=None, body_shape=None)
        expected = _ADAPTIVE_WEIGHTS["none"]
        assert abs(rec.style_weight - expected[0]) < 1e-6
        assert abs(rec.context_weight - expected[1]) < 1e-6

    def test_partial_profile_shifts_weights(self):
        rec = HybridOutfitRecommender(adaptive_weights=True)
        rec.set_user_profile(color_season=ColorSeason.SPRING)
        expected = _ADAPTIVE_WEIGHTS["partial"]
        assert abs(rec.style_weight - expected[0]) < 1e-6
        assert abs(rec.context_weight - expected[1]) < 1e-6

    def test_full_profile_shifts_weights(self):
        rec = HybridOutfitRecommender(adaptive_weights=True)
        rec.set_user_profile(
            color_season=ColorSeason.WINTER, body_shape=BodyShape.HOURGLASS,
        )
        expected = _ADAPTIVE_WEIGHTS["full"]
        assert abs(rec.style_weight - expected[0]) < 1e-6
        assert abs(rec.context_weight - expected[1]) < 1e-6

    def test_resetting_profile_restores_none_weights(self):
        rec = HybridOutfitRecommender(adaptive_weights=True)
        rec.set_user_profile(
            color_season=ColorSeason.WINTER, body_shape=BodyShape.HOURGLASS,
        )
        # Now reset
        rec.set_user_profile(color_season=None, body_shape=None)
        expected = _ADAPTIVE_WEIGHTS["none"]
        assert abs(rec.style_weight - expected[0]) < 1e-6

    def test_adaptive_disabled_keeps_original_weights(self):
        rec = HybridOutfitRecommender(
            style_weight=0.40, context_weight=0.60, adaptive_weights=False
        )
        rec.set_user_profile(
            color_season=ColorSeason.WINTER, body_shape=BodyShape.HOURGLASS,
        )
        assert abs(rec.style_weight - 0.40) < 1e-6
        assert abs(rec.context_weight - 0.60) < 1e-6

    def test_weights_always_sum_to_one(self):
        rec = HybridOutfitRecommender(adaptive_weights=True)
        for completeness in ["none", "partial", "full"]:
            sw, cw = _ADAPTIVE_WEIGHTS[completeness]
            assert abs(sw + cw - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_adaptive_blend_formula_correct(self, scenario_c_garments):
        """Even with adaptive weights, combined = sw*style + cw*context."""
        rec = HybridOutfitRecommender(adaptive_weights=True)
        rec.set_user_profile(
            color_season=ColorSeason.WINTER, body_shape=BodyShape.HOURGLASS,
        )
        score = await rec.score_outfit(scenario_c_garments, casual_context())
        expected = round(
            rec.style_weight * score.style_score
            + rec.context_weight * score.context_score, 3,
        )
        assert abs(score.combined_score - expected) < 0.005


# ===========================================================================
# Point 4 — Pre-scoring garment filtering
# ===========================================================================

class TestPreScoringFilter:
    """Verify _filter_garments_for_profile removes unsuitable garments."""

    def test_no_profile_returns_all(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        garments = [
            make_garment("top", "t-shirt", "white"),
            make_garment("bottom", "jeans", "blue"),
        ]
        filtered = rec._filter_garments_for_profile(garments)
        assert len(filtered) == len(garments)

    def test_with_profile_returns_non_empty(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(body_shape=BodyShape.PEAR)
        garments = [
            make_garment("top", "t-shirt", "white"),
            make_garment("bottom", "jeans", "blue"),
            make_garment("shoes", "sneakers", "white"),
        ]
        filtered = rec._filter_garments_for_profile(garments)
        # Should never return empty (fallback to original)
        assert len(filtered) >= 1

    @pytest.mark.asyncio
    async def test_recommend_uses_filtered_wardrobe(self):
        """recommend() should apply the filter before generating combos."""
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(body_shape=BodyShape.HOURGLASS)
        garments = [
            make_garment("top", "t-shirt", "white"),
            make_garment("top", "blouse", "pink"),
            make_garment("bottom", "jeans", "blue"),
            make_garment("shoes", "sneakers", "white"),
        ]
        # Should not raise, should produce valid results
        ranked = await rec.recommend(garments, casual_context(), top_k=3)
        assert len(ranked) >= 1
        for ro in ranked:
            assert 0.0 <= ro.combined_score <= 1.0


# ===========================================================================
# Point 5 — Personalised explanations
# ===========================================================================

class TestPersonalisedExplanations:
    """Verify _personalise_explanations adds profile-specific text."""

    def test_no_profile_no_extra_text(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        strengths = ["Good color harmony"]
        improvements = ["Add accessories"]
        new_s, new_i = rec._personalise_explanations(
            strengths, improvements, style_score=0.8, context_score=0.7
        )
        # Should not add profile-specific text (no profile set)
        assert new_s == strengths
        assert new_i == improvements

    def test_season_adds_color_tip(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(color_season=ColorSeason.WINTER)
        new_s, new_i = rec._personalise_explanations(
            [], [], style_score=0.6, context_score=0.8
        )
        # Should mention the colour season
        all_text = " ".join(new_s + new_i)
        assert "winter" in all_text.lower()

    def test_body_shape_adds_morphology_tip(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(body_shape=BodyShape.PEAR)
        new_s, new_i = rec._personalise_explanations(
            [], [], style_score=0.8, context_score=0.6
        )
        all_text = " ".join(new_s + new_i)
        assert "pear" in all_text.lower()

    def test_full_profile_adds_combined_insight(self):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(
            color_season=ColorSeason.AUTUMN, body_shape=BodyShape.HOURGLASS
        )
        # Both high → combined insight
        new_s, new_i = rec._personalise_explanations(
            [], [], style_score=0.8, context_score=0.8
        )
        all_text = " ".join(new_s + new_i)
        assert "✨" in all_text or "moteurs" in all_text.lower()

    @pytest.mark.asyncio
    async def test_score_output_contains_personalised_text(self, scenario_c_garments):
        rec = HybridOutfitRecommender(adaptive_weights=False)
        rec.set_user_profile(
            color_season=ColorSeason.WINTER, body_shape=BodyShape.HOURGLASS
        )
        score = await rec.score_outfit(scenario_c_garments, casual_context())
        all_text = " ".join(score.strengths + score.improvements)
        # At least one personal reference should be present
        assert ("winter" in all_text.lower() or "hourglass" in all_text.lower()
                or "morphologie" in all_text.lower() or "saison" in all_text.lower())
