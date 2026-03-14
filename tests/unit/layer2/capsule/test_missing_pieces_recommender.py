"""Tests — MissingPiecesRecommender (F2)"""
import pytest
from src.core.models import (
    CapsuleAnalysisResult,
    CapsuleGarmentScore,
    CapsuleGarmentRole,
    GarmentCategory,
    MissingPiecesResult,
)
from src.layer2_style.capsule.missing_pieces_recommender import MissingPiecesRecommender


# ============================================================================
# Helpers
# ============================================================================

def _make_analysis(cohesion: float = 55.0, total_garments: int = 12, total_outfits: int = 30) -> CapsuleAnalysisResult:
    return CapsuleAnalysisResult(
        cohesion_score=cohesion,
        color_cohesion_score=0.6,
        versatility_ratio=0.3,
        redundancy_penalty=0.1,
        orphan_penalty=0.1,
        dominant_colors=["black", "white"],
        color_coverage_pct=0.8,
        garment_scores=[],
        key_pieces=[],
        orphan_pieces=[],
        redundant_pairs=[],
        total_garments=total_garments,
        total_outfits=total_outfits,
        capsule_profile="standard",
        recommendation="",
        projected_score_after_cleanup=60.0,
    )


@pytest.fixture
def recommender():
    return MissingPiecesRecommender(top_n=5)


# ============================================================================
# Basic output
# ============================================================================

class TestBasicOutput:
    def test_returns_missing_pieces_result(self, recommender):
        result = recommender.recommend(_make_analysis())
        assert isinstance(result, MissingPiecesResult)

    def test_recommendations_not_empty(self, recommender):
        result = recommender.recommend(_make_analysis())
        assert len(result.recommendations) > 0

    def test_max_top_n_respected(self, recommender):
        result = recommender.recommend(_make_analysis())
        assert len(result.recommendations) <= 5

    def test_custom_top_n(self):
        r = MissingPiecesRecommender(top_n=3)
        result = r.recommend(_make_analysis())
        assert len(result.recommendations) <= 3

    def test_priorities_are_sequential(self, recommender):
        result = recommender.recommend(_make_analysis())
        priorities = [rec.priority for rec in result.recommendations]
        assert priorities == list(range(1, len(priorities) + 1))


# ============================================================================
# Impact estimation
# ============================================================================

class TestImpactEstimation:
    def test_impact_outfits_positive(self, recommender):
        result = recommender.recommend(_make_analysis())
        for rec in result.recommendations:
            assert rec.impact_outfits >= 1

    def test_larger_wardrobe_higher_impact(self, recommender):
        small = recommender.recommend(_make_analysis(total_garments=5, total_outfits=10))
        large = recommender.recommend(_make_analysis(total_garments=30, total_outfits=80))
        avg_small = sum(r.impact_outfits for r in small.recommendations) / len(small.recommendations)
        avg_large = sum(r.impact_outfits for r in large.recommendations) / len(large.recommendations)
        assert avg_large > avg_small

    def test_sorted_by_impact_descending(self, recommender):
        result = recommender.recommend(_make_analysis())
        impacts = [r.impact_outfits for r in result.recommendations]
        assert impacts == sorted(impacts, reverse=True)


# ============================================================================
# Projected cohesion
# ============================================================================

class TestProjectedCohesion:
    def test_projected_gte_current(self, recommender):
        analysis = _make_analysis(cohesion=55.0)
        result = recommender.recommend(analysis)
        assert result.projected_cohesion >= result.current_cohesion

    def test_projected_capped_at_100(self, recommender):
        analysis = _make_analysis(cohesion=98.0)
        result = recommender.recommend(analysis)
        assert result.projected_cohesion <= 100.0


# ============================================================================
# Season-aware colours
# ============================================================================

class TestSeasonColors:
    def test_autumn_user_gets_warm_colors(self, recommender):
        result = recommender.recommend(_make_analysis(), user_season="autumn")
        all_colors = [c for r in result.recommendations for c in r.suggested_colors]
        warm_autumn = {"rust", "olive", "camel", "burgundy", "mustard", "brown"}
        # At least some warm autumn colours should appear
        assert any(c in warm_autumn for c in all_colors)

    def test_no_season_returns_default_colors(self, recommender):
        result = recommender.recommend(_make_analysis(), user_season=None)
        for rec in result.recommendations:
            assert len(rec.suggested_colors) <= 3

    def test_colors_list_not_empty(self, recommender):
        result = recommender.recommend(_make_analysis(), user_season="winter")
        for rec in result.recommendations:
            assert len(rec.suggested_colors) > 0


# ============================================================================
# Morphology notes
# ============================================================================

class TestMorphologyNotes:
    def test_profile_note_populated_for_known_shape(self, recommender):
        result = recommender.recommend(_make_analysis(), body_shape="pear")
        # At least one recommendation should have a profile note
        notes = [r.profile_note for r in result.recommendations if r.profile_note]
        assert len(notes) > 0

    def test_no_crash_for_unknown_shape(self, recommender):
        result = recommender.recommend(_make_analysis(), body_shape="triangular_alien")
        assert result is not None

    def test_no_crash_for_no_shape(self, recommender):
        result = recommender.recommend(_make_analysis(), body_shape=None)
        assert result is not None


# ============================================================================
# LLM narration not set by default
# ============================================================================

class TestNoLLMByDefault:
    def test_llm_narration_none_by_default(self, recommender):
        result = recommender.recommend(_make_analysis())
        for rec in result.recommendations:
            assert rec.llm_narration is None or rec.llm_narration == ""

    def test_summary_is_string(self, recommender):
        result = recommender.recommend(_make_analysis())
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0
