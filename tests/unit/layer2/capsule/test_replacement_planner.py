"""Tests — ReplacementPlanner (F3)"""
import pytest
from src.core.models import (
    CapsuleGarmentScore,
    CapsuleGarmentRole,
    RedundantPair,
    ReplacementPlanResult,
    ReplacementVerdict,
)
from src.layer2_style.capsule.replacement_planner import ReplacementPlanner


# ============================================================================
# Helpers
# ============================================================================

def _score(garment_id: str, versatility: float) -> CapsuleGarmentScore:
    return CapsuleGarmentScore(
        garment_id=garment_id,
        garment_description=f"Test garment {garment_id}",
        versatility_score=versatility,
        outfit_count=int(versatility * 20),
        total_outfits=20,
        capsule_role=CapsuleGarmentRole.ACCEPTABLE,
    )


def _pair(a: str, b: str, similarity: float = 1.0) -> RedundantPair:
    return RedundantPair(
        garment_a_id=a,
        garment_a_description=f"Garment {a}",
        garment_b_id=b,
        garment_b_description=f"Garment {b}",
        similarity_score=similarity,
        reason="Same subcategory and colour",
    )


@pytest.fixture
def planner():
    return ReplacementPlanner()


# ============================================================================
# Empty inputs
# ============================================================================

class TestEmptyInputs:
    def test_no_pairs_returns_empty_result(self, planner):
        result = planner.plan([], [_score("g1", 0.7)])
        assert result.verdicts == []
        assert result.total_outfits_gained == 0

    def test_no_scores_still_works(self, planner):
        result = planner.plan([_pair("a", "b")], [])
        assert isinstance(result, ReplacementPlanResult)


# ============================================================================
# Decision logic
# ============================================================================

class TestDecisionLogic:
    def test_winner_is_higher_versatility(self, planner):
        pairs = [_pair("a", "b")]
        scores = [_score("a", 0.8), _score("b", 0.2)]
        result = planner.plan(pairs, scores)
        assert len(result.verdicts) == 1
        assert result.verdicts[0].garment_keep_id == "a"
        assert result.verdicts[0].garment_remove_id == "b"

    def test_reverse_winner_correct(self, planner):
        pairs = [_pair("a", "b")]
        scores = [_score("a", 0.1), _score("b", 0.9)]
        result = planner.plan(pairs, scores)
        assert result.verdicts[0].garment_keep_id == "b"
        assert result.verdicts[0].garment_remove_id == "a"

    def test_too_close_no_verdict(self, planner):
        # Gain = 0.02 < threshold 0.05
        pairs = [_pair("a", "b")]
        scores = [_score("a", 0.51), _score("b", 0.49)]
        result = planner.plan(pairs, scores)
        assert result.verdicts == []

    def test_multiple_pairs_sorted_by_confidence(self, planner):
        pairs = [_pair("a", "b"), _pair("c", "d")]
        scores = [
            _score("a", 0.9), _score("b", 0.1),   # gain 0.8 → high confidence
            _score("c", 0.6), _score("d", 0.4),   # gain 0.2 → medium confidence
        ]
        result = planner.plan(pairs, scores)
        assert len(result.verdicts) == 2
        assert result.verdicts[0].confidence >= result.verdicts[1].confidence


# ============================================================================
# Confidence and timing
# ============================================================================

class TestConfidenceAndTiming:
    def test_high_gain_yields_high_confidence(self, planner):
        pairs = [_pair("a", "b")]
        scores = [_score("a", 1.0), _score("b", 0.0)]
        result = planner.plan(pairs, scores)
        assert result.verdicts[0].confidence >= 0.85

    def test_medium_gain_yields_medium_confidence(self, planner):
        pairs = [_pair("a", "b")]
        scores = [_score("a", 0.5), _score("b", 0.2)]  # gain = 0.3
        result = planner.plan(pairs, scores)
        v = result.verdicts[0]
        assert 0.45 <= v.confidence < 1.0

    def test_transition_timing_is_string(self, planner):
        pairs = [_pair("a", "b")]
        scores = [_score("a", 0.8), _score("b", 0.1)]
        result = planner.plan(pairs, scores)
        assert isinstance(result.verdicts[0].transition_timing, str)
        assert len(result.verdicts[0].transition_timing) > 5


# ============================================================================
# Outfit gain estimation
# ============================================================================

class TestOutfitGainEstimation:
    def test_total_outfits_gained_nonnegative(self, planner):
        pairs = [_pair("a", "b")]
        scores = [_score("a", 0.9), _score("b", 0.1)]
        result = planner.plan(pairs, scores)
        assert result.total_outfits_gained >= 0

    def test_high_confidence_verdict_gains_more_outfits(self, planner):
        high_pairs = [_pair("a", "b")]
        high_scores = [_score("a", 1.0), _score("b", 0.0)]
        low_pairs = [_pair("c", "d")]
        low_scores = [_score("c", 0.55), _score("d", 0.45)]  # too close
        high_result = planner.plan(high_pairs, high_scores)
        low_result = planner.plan(low_pairs, low_scores)
        assert high_result.total_outfits_gained >= low_result.total_outfits_gained


# ============================================================================
# Summary
# ============================================================================

class TestSummary:
    def test_summary_is_string(self, planner):
        result = planner.plan([], [])
        assert isinstance(result.summary, str)

    def test_summary_mentions_count(self, planner):
        pairs = [_pair("a", "b")]
        scores = [_score("a", 0.9), _score("b", 0.1)]
        result = planner.plan(pairs, scores)
        assert "1" in result.summary or "replacement" in result.summary.lower()
