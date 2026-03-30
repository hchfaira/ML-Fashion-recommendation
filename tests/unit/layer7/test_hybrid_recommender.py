"""Unit tests for CFHybridRecommender — Layer 7 CF.

The hybrid recommender blends style scores with CF scores
using the formula:

    combined = style_weight × style_score + cf_weight × cf_score

Default weights: style=0.70, cf=0.30.
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.layer7_cf.hybrid_recommender import (
    CFHybridRecommender,
    _DEFAULT_CF_WEIGHT,
    _DEFAULT_STYLE_WEIGHT,
)
from src.layer7_cf.models import CFScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidates(n: int = 3) -> List[Dict[str, Any]]:
    """Return a list of candidate dicts like the pipeline produces."""
    return [
        {
            "id": f"outfit_{i}",
            "overall_score": round(0.9 - i * 0.1, 2),
            "garments": [],
        }
        for i in range(n)
    ]


def _mock_cf(trained: bool = True, boost_scores: dict | None = None):
    """Create a mock CollaborativeFilter."""
    cf = MagicMock()
    cf.is_trained = trained
    if boost_scores:
        def _boost(user_id, outfit_id):
            return boost_scores.get(outfit_id, CFScore(garment_id=outfit_id, score=0.5, confidence=0.0))
        cf.get_boost_score.side_effect = _boost
    else:
        cf.get_boost_score.return_value = CFScore(
            garment_id="x", score=0.5, confidence=0.0
        )
    return cf


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDefaults:
    def test_default_style_weight(self):
        assert _DEFAULT_STYLE_WEIGHT == 0.70

    def test_default_cf_weight(self):
        assert _DEFAULT_CF_WEIGHT == 0.30

    def test_weights_sum_to_one(self):
        assert _DEFAULT_STYLE_WEIGHT + _DEFAULT_CF_WEIGHT == pytest.approx(1.0)

    def test_custom_weights(self):
        r = CFHybridRecommender(style_weight=0.5, cf_weight=0.5)
        assert r.style_weight == 0.5
        assert r.cf_weight == 0.5


# ---------------------------------------------------------------------------
# Blending logic
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBlending:
    def test_combined_score_formula(self):
        """combined = 0.70 × style + 0.30 × cf."""
        cands = [{"id": "o1", "overall_score": 0.8}]
        cf = _mock_cf(trained=True, boost_scores={
            "o1": CFScore(garment_id="o1", score=1.0, confidence=0.8),
        })

        r = CFHybridRecommender()
        result = r.rerank(cands, "u0", cf)

        expected = 0.70 * 0.8 + 0.30 * 1.0
        assert result[0]["combined_score"] == pytest.approx(expected, abs=0.001)

    def test_equal_weights(self):
        cands = [{"id": "o1", "overall_score": 0.6}]
        cf = _mock_cf(trained=True, boost_scores={
            "o1": CFScore(garment_id="o1", score=0.4, confidence=0.5),
        })
        r = CFHybridRecommender(style_weight=0.5, cf_weight=0.5)
        result = r.rerank(cands, "u0", cf)
        expected = 0.5 * 0.6 + 0.5 * 0.4
        assert result[0]["combined_score"] == pytest.approx(expected, abs=0.001)

    def test_cf_only_weight(self):
        cands = [{"id": "o1", "overall_score": 0.1}]
        cf = _mock_cf(trained=True, boost_scores={
            "o1": CFScore(garment_id="o1", score=0.9, confidence=0.9),
        })
        r = CFHybridRecommender(style_weight=0.0, cf_weight=1.0)
        result = r.rerank(cands, "u0", cf)
        assert result[0]["combined_score"] == pytest.approx(0.9, abs=0.001)


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestReranking:
    def test_rerank_reorders_by_combined_score(self):
        """When CF is active, the list is sorted by combined_score descending."""
        cands = _make_candidates(3)
        # Give the last candidate the highest CF score so it jumps to #1
        boost_scores = {
            "outfit_0": CFScore(garment_id="outfit_0", score=0.1, confidence=0.8),
            "outfit_1": CFScore(garment_id="outfit_1", score=0.5, confidence=0.8),
            "outfit_2": CFScore(garment_id="outfit_2", score=1.0, confidence=0.8),
        }
        cf = _mock_cf(trained=True, boost_scores=boost_scores)
        r = CFHybridRecommender()

        result = r.rerank(cands, "u0", cf)
        # First must have highest combined_score
        assert result[0]["combined_score"] >= result[1]["combined_score"]
        assert result[1]["combined_score"] >= result[2]["combined_score"]

    def test_original_order_preserved_when_untrained(self):
        cands = _make_candidates(3)
        cf = _mock_cf(trained=False)
        r = CFHybridRecommender()

        result = r.rerank(cands, "u0", cf)
        assert [c["id"] for c in result] == ["outfit_0", "outfit_1", "outfit_2"]


# ---------------------------------------------------------------------------
# personalization_active flag
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPersonalizationFlag:
    def test_active_when_trained(self):
        cands = _make_candidates(1)
        cf = _mock_cf(trained=True)
        r = CFHybridRecommender()

        result = r.rerank(cands, "u0", cf)
        assert result[0]["personalization_active"] is True

    def test_inactive_when_untrained(self):
        cands = _make_candidates(1)
        cf = _mock_cf(trained=False)
        r = CFHybridRecommender()

        result = r.rerank(cands, "u0", cf)
        assert result[0]["personalization_active"] is False


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFallback:
    def test_untrained_cf_score_is_half(self):
        """When CF not trained, cf_score defaults to 0.5."""
        cands = _make_candidates(1)
        cf = _mock_cf(trained=False)
        r = CFHybridRecommender()

        result = r.rerank(cands, "u0", cf)
        assert result[0]["cf_score"] == 0.5

    def test_untrained_combined_equals_style_plus_half_cf(self):
        cands = [{"id": "o1", "overall_score": 0.8}]
        cf = _mock_cf(trained=False)
        r = CFHybridRecommender()

        result = r.rerank(cands, "u0", cf)
        expected = 0.70 * 0.8 + 0.30 * 0.5
        assert result[0]["combined_score"] == pytest.approx(expected, abs=0.001)

    def test_empty_candidates_returns_empty(self):
        cf = _mock_cf(trained=True)
        r = CFHybridRecommender()
        assert r.rerank([], "u0", cf) == []


# ---------------------------------------------------------------------------
# Output keys
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOutputKeys:
    def test_added_keys(self):
        cands = _make_candidates(1)
        cf = _mock_cf(trained=True)
        r = CFHybridRecommender()

        result = r.rerank(cands, "u0", cf)
        for key in ["style_score", "cf_score", "combined_score", "personalization_active"]:
            assert key in result[0], f"Missing key: {key}"

    def test_style_score_matches_overall(self):
        cands = [{"id": "o1", "overall_score": 0.77}]
        cf = _mock_cf(trained=True)
        r = CFHybridRecommender()

        result = r.rerank(cands, "u0", cf)
        assert result[0]["style_score"] == pytest.approx(0.77, abs=0.001)

    def test_outfit_id_fallback(self):
        """If candidate has 'outfit_id' instead of 'id', it still works."""
        cands = [{"outfit_id": "o1", "overall_score": 0.5}]
        cf = _mock_cf(trained=True)
        r = CFHybridRecommender()

        result = r.rerank(cands, "u0", cf)
        assert len(result) == 1
