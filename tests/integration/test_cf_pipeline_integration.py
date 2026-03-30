"""Integration tests — Collaborative Filtering pipeline.

These tests verify the full CF subsystem works end-to-end:
InteractionBuilder → CollaborativeFilter → CFHybridRecommender.

The ``implicit`` library is mocked via ``sys.modules`` injection since
it is not installed in CI.  ``scipy`` is available.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Inject a fake ``implicit`` package BEFORE importing CF modules
# ---------------------------------------------------------------------------
_mock_als_class = MagicMock(name="AlternatingLeastSquares")

if "implicit" not in sys.modules:
    _mock_implicit = MagicMock()
    _mock_implicit_als = MagicMock()
    _mock_implicit_als.AlternatingLeastSquares = _mock_als_class
    sys.modules["implicit"] = _mock_implicit
    sys.modules["implicit.als"] = _mock_implicit_als
else:
    # Grab the existing mock
    _mock_als_class = sys.modules["implicit.als"].AlternatingLeastSquares

from src.layer7_cf.cf_engine import CFEngine, reset_cf_engine  # noqa: E402
from src.layer7_cf.interaction_builder import InteractionBuilder  # noqa: E402
from src.layer7_cf.models import CFScore, InteractionMatrix  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_interactions(n_users: int = 15, n_garments: int = 10) -> List[Dict[str, Any]]:
    """Generate synthetic raw interaction data above the _MIN_USERS threshold."""
    data = []
    for u in range(n_users):
        for g in range(n_garments):
            if (u + g) % 3 == 0:
                data.append({
                    "user_id": f"u{u}",
                    "garment_id": f"g{g}",
                    "times_worn": (u + 1) * (g + 1),
                    "is_favorite": 1 if (u + g) % 5 == 0 else 0,
                })
    return data


@pytest.fixture(autouse=True)
def _reset_als_mock():
    """Reset the global ALS mock before each test so .fit() is fresh."""
    _mock_als_class.reset_mock()
    mock_instance = MagicMock(name="als_instance")
    _mock_als_class.return_value = mock_instance
    yield


# ---------------------------------------------------------------------------
# Builder → Filter flow
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBuilderToFilter:
    def test_build_then_train(self):
        """InteractionBuilder output is accepted by CollaborativeFilter.train."""
        builder = InteractionBuilder()
        matrix = builder.build(_raw_interactions())

        assert len(matrix.user_ids) >= 10
        assert len(matrix.garment_ids) >= 1

        from src.layer7_cf.collaborative_filter import CollaborativeFilter

        cf = CollaborativeFilter(factors=8, iterations=2)
        success = cf.train(matrix)
        assert success is True

    def test_too_few_users_not_trained(self):
        """With <10 users the filter refuses to train."""
        from src.layer7_cf.collaborative_filter import CollaborativeFilter

        small_data = _raw_interactions(n_users=5)
        builder = InteractionBuilder()
        matrix = builder.build(small_data)

        cf = CollaborativeFilter()
        success = cf.train(matrix)
        assert success is False


# ---------------------------------------------------------------------------
# CFEngine retrain flow
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCFEngineRetrain:
    def setup_method(self):
        reset_cf_engine()

    def teardown_method(self):
        reset_cf_engine()

    def test_retrain_returns_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CFEngine(model_dir=tmpdir)
            result = engine.retrain(_raw_interactions())

            assert "trained" in result
            assert "matrix" in result
            assert "model" in result
            assert result["trained"] is True

    def test_retrain_with_small_data_not_trained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CFEngine(model_dir=tmpdir)
            result = engine.retrain(_raw_interactions(n_users=3))

            assert result["trained"] is False


# ---------------------------------------------------------------------------
# Hybrid rerank integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestHybridRerankIntegration:
    def test_rerank_with_trained_model(self):
        """The hybrid recommender uses a trained CF to re-score candidates."""
        from src.layer7_cf.collaborative_filter import CollaborativeFilter
        from src.layer7_cf.hybrid_recommender import CFHybridRecommender
        import numpy as np

        cf = CollaborativeFilter(factors=4)
        cf.is_trained = True
        cf._user_to_idx = {"u0": 0}
        cf._garment_to_idx = {"o1": 0, "o2": 1, "o3": 2}
        cf._idx_to_garment = {0: "o1", 1: "o2", 2: "o3"}

        mock_model = MagicMock()
        # User vector
        mock_model.user_factors = np.array([[1.0, 0.0, 0.0, 0.0]])
        # Item vectors — o3 most aligned with user
        mock_model.item_factors = np.array([
            [0.5, 0.5, 0.5, 0.0],  # o1
            [0.0, 1.0, 0.0, 0.0],  # o2
            [0.9, 0.1, 0.0, 0.0],  # o3 — close to user
        ])
        cf._model = mock_model

        candidates = [
            {"id": "o1", "overall_score": 0.9},
            {"id": "o2", "overall_score": 0.85},
            {"id": "o3", "overall_score": 0.7},
        ]

        recommender = CFHybridRecommender()
        result = recommender.rerank(candidates, "u0", cf)

        # All candidates should have the new keys
        for c in result:
            assert "combined_score" in c
            assert "personalization_active" in c
            assert c["personalization_active"] is True

    def test_rerank_with_untrained_model_preserves_order(self):
        from src.layer7_cf.collaborative_filter import CollaborativeFilter
        from src.layer7_cf.hybrid_recommender import CFHybridRecommender

        cf = CollaborativeFilter()  # not trained

        candidates = [
            {"id": "o1", "overall_score": 0.9},
            {"id": "o2", "overall_score": 0.5},
        ]

        recommender = CFHybridRecommender()
        result = recommender.rerank(candidates, "u0", cf)

        assert [c["id"] for c in result] == ["o1", "o2"]
        assert result[0]["personalization_active"] is False


# ---------------------------------------------------------------------------
# Cold start via engine
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestColdStart:
    def test_engine_status_before_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CFEngine(model_dir=tmpdir)
            s = engine.status()
            assert s["trained"] is False

    def test_recommend_before_training_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CFEngine(model_dir=tmpdir)
            result = engine.cf.recommend("unknown_user")
            assert result == []
