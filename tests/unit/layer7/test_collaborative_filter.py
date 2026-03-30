"""Unit tests for CollaborativeFilter — Layer 7 CF.

All ``implicit`` calls are mocked via ``sys.modules`` injection so these
tests run without the library installed.  ``scipy`` **is** available in
the test environment.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.layer7_cf.models import CFScore, InteractionMatrix


# ---------------------------------------------------------------------------
# Install a fake ``implicit`` package into sys.modules so that
# ``from implicit.als import AlternatingLeastSquares`` resolves.
# ---------------------------------------------------------------------------

_mock_als_class = MagicMock(name="AlternatingLeastSquares")

_mock_implicit = MagicMock()
_mock_implicit_als = MagicMock()
_mock_implicit_als.AlternatingLeastSquares = _mock_als_class

# Inject BEFORE importing CollaborativeFilter
if "implicit" not in sys.modules:
    sys.modules["implicit"] = _mock_implicit
    sys.modules["implicit.als"] = _mock_implicit_als

from src.layer7_cf.collaborative_filter import (  # noqa: E402
    CollaborativeFilter,
    _MIN_USERS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_matrix(n_users: int = 15, n_garments: int = 10) -> InteractionMatrix:
    """Create a synthetic InteractionMatrix with deterministic values."""
    user_ids = [f"u{i}" for i in range(n_users)]
    garment_ids = [f"g{j}" for j in range(n_garments)]
    data: List[List[float]] = []
    for i in range(n_users):
        row = [0.0] * n_garments
        for j in range(n_garments):
            if (i + j) % 3 == 0:
                row[j] = float((i + 1) * (j + 1))
        data.append(row)
    return InteractionMatrix(
        user_ids=user_ids,
        garment_ids=garment_ids,
        data=data,
        user_to_idx={u: i for i, u in enumerate(user_ids)},
        garment_to_idx={g: j for j, g in enumerate(garment_ids)},
    )


@pytest.fixture(autouse=True)
def _reset_als_mock():
    """Reset the global ALS mock before each test."""
    _mock_als_class.reset_mock()
    mock_instance = MagicMock(name="als_instance")
    _mock_als_class.return_value = mock_instance
    yield


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTraining:
    def test_too_few_users_returns_false(self):
        cf = CollaborativeFilter()
        small = _make_matrix(n_users=5)  # below _MIN_USERS
        result = cf.train(small)
        assert result is False
        assert cf.is_trained is False

    def test_min_users_threshold(self):
        assert _MIN_USERS == 10

    def test_train_success_sets_is_trained(self):
        cf = CollaborativeFilter(factors=8, iterations=2)
        matrix = _make_matrix(n_users=15, n_garments=10)

        result = cf.train(matrix)
        assert result is True
        assert cf.is_trained is True
        assert len(cf._user_to_idx) == 15
        assert len(cf._garment_to_idx) == 10

    def test_train_builds_correct_index_maps(self):
        cf = CollaborativeFilter(factors=4, iterations=1)
        matrix = _make_matrix(n_users=12, n_garments=5)

        cf.train(matrix)

        assert cf._idx_to_user[0] == "u0"
        assert cf._idx_to_garment[0] == "g0"
        for uid, idx in cf._user_to_idx.items():
            assert cf._idx_to_user[idx] == uid


# ---------------------------------------------------------------------------
# Sparse matrix format
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSparseMatrixFormat:
    def test_item_users_is_transpose(self):
        """The matrix passed to model.fit() must be items × users (transposed)."""
        cf = CollaborativeFilter(factors=4, iterations=1)
        matrix = _make_matrix(n_users=12, n_garments=5)
        cf.train(matrix)

        # ALS constructor was called once
        _mock_als_class.assert_called_once()
        # model.fit was called with the item_users CSR (the transpose)
        mock_model = _mock_als_class.return_value
        mock_model.fit.assert_called_once()
        arg = mock_model.fit.call_args[0][0]
        # Should be an actual scipy CSR matrix (items × users)
        from scipy.sparse import issparse
        assert issparse(arg)
        # Shape: (n_garments, n_users) — the transpose
        assert arg.shape == (5, 12)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRecommend:
    def test_not_trained_returns_empty(self):
        cf = CollaborativeFilter()
        assert cf.recommend("u0") == []

    def test_unknown_user_triggers_cold_start(self):
        cf = CollaborativeFilter()
        cf.is_trained = True
        cf._user_to_idx = {"u0": 0}
        cf._idx_to_garment = {0: "g0", 1: "g1"}
        cf._popularity = np.array([3.0, 1.0])

        result = cf._cold_start_recommend(2)
        assert len(result) == 2
        assert result[0].confidence == 0.3
        # Most popular first
        assert result[0].garment_id == "g0"


# ---------------------------------------------------------------------------
# Boost score
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBoostScore:
    def test_not_trained_returns_neutral(self):
        cf = CollaborativeFilter()
        result = cf.get_boost_score("u0", "g0")
        assert result.score == 0.5
        assert result.confidence == 0.0

    def test_unknown_user_returns_neutral(self):
        cf = CollaborativeFilter()
        cf.is_trained = True
        cf._user_to_idx = {}
        cf._garment_to_idx = {"g0": 0}
        result = cf.get_boost_score("unknown", "g0")
        assert result.score == 0.5
        assert result.confidence == 0.0

    def test_unknown_garment_returns_neutral(self):
        cf = CollaborativeFilter()
        cf.is_trained = True
        cf._user_to_idx = {"u0": 0}
        cf._garment_to_idx = {}
        result = cf.get_boost_score("u0", "unknown")
        assert result.score == 0.5

    def test_cosine_similarity_calculated(self):
        cf = CollaborativeFilter()
        cf.is_trained = True
        cf._user_to_idx = {"u0": 0}
        cf._garment_to_idx = {"g0": 0}

        mock_model = MagicMock()
        # Identical vectors → cosine = 1.0 → score = (1+1)/2 = 1.0
        mock_model.user_factors = [np.array([1.0, 0.0, 0.0])]
        mock_model.item_factors = [np.array([1.0, 0.0, 0.0])]
        cf._model = mock_model

        result = cf.get_boost_score("u0", "g0")
        assert result.score == pytest.approx(1.0)
        assert result.confidence == 0.7

    def test_orthogonal_vectors_score_half(self):
        cf = CollaborativeFilter()
        cf.is_trained = True
        cf._user_to_idx = {"u0": 0}
        cf._garment_to_idx = {"g0": 0}

        mock_model = MagicMock()
        mock_model.user_factors = [np.array([1.0, 0.0])]
        mock_model.item_factors = [np.array([0.0, 1.0])]
        cf._model = mock_model

        result = cf.get_boost_score("u0", "g0")
        assert result.score == pytest.approx(0.5)

    def test_opposite_vectors_score_zero(self):
        cf = CollaborativeFilter()
        cf.is_trained = True
        cf._user_to_idx = {"u0": 0}
        cf._garment_to_idx = {"g0": 0}

        mock_model = MagicMock()
        mock_model.user_factors = [np.array([1.0, 0.0])]
        mock_model.item_factors = [np.array([-1.0, 0.0])]
        cf._model = mock_model

        result = cf.get_boost_score("u0", "g0")
        assert result.score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestColdStart:
    def test_cold_start_returns_popularity_order(self):
        cf = CollaborativeFilter()
        cf.is_trained = True
        cf._popularity = np.array([1.0, 5.0, 3.0])
        cf._idx_to_garment = {0: "g0", 1: "g1", 2: "g2"}

        results = cf._cold_start_recommend(3)
        ids = [r.garment_id for r in results]
        assert ids[0] == "g1"  # most popular
        assert ids[1] == "g2"

    def test_cold_start_confidence_is_low(self):
        cf = CollaborativeFilter()
        cf.is_trained = True
        cf._popularity = np.array([1.0, 2.0])
        cf._idx_to_garment = {0: "g0", 1: "g1"}

        results = cf._cold_start_recommend(2)
        assert all(r.confidence == 0.3 for r in results)

    def test_cold_start_no_popularity_returns_empty(self):
        cf = CollaborativeFilter()
        cf.is_trained = True
        cf._popularity = None
        assert cf._cold_start_recommend(5) == []


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSaveLoad:
    def test_save_creates_files(self, tmp_path):
        from scipy.sparse import csr_matrix

        cf = CollaborativeFilter()
        cf.is_trained = True
        cf._model = "fake_model_for_pickle"  # must be picklable
        cf._user_to_idx = {"u0": 0}
        cf._garment_to_idx = {"g0": 0}
        cf._idx_to_user = {0: "u0"}
        cf._idx_to_garment = {0: "g0"}
        cf._popularity = np.array([1.0])
        cf._user_items_csr = csr_matrix(np.array([[1.0]]))
        cf._item_users_csr = csr_matrix(np.array([[1.0]]))

        cf.save(tmp_path)

        assert (tmp_path / "cf_model.pkl").exists()
        assert (tmp_path / "user_items.npz").exists()
        assert (tmp_path / "item_users.npz").exists()

    def test_load_missing_file_returns_false(self, tmp_path):
        cf = CollaborativeFilter()
        result = cf.load(tmp_path / "nonexistent")
        assert result is False
        assert cf.is_trained is False

    def test_save_load_roundtrip(self, tmp_path):
        cf = CollaborativeFilter(factors=16, iterations=5)
        cf.is_trained = True
        cf._model = "fake_model"
        cf._user_to_idx = {"alice": 0, "bob": 1}
        cf._garment_to_idx = {"hat": 0}
        cf._idx_to_user = {0: "alice", 1: "bob"}
        cf._idx_to_garment = {0: "hat"}
        cf._popularity = np.array([42.0])
        cf._user_items_csr = None  # skip sparse save
        cf._item_users_csr = None

        cf.save(tmp_path)

        cf2 = CollaborativeFilter()
        result = cf2.load(tmp_path)

        assert result is True
        assert cf2.is_trained is True
        assert cf2.factors == 16
        assert cf2._user_to_idx == {"alice": 0, "bob": 1}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStatus:
    def test_status_keys(self):
        cf = CollaborativeFilter()
        s = cf.status()
        for key in ["trained", "factors", "iterations", "regularization", "n_users", "n_garments"]:
            assert key in s

    def test_status_untrained(self):
        cf = CollaborativeFilter()
        s = cf.status()
        assert s["trained"] is False
        assert s["n_users"] == 0

    def test_status_trained(self):
        cf = CollaborativeFilter()
        cf.is_trained = True
        cf._user_to_idx = {"u0": 0, "u1": 1}
        cf._garment_to_idx = {"g0": 0}
        s = cf.status()
        assert s["trained"] is True
        assert s["n_users"] == 2
        assert s["n_garments"] == 1
