"""API tests for the Collaborative Filtering routes.

These tests use FastAPI TestClient and follow the same patterns as the
existing prompt-search API tests (in-memory SQLite, auth tokens, etc.).
"""
from __future__ import annotations

import atexit
import tempfile
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app
from src.database import Base, get_db
from src.database.models import UserSubscription
from src.api.middleware.auth import create_token
from src.layer7_cf.cf_engine import reset_cf_engine
from src.layer7_cf.models import CFScore


# ---------------------------------------------------------------------------
# In-memory test DB
# ---------------------------------------------------------------------------

test_db_dir = tempfile.mkdtemp()
test_db_file = f"{test_db_dir}/test_cf_api.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{test_db_file}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def cleanup():
    import shutil
    try:
        shutil.rmtree(test_db_dir)
    except Exception:
        pass


atexit.register(cleanup)


def override_get_db() -> Generator:
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the global CF singleton before each test."""
    reset_cf_engine()
    yield
    reset_cf_engine()


@pytest.fixture
def user_id() -> str:
    return "cf-api-test-user"


@pytest.fixture
def auth_headers(user_id) -> dict:
    return {"Authorization": f"Bearer {create_token(user_id)}"}


# ---------------------------------------------------------------------------
# Helper to build a mock CF engine
# ---------------------------------------------------------------------------

def _mock_cf_engine(trained: bool = False):
    """Return a mock CFEngine."""
    engine = MagicMock()
    engine.cf.is_trained = trained
    engine.cf.recommend.return_value = [
        CFScore(garment_id="g1", score=0.9, confidence=0.8),
        CFScore(garment_id="g2", score=0.7, confidence=0.6),
    ] if trained else []
    engine.cf.find_similar_garments.return_value = [
        CFScore(garment_id="g_sim", score=0.85, confidence=0.7),
    ] if trained else []
    engine.cf.get_boost_score.return_value = CFScore(
        garment_id="g1",
        score=0.75 if trained else 0.5,
        confidence=0.7 if trained else 0.0,
    )
    engine.cf.status.return_value = {
        "trained": trained,
        "factors": 64,
        "iterations": 20,
        "regularization": 0.1,
        "n_users": 50 if trained else 0,
        "n_garments": 100 if trained else 0,
    }
    engine.status.return_value = engine.cf.status.return_value
    engine.retrain.return_value = {
        "trained": True,
        "matrix": {"n_users": 15, "n_garments": 10, "total_cells": 150, "non_zero": 30, "sparsity": 0.8},
        "model": engine.cf.status.return_value,
    }
    return engine


# ===========================================================================
# GET /cf/recommendations/{user_id}
# ===========================================================================

@pytest.mark.api
class TestGetRecommendations:
    def test_untrained_returns_200_empty(self, auth_headers):
        with patch("src.api.routes.cf.get_cf_engine", return_value=_mock_cf_engine(trained=False)):
            resp = client.get("/api/v1/cf/recommendations/u0", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["trained"] is False
        assert body["recommendations"] == []

    def test_trained_returns_recommendations(self, auth_headers):
        with patch("src.api.routes.cf.get_cf_engine", return_value=_mock_cf_engine(trained=True)):
            resp = client.get("/api/v1/cf/recommendations/u0?n=5", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["trained"] is True
        assert len(body["recommendations"]) == 2

    def test_recommendation_item_shape(self, auth_headers):
        with patch("src.api.routes.cf.get_cf_engine", return_value=_mock_cf_engine(trained=True)):
            resp = client.get("/api/v1/cf/recommendations/u0", headers=auth_headers)
        item = resp.json()["recommendations"][0]
        assert "garment_id" in item
        assert "score" in item
        assert "confidence" in item


# ===========================================================================
# GET /cf/similar-garments/{garment_id}
# ===========================================================================

@pytest.mark.api
class TestSimilarGarments:
    def test_untrained_returns_200_empty(self, auth_headers):
        with patch("src.api.routes.cf.get_cf_engine", return_value=_mock_cf_engine(trained=False)):
            resp = client.get("/api/v1/cf/similar-garments/g1", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["trained"] is False
        assert body["similar"] == []

    def test_trained_returns_similar(self, auth_headers):
        with patch("src.api.routes.cf.get_cf_engine", return_value=_mock_cf_engine(trained=True)):
            resp = client.get("/api/v1/cf/similar-garments/g1?n=3", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["similar"]) == 1


# ===========================================================================
# GET /cf/boost-score/{user_id}/{garment_id}
# ===========================================================================

@pytest.mark.api
class TestBoostScore:
    def test_untrained_returns_neutral_score(self, auth_headers):
        with patch("src.api.routes.cf.get_cf_engine", return_value=_mock_cf_engine(trained=False)):
            resp = client.get("/api/v1/cf/boost-score/u0/g1", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["score"] == 0.5
        assert body["confidence"] == 0.0

    def test_trained_returns_affinity(self, auth_headers):
        with patch("src.api.routes.cf.get_cf_engine", return_value=_mock_cf_engine(trained=True)):
            resp = client.get("/api/v1/cf/boost-score/u0/g1", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["score"] == 0.75
        assert body["trained"] is True


# ===========================================================================
# POST /cf/retrain
# ===========================================================================

@pytest.mark.api
class TestRetrain:
    def test_retrain_returns_status(self, auth_headers):
        with patch("src.api.routes.cf.get_cf_engine", return_value=_mock_cf_engine(trained=True)):
            payload = {
                "interactions": [
                    {"user_id": "u0", "garment_id": "g0", "times_worn": 5},
                    {"user_id": "u1", "garment_id": "g1", "is_favorite": 1},
                ]
            }
            resp = client.post("/api/v1/cf/retrain", json=payload, headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "trained" in body
        assert "matrix" in body
        assert "model" in body

    def test_retrain_empty_interactions_accepted(self, auth_headers):
        """Empty interactions list is accepted (retrain fails gracefully)."""
        mock_engine = _mock_cf_engine(trained=False)
        mock_engine.retrain.return_value = {
            "trained": False,
            "matrix": {"n_users": 0, "n_garments": 0, "total_cells": 0, "non_zero": 0, "sparsity": 1.0},
            "model": {"trained": False, "factors": 64, "iterations": 20, "regularization": 0.1, "n_users": 0, "n_garments": 0},
        }
        with patch("src.api.routes.cf.get_cf_engine", return_value=mock_engine):
            resp = client.post(
                "/api/v1/cf/retrain",
                json={"interactions": []},
                headers=auth_headers,
            )
        assert resp.status_code == 200
        assert resp.json()["trained"] is False


# ===========================================================================
# GET /cf/status
# ===========================================================================

@pytest.mark.api
class TestStatus:
    def test_status_untrained(self, auth_headers):
        with patch("src.api.routes.cf.get_cf_engine", return_value=_mock_cf_engine(trained=False)):
            resp = client.get("/api/v1/cf/status", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["trained"] is False
        assert body["n_users"] == 0

    def test_status_trained(self, auth_headers):
        with patch("src.api.routes.cf.get_cf_engine", return_value=_mock_cf_engine(trained=True)):
            resp = client.get("/api/v1/cf/status", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["trained"] is True
        assert body["n_users"] == 50
        assert body["n_garments"] == 100

    def test_status_response_shape(self, auth_headers):
        with patch("src.api.routes.cf.get_cf_engine", return_value=_mock_cf_engine()):
            resp = client.get("/api/v1/cf/status", headers=auth_headers)
        body = resp.json()
        for key in ["trained", "factors", "iterations", "regularization", "n_users", "n_garments"]:
            assert key in body, f"Missing key: {key}"
