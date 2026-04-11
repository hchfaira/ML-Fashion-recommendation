"""
API tests — POST /capsule/optimize endpoint
============================================
Uses FastAPI TestClient with an in-memory SQLite database,
following the same patterns as the existing API test modules.
"""
from __future__ import annotations

import atexit
import tempfile
from typing import Any, Dict, Generator, List
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app
from src.database import Base, get_db
from src.database.models import UserSubscription
from src.api.middleware.auth import create_token


# ---------------------------------------------------------------------------
# In-memory test DB
# ---------------------------------------------------------------------------

test_db_dir = tempfile.mkdtemp()
test_db_file = f"{test_db_dir}/test_capsule_optimize_api.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{test_db_file}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def _cleanup():
    import shutil
    try:
        shutil.rmtree(test_db_dir)
    except Exception:
        pass


atexit.register(_cleanup)


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

@pytest.fixture
def user_id() -> str:
    return "capsule-opt-test-user"


@pytest.fixture
def auth_headers(user_id) -> dict:
    return {"Authorization": f"Bearer {create_token(user_id)}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _g(
    garment_id: str | None = None,
    category: str = "top",
    subcategory: str = "t-shirt",
    color_hex: str = "#000000",
    formality: str = "casual",
    seasons: list[str] | None = None,
    material: str = "cotton",
) -> Dict[str, Any]:
    return {
        "id": garment_id or f"g_{uuid4().hex[:8]}",
        "attributes": {
            "category": category,
            "subcategory": subcategory,
            "color_hex": color_hex,
            "formality": formality,
            "seasons": seasons or ["spring", "summer", "autumn", "winter"],
            "material": material,
            "confidence": 0.95,
        },
    }


def _wardrobe_payload(n_pieces: int = 8, **overrides) -> Dict[str, Any]:
    """Build a valid /capsule/optimize request body."""
    garments = [
        _g("t1", "top", "t-shirt", "#FFFFFF", "casual"),
        _g("t2", "top", "shirt", "#1C3A5F", "smart_casual"),
        _g("t3", "top", "blouse", "#000000", "business"),
        _g("b1", "bottom", "jeans", "#00008B", "casual"),
        _g("b2", "bottom", "chinos", "#D2B48C", "smart_casual"),
        _g("b3", "bottom", "trousers", "#808080", "business"),
        _g("d1", "dress", "midi-dress", "#FF69B4", "smart_casual"),
        _g("s1", "shoes", "sneakers", "#FFFFFF", "casual"),
        _g("s2", "shoes", "loafers", "#8B4513", "smart_casual"),
        _g("o1", "outerwear", "blazer", "#000000", "business",
           seasons=["autumn", "winter"]),
        _g("a1", "accessory", "watch", "#C0C0C0", "casual"),
    ]
    request = {
        "user_id": "test-user",
        "n_pieces": n_pieces,
        "occasion_types": ["casual", "smart_casual"],
        "climate": "mixed",
        "anchor_ids": [],
        "excluded_ids": [],
    }
    request.update(overrides)
    return {"request": request, "garments": garments}


URL = "/api/v1/capsule/optimize"


# ===========================================================================
# POST /capsule/optimize — Happy paths
# ===========================================================================

@pytest.mark.api
class TestOptimizeHappyPath:

    def test_200_basic_request(self, auth_headers):
        resp = client.post(URL, json=_wardrobe_payload(6), headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_pieces"] == 6
        assert body["total_score"] > 0
        assert body["source"] == "greedy_2opt"

    def test_response_schema_fields(self, auth_headers):
        resp = client.post(URL, json=_wardrobe_payload(5), headers=auth_headers)
        body = resp.json()
        expected_keys = {
            "n_pieces", "total_wardrobe", "total_score",
            "valid_combinations", "color_palette", "color_names",
            "occasion_coverage", "practicality_score",
            "selected_garments", "missing_pieces", "alternatives",
            "summary", "source",
        }
        assert expected_keys.issubset(set(body.keys()))

    def test_selected_garments_have_details(self, auth_headers):
        resp = client.post(URL, json=_wardrobe_payload(5), headers=auth_headers)
        body = resp.json()
        for piece in body["selected_garments"]:
            assert "garment_id" in piece
            assert "category" in piece
            assert "role" in piece
            assert "outfit_contribution" in piece
            assert "versatility_score" in piece

    def test_with_anchor_ids(self, auth_headers):
        payload = _wardrobe_payload(6, anchor_ids=["t1", "b1"])
        resp = client.post(URL, json=payload, headers=auth_headers)
        body = resp.json()
        selected_ids = {p["garment_id"] for p in body["selected_garments"]}
        assert "t1" in selected_ids
        assert "b1" in selected_ids

    def test_with_excluded_ids(self, auth_headers):
        payload = _wardrobe_payload(5, excluded_ids=["t1", "t2"])
        resp = client.post(URL, json=payload, headers=auth_headers)
        body = resp.json()
        selected_ids = {p["garment_id"] for p in body["selected_garments"]}
        assert "t1" not in selected_ids
        assert "t2" not in selected_ids

    def test_occasion_coverage_keys(self, auth_headers):
        payload = _wardrobe_payload(6, occasion_types=["casual", "business"])
        resp = client.post(URL, json=payload, headers=auth_headers)
        body = resp.json()
        assert "casual" in body["occasion_coverage"]
        assert "business" in body["occasion_coverage"]

    def test_summary_text(self, auth_headers):
        resp = client.post(URL, json=_wardrobe_payload(6), headers=auth_headers)
        body = resp.json()
        assert "6 pieces selected" in body["summary"]

    def test_warm_climate(self, auth_headers):
        payload = _wardrobe_payload(5, climate="warm")
        resp = client.post(URL, json=payload, headers=auth_headers)
        assert resp.status_code == 200

    def test_cold_climate(self, auth_headers):
        payload = _wardrobe_payload(5, climate="cold")
        resp = client.post(URL, json=payload, headers=auth_headers)
        assert resp.status_code == 200

    def test_alternatives_in_response(self, auth_headers):
        resp = client.post(URL, json=_wardrobe_payload(6), headers=auth_headers)
        body = resp.json()
        assert isinstance(body["alternatives"], list)
        for alt in body["alternatives"]:
            assert "total_score" in alt
            assert "valid_combinations" in alt
            assert "label" in alt


# ===========================================================================
# POST /capsule/optimize — Validation / errors
# ===========================================================================

@pytest.mark.api
class TestOptimizeValidation:

    def test_422_empty_garments(self, auth_headers):
        payload = {
            "request": {
                "user_id": "u1",
                "n_pieces": 5,
            },
            "garments": [],
        }
        resp = client.post(URL, json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_422_missing_n_pieces(self, auth_headers):
        payload = {
            "request": {
                "user_id": "u1",
            },
            "garments": [_g("t1", "top")],
        }
        resp = client.post(URL, json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_422_n_pieces_zero(self, auth_headers):
        payload = {
            "request": {
                "user_id": "u1",
                "n_pieces": 0,
            },
            "garments": [_g("t1", "top")],
        }
        resp = client.post(URL, json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_422_n_pieces_negative(self, auth_headers):
        payload = {
            "request": {
                "user_id": "u1",
                "n_pieces": -3,
            },
            "garments": [_g("t1", "top")],
        }
        resp = client.post(URL, json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_422_missing_user_id(self, auth_headers):
        payload = {
            "request": {
                "n_pieces": 5,
            },
            "garments": [_g("t1", "top")],
        }
        resp = client.post(URL, json=payload, headers=auth_headers)
        assert resp.status_code == 422
