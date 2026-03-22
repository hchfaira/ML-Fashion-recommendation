"""API tests for the Natural Language Outfit Search routes."""
from __future__ import annotations

import atexit
import tempfile
from typing import Any, Dict, Generator, List
from unittest.mock import AsyncMock, patch
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
test_db_file = f"{test_db_dir}/test_prompt_search_api.db"
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
def clean_tables():
    yield
    with TestingSessionLocal() as db:
        try:
            db.query(UserSubscription).delete()
            db.commit()
        except Exception:
            db.rollback()


@pytest.fixture
def user_id() -> str:
    return "ps-api-test-user"


@pytest.fixture
def auth_headers(user_id) -> dict:
    return {"Authorization": f"Bearer {create_token(user_id)}"}


def _outfit(colors=None, styles=None, occasion_tags=None, formality=0.5, oid=None) -> Dict[str, Any]:
    return {
        "id": oid or str(uuid4()),
        "dominant_colors": colors or [],
        "dominant_styles": {s: 0.8 for s in (styles or [])},
        "occasion_tags": occasion_tags or [],
        "formality_score": formality,
        "title": "test outfit",
    }


# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------

class TestAuthGuard:
    def test_search_requires_auth(self):
        resp = client.post("/api/v1/outfits/search/prompt", json={"prompt": "blue wedding", "candidate_outfits": []})
        assert resp.status_code == 401

    def test_parse_requires_auth(self):
        resp = client.post("/api/v1/outfits/search/prompt/parse", json={"prompt": "test", "candidate_outfits": []})
        assert resp.status_code == 401

    def test_conversation_requires_auth(self):
        resp = client.post("/api/v1/outfits/search/conversation", json={"prompt": "test", "candidate_outfits": []})
        assert resp.status_code == 401

    def test_suggestions_is_public(self):
        resp = client.get("/api/v1/outfits/search/prompt/suggestions")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /outfits/search/prompt/suggestions
# ---------------------------------------------------------------------------

class TestSuggestionsEndpoint:
    def test_returns_list(self):
        resp = client.get("/api/v1/outfits/search/prompt/suggestions")
        assert resp.status_code == 200
        data = resp.json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)
        assert len(data["suggestions"]) > 0

    def test_all_strings(self):
        resp = client.get("/api/v1/outfits/search/prompt/suggestions")
        assert all(isinstance(s, str) for s in resp.json()["suggestions"])


# ---------------------------------------------------------------------------
# POST /outfits/search/prompt/parse
# ---------------------------------------------------------------------------

class TestParseEndpoint:
    def test_returns_parsed_structure(self, auth_headers):
        resp = client.post(
            "/api/v1/outfits/search/prompt/parse",
            json={"prompt": "blue dress for a wedding", "candidate_outfits": []},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "parsed" in data
        assert "filters" in data
        assert data["parsed"]["raw_prompt"] == "blue dress for a wedding"

    def test_parsed_has_colors(self, auth_headers):
        resp = client.post(
            "/api/v1/outfits/search/prompt/parse",
            json={"prompt": "navy blue outfit", "candidate_outfits": []},
            headers=auth_headers,
        )
        parsed = resp.json()["parsed"]
        assert "navy" in parsed["colors"] or "blue" in parsed["colors"]

    def test_parsed_has_occasion(self, auth_headers):
        resp = client.post(
            "/api/v1/outfits/search/prompt/parse",
            json={"prompt": "something for a wedding", "candidate_outfits": []},
            headers=auth_headers,
        )
        parsed = resp.json()["parsed"]
        assert parsed["occasion"] == "wedding"

    def test_filters_has_required_keys(self, auth_headers):
        resp = client.post(
            "/api/v1/outfits/search/prompt/parse",
            json={"prompt": "blue dress", "candidate_outfits": []},
            headers=auth_headers,
        )
        filters = resp.json()["filters"]
        for key in ["formality_min", "formality_max", "target_colors", "excluded_types"]:
            assert key in filters


# ---------------------------------------------------------------------------
# POST /outfits/search/prompt
# ---------------------------------------------------------------------------

class TestSearchEndpoint:
    def test_returns_response_structure(self, auth_headers):
        resp = client.post(
            "/api/v1/outfits/search/prompt",
            json={
                "prompt": "blue outfit for work",
                "candidate_outfits": [_outfit(colors=["blue"], formality=0.6)],
                "explain": False,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        for key in ["outfits", "prompt_interpretation", "confidence",
                    "explanation", "total_candidates", "is_off_topic"]:
            assert key in data

    def test_off_topic_returns_422(self, auth_headers):
        resp = client.post(
            "/api/v1/outfits/search/prompt",
            json={"prompt": "pizza recipe please", "candidate_outfits": []},
            headers=auth_headers,
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "OFF_TOPIC"

    def test_ranking_respects_prompt(self, auth_headers):
        candidates = [
            _outfit(colors=["red"], formality=0.2, oid="red"),
            _outfit(colors=["blue"], formality=0.85, occasion_tags=["wedding"], oid="blue"),
        ]
        resp = client.post(
            "/api/v1/outfits/search/prompt",
            json={
                "prompt": "blue dress for a wedding",
                "candidate_outfits": candidates,
                "explain": False,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        outfits = resp.json()["outfits"]
        if len(outfits) >= 2:
            assert outfits[0]["id"] == "blue"

    def test_max_results_respected(self, auth_headers):
        candidates = [_outfit(colors=["blue"], oid=str(i)) for i in range(20)]
        resp = client.post(
            "/api/v1/outfits/search/prompt",
            json={
                "prompt": "blue outfit",
                "candidate_outfits": candidates,
                "max_results": 3,
                "explain": False,
            },
            headers=auth_headers,
        )
        assert len(resp.json()["outfits"]) <= 3

    def test_excluded_type_not_in_results(self, auth_headers):
        candidates = [
            {**_outfit(colors=["blue"], oid="skirt"), "garment_type": "skirt"},
            {**_outfit(colors=["blue"], oid="dress"), "garment_type": "dress"},
        ]
        resp = client.post(
            "/api/v1/outfits/search/prompt",
            json={
                "prompt": "blue no skirts",
                "candidate_outfits": candidates,
                "explain": False,
            },
            headers=auth_headers,
        )
        ids = [o["id"] for o in resp.json()["outfits"]]
        assert "skirt" not in ids

    def test_explanation_field_present(self, auth_headers):
        candidates = [_outfit(colors=["blue"], formality=0.85, occasion_tags=["wedding"])]
        with patch(
            "src.layer4_llm.prompt_search_explainer.PromptSearchExplainer.explain_result",
            new_callable=AsyncMock,
            return_value="Perfect match!",
        ):
            resp = client.post(
                "/api/v1/outfits/search/prompt",
                json={
                    "prompt": "blue wedding",
                    "candidate_outfits": candidates,
                    "explain": True,
                },
                headers=auth_headers,
            )
        assert resp.status_code == 200
        assert isinstance(resp.json()["explanation"], str)

    def test_missing_suggestion_when_empty_wardrobe(self, auth_headers):
        with patch(
            "src.layer4_llm.prompt_search_explainer.PromptSearchExplainer.suggest_missing",
            new_callable=AsyncMock,
            return_value="Buy a blue gown.",
        ):
            resp = client.post(
                "/api/v1/outfits/search/prompt",
                json={
                    "prompt": "blue wedding outfit",
                    "candidate_outfits": [],
                    "explain": True,
                },
                headers=auth_headers,
            )
        data = resp.json()
        assert data["missing_piece_suggestion"] is not None


# ---------------------------------------------------------------------------
# POST /outfits/search/conversation
# ---------------------------------------------------------------------------

class TestConversationStartEndpoint:
    def test_returns_session_id(self, auth_headers):
        resp = client.post(
            "/api/v1/outfits/search/conversation",
            json={
                "prompt": "blue outfit",
                "candidate_outfits": [_outfit(colors=["blue"])],
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["session_id"] is not None

    def test_off_topic_returns_422(self, auth_headers):
        resp = client.post(
            "/api/v1/outfits/search/conversation",
            json={"prompt": "pizza recipe", "candidate_outfits": []},
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /outfits/search/prompt/refine
# ---------------------------------------------------------------------------

class TestRefineEndpoint:
    def _start_session(self, auth_headers, candidates=None) -> str:
        resp = client.post(
            "/api/v1/outfits/search/conversation",
            json={
                "prompt": "blue outfit",
                "candidate_outfits": candidates or [_outfit(colors=["blue"])],
            },
            headers=auth_headers,
        )
        return resp.json()["session_id"]

    def test_refine_returns_200(self, auth_headers):
        session_id = self._start_session(auth_headers)
        resp = client.post(
            f"/api/v1/outfits/search/prompt/refine?session_id={session_id}",
            json={
                "refinement": "more formal",
                "candidate_outfits": [_outfit(colors=["blue"])],
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200

    def test_refine_returns_same_session_id(self, auth_headers):
        session_id = self._start_session(auth_headers)
        resp = client.post(
            f"/api/v1/outfits/search/prompt/refine?session_id={session_id}",
            json={
                "refinement": "more formal",
                "candidate_outfits": [_outfit(colors=["blue"])],
            },
            headers=auth_headers,
        )
        assert resp.json()["session_id"] == session_id

    def test_missing_session_returns_404(self, auth_headers):
        resp = client.post(
            "/api/v1/outfits/search/prompt/refine?session_id=bad-session-id",
            json={"refinement": "more formal", "candidate_outfits": []},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_refine_updates_formality(self, auth_headers):
        candidates = [
            _outfit(colors=["blue"], formality=0.3, oid="casual"),
            _outfit(colors=["blue"], formality=0.9, oid="formal"),
        ]
        session_id = self._start_session(auth_headers, candidates)
        resp = client.post(
            f"/api/v1/outfits/search/prompt/refine?session_id={session_id}",
            json={"refinement": "more formal", "candidate_outfits": candidates},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        # After "more formal", formal outfit should rank higher
        data = resp.json()
        if data["outfits"]:
            assert data["outfits"][0]["id"] == "formal"
