"""API tests for the Mood Board routes."""
import pytest
import tempfile
import atexit
from typing import Generator
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app
from src.database import Base, get_db
from src.database.models import (
    MoodBoard,
    MoodBoardItem,
    MoodBoardStyleProfile,
    SharedOutfit,
    SharedOutfitLike,
    UserSubscription,
    CustomOutfit,
    OutfitAnalysis,
)
from src.api.middleware.auth import create_token

# ── In-memory test DB ─────────────────────────────────────────────────────────

test_db_dir = tempfile.mkdtemp()
test_db_file = f"{test_db_dir}/test_moodboard_api.db"
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


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_tables():
    yield
    with TestingSessionLocal() as db:
        try:
            db.query(MoodBoardItem).delete()
            db.query(MoodBoardStyleProfile).delete()
            db.query(MoodBoard).delete()
            db.query(SharedOutfitLike).delete()
            db.query(SharedOutfit).delete()
            db.query(OutfitAnalysis).delete()
            db.query(CustomOutfit).delete()
            db.query(UserSubscription).delete()
            db.commit()
        except Exception:
            db.rollback()


@pytest.fixture
def user_id() -> str:
    return "api-test-user"


@pytest.fixture
def auth_headers(user_id) -> dict:
    token = create_token(user_id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def other_user_headers() -> dict:
    token = create_token("other-api-user")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def _make_shared_outfit(db, owner="owner-1", public=True) -> SharedOutfit:
    o = SharedOutfit(
        id=str(uuid4()),
        owner_user_id=owner,
        outfit_data={"garment_ids": ["g1", "g2"]},
        title="Test outfit",
        occasion_tags=["casual"],
        style_tags=["minimalist"],
        formality_score=0.3,
        dominant_colors=["beige", "white"],
        dominant_styles={"minimalist": 0.8},
        embedding_vector=[1.0, 0.0],
        is_public=public,
        saves_count=0,
        likes_count=0,
    )
    db.add(o)
    db.commit()
    db.refresh(o)
    return o


# ── Auth guard ────────────────────────────────────────────────────────────────

class TestAuthGuard:
    def test_share_requires_auth(self):
        resp = client.post("/api/v1/shared-outfits", json={"outfit_data": {}})
        assert resp.status_code == 401

    def test_create_board_requires_auth(self):
        resp = client.post("/api/v1/moodboards", json={"name": "Board"})
        assert resp.status_code == 401

    def test_discover_feed_is_public(self):
        resp = client.get("/api/v1/discover")
        assert resp.status_code == 200


# ── Share outfit ──────────────────────────────────────────────────────────────

class TestShareOutfitEndpoint:
    def test_share_creates_outfit(self, auth_headers):
        body = {
            "outfit_data": {"garment_ids": ["g1"]},
            "title": "My look",
            "occasion_tags": ["casual"],
            "style_tags": ["minimalist"],
            "formality_score": 0.3,
            "dominant_colors": ["beige"],
            "dominant_styles": {"minimalist": 0.8},
            "embedding_vector": [1.0, 0.0],
            "is_public": True,
        }
        resp = client.post("/api/v1/shared-outfits", json=body, headers=auth_headers)
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "My look"
        assert data["is_public"] is True

    def test_unpublish_outfit(self, auth_headers, user_id, db):
        shared = _make_shared_outfit(db, owner=user_id)
        resp = client.delete(f"/api/v1/shared-outfits/{shared.id}", headers=auth_headers)
        assert resp.status_code == 204

    def test_unpublish_wrong_owner_returns_403(self, auth_headers, db):
        shared = _make_shared_outfit(db, owner="someone-else")
        resp = client.delete(f"/api/v1/shared-outfits/{shared.id}", headers=auth_headers)
        assert resp.status_code == 403


# ── Likes ─────────────────────────────────────────────────────────────────────

class TestLikeEndpoints:
    def test_like_outfit(self, auth_headers, db):
        shared = _make_shared_outfit(db)
        resp = client.post(f"/api/v1/shared-outfits/{shared.id}/like", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["likes_count"] == 1

    def test_unlike_outfit(self, auth_headers, db):
        shared = _make_shared_outfit(db)
        client.post(f"/api/v1/shared-outfits/{shared.id}/like", headers=auth_headers)
        resp = client.delete(f"/api/v1/shared-outfits/{shared.id}/like", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["likes_count"] == 0


# ── Discovery ─────────────────────────────────────────────────────────────────

class TestDiscoverEndpoint:
    def test_returns_list(self, db):
        _make_shared_outfit(db)
        _make_shared_outfit(db)
        resp = client.get("/api/v1/discover")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_filters_private_outfits(self, db):
        _make_shared_outfit(db, public=True)
        _make_shared_outfit(db, public=False)
        resp = client.get("/api/v1/discover")
        assert len(resp.json()) == 1

    def test_pagination_params(self, db):
        for _ in range(5):
            _make_shared_outfit(db)
        resp = client.get("/api/v1/discover?page=1&limit=2")
        assert len(resp.json()) == 2

    def test_filter_by_style_tag(self, db):
        _make_shared_outfit(db)  # style_tags: ["minimalist"]
        resp = client.get("/api/v1/discover?style_tag=minimalist")
        data = resp.json()
        for outfit in data:
            assert "minimalist" in outfit["style_tags"]


# ── Board CRUD ────────────────────────────────────────────────────────────────

class TestBoardCRUDEndpoints:
    def test_create_board(self, auth_headers):
        resp = client.post(
            "/api/v1/moodboards",
            json={"name": "Summer Vibes", "description": "For summer"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Summer Vibes"
        assert data["is_active"] is False

    def test_list_boards(self, auth_headers):
        client.post("/api/v1/moodboards", json={"name": "Board 1"}, headers=auth_headers)
        client.post("/api/v1/moodboards", json={"name": "Board 2"}, headers=auth_headers)
        resp = client.get("/api/v1/moodboards", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_board(self, auth_headers):
        create_resp = client.post("/api/v1/moodboards", json={"name": "My Board"}, headers=auth_headers)
        board_id = create_resp.json()["id"]
        resp = client.get(f"/api/v1/moodboards/{board_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["name"] == "My Board"

    def test_get_board_wrong_user_returns_403(self, auth_headers, other_user_headers):
        create_resp = client.post("/api/v1/moodboards", json={"name": "Private"}, headers=auth_headers)
        board_id = create_resp.json()["id"]
        resp = client.get(f"/api/v1/moodboards/{board_id}", headers=other_user_headers)
        assert resp.status_code == 403

    def test_update_board(self, auth_headers):
        create_resp = client.post("/api/v1/moodboards", json={"name": "Old"}, headers=auth_headers)
        board_id = create_resp.json()["id"]
        resp = client.put(
            f"/api/v1/moodboards/{board_id}",
            json={"name": "New Name"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "New Name"

    def test_delete_board(self, auth_headers):
        create_resp = client.post("/api/v1/moodboards", json={"name": "To Delete"}, headers=auth_headers)
        board_id = create_resp.json()["id"]
        del_resp = client.delete(f"/api/v1/moodboards/{board_id}", headers=auth_headers)
        assert del_resp.status_code == 204
        get_resp = client.get(f"/api/v1/moodboards/{board_id}", headers=auth_headers)
        assert get_resp.status_code == 404

    def test_activate_board(self, auth_headers):
        b1 = client.post("/api/v1/moodboards", json={"name": "B1"}, headers=auth_headers).json()
        b2 = client.post("/api/v1/moodboards", json={"name": "B2"}, headers=auth_headers).json()
        client.post(f"/api/v1/moodboards/{b1['id']}/activate", headers=auth_headers)
        resp = client.post(f"/api/v1/moodboards/{b2['id']}/activate", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["is_active"] is True

        # B1 should now be inactive
        b1_resp = client.get(f"/api/v1/moodboards/{b1['id']}", headers=auth_headers)
        assert b1_resp.json()["is_active"] is False


# ── Board Items ───────────────────────────────────────────────────────────────

class TestBoardItemsEndpoints:
    def _create_board(self, headers):
        return client.post("/api/v1/moodboards", json={"name": "Test"}, headers=headers).json()

    def test_save_outfit_to_board(self, auth_headers, db):
        board = self._create_board(auth_headers)
        shared = _make_shared_outfit(db)
        resp = client.post(
            f"/api/v1/moodboards/{board['id']}/items",
            json={"shared_outfit_id": shared.id},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["shared_outfit_id"] == shared.id

    def test_list_items(self, auth_headers, db):
        board = self._create_board(auth_headers)
        for _ in range(3):
            shared = _make_shared_outfit(db)
            client.post(
                f"/api/v1/moodboards/{board['id']}/items",
                json={"shared_outfit_id": shared.id},
                headers=auth_headers,
            )
        resp = client.get(f"/api/v1/moodboards/{board['id']}/items", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_remove_item(self, auth_headers, db):
        board = self._create_board(auth_headers)
        shared = _make_shared_outfit(db)
        item = client.post(
            f"/api/v1/moodboards/{board['id']}/items",
            json={"shared_outfit_id": shared.id},
            headers=auth_headers,
        ).json()
        del_resp = client.delete(
            f"/api/v1/moodboards/{board['id']}/items/{item['id']}",
            headers=auth_headers,
        )
        assert del_resp.status_code == 204

    def test_save_private_outfit_returns_404(self, auth_headers, db):
        board = self._create_board(auth_headers)
        private = _make_shared_outfit(db, public=False)
        resp = client.post(
            f"/api/v1/moodboards/{board['id']}/items",
            json={"shared_outfit_id": private.id},
            headers=auth_headers,
        )
        assert resp.status_code == 404


# ── Style Profile & Analysis ──────────────────────────────────────────────────

class TestAnalysisEndpoints:
    def _create_board_with_item(self, auth_headers, db):
        board = client.post(
            "/api/v1/moodboards", json={"name": "Test"}, headers=auth_headers
        ).json()
        shared = _make_shared_outfit(db)
        client.post(
            f"/api/v1/moodboards/{board['id']}/items",
            json={"shared_outfit_id": shared.id},
            headers=auth_headers,
        )
        return board

    def test_style_profile_no_items_returns_404(self, auth_headers):
        board = client.post("/api/v1/moodboards", json={"name": "Empty"}, headers=auth_headers).json()
        resp = client.get(f"/api/v1/moodboards/{board['id']}/style-profile", headers=auth_headers)
        assert resp.status_code == 404

    def test_style_profile_with_items(self, auth_headers, db):
        board = self._create_board_with_item(auth_headers, db)
        resp = client.get(f"/api/v1/moodboards/{board['id']}/style-profile", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "dominant_colors" in data
        assert "dominant_styles" in data
        assert "items_count" in data
        assert data["items_count"] == 1

    def test_gap_analysis_endpoint(self, auth_headers, db):
        board = self._create_board_with_item(auth_headers, db)
        resp = client.get(
            f"/api/v1/moodboards/{board['id']}/gap-analysis"
            "?wardrobe_colors=beige,white&wardrobe_styles=minimalist",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "alignment_score" in data

    def test_board_summary_endpoint(self, auth_headers, db):
        board = self._create_board_with_item(auth_headers, db)
        resp = client.get(f"/api/v1/moodboards/{board['id']}/summary", headers=auth_headers)
        assert resp.status_code == 200
        assert "summary" in resp.json()
