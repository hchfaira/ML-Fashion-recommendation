"""Integration tests for MoodBoardService — uses a real in-memory SQLite DB."""
import pytest
import asyncio
import tempfile
import atexit
from typing import Generator
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
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
from src.api.services.moodboard_service import MoodBoardService


# ── Database setup ────────────────────────────────────────────────────────────

test_db_dir = tempfile.mkdtemp()
test_db_file = f"{test_db_dir}/test_moodboard_integration.db"
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


@pytest.fixture(autouse=True)
def clean_db():
    """Truncate all mood board tables before each test."""
    yield
    with TestingSessionLocal() as db:
        try:
            db.query(MoodBoardItem).delete()
            db.query(MoodBoardStyleProfile).delete()
            db.query(MoodBoard).delete()
            db.query(SharedOutfitLike).delete()
            db.query(SharedOutfit).delete()
            db.commit()
        except Exception:
            db.rollback()


@pytest.fixture
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def svc(db):
    return MoodBoardService(db)


@pytest.fixture
def user_id():
    return "user-integration-test"


@pytest.fixture
def other_user_id():
    return "user-other"


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


# ── Sharing ───────────────────────────────────────────────────────────────────

class TestShareOutfit:
    def test_creates_shared_outfit(self, svc, db, user_id):
        data = {
            "outfit_data": {"garment_ids": ["g1"]},
            "title": "My first share",
            "occasion_tags": ["casual"],
            "style_tags": ["minimalist"],
            "formality_score": 0.3,
            "dominant_colors": ["beige"],
            "dominant_styles": {"minimalist": 0.8},
            "embedding_vector": [1.0, 0.0],
            "is_public": True,
        }
        shared = svc.share_outfit(user_id, data)
        assert shared.id is not None
        assert shared.owner_user_id == user_id
        assert shared.title == "My first share"

    def test_unpublish_hides_outfit(self, svc, db, user_id):
        shared = _make_shared_outfit(db, owner=user_id)
        svc.unpublish_outfit(user_id, shared.id)
        db.refresh(shared)
        assert shared.is_public is False

    def test_unpublish_by_other_user_raises_403(self, svc, db, user_id, other_user_id):
        shared = _make_shared_outfit(db, owner=other_user_id)
        with pytest.raises(HTTPException) as exc_info:
            svc.unpublish_outfit(user_id, shared.id)
        assert exc_info.value.status_code == 403

    def test_unpublish_missing_raises_404(self, svc):
        with pytest.raises(HTTPException) as exc_info:
            svc.unpublish_outfit("user-x", "non-existent-id")
        assert exc_info.value.status_code == 404


# ── Likes ─────────────────────────────────────────────────────────────────────

class TestLikes:
    def test_like_increments_count(self, svc, db, user_id):
        shared = _make_shared_outfit(db)
        count = svc.like_outfit(user_id, shared.id)
        assert count == 1

    def test_like_is_idempotent(self, svc, db, user_id):
        shared = _make_shared_outfit(db)
        svc.like_outfit(user_id, shared.id)
        count = svc.like_outfit(user_id, shared.id)  # second like
        assert count == 1

    def test_unlike_decrements_count(self, svc, db, user_id):
        shared = _make_shared_outfit(db)
        svc.like_outfit(user_id, shared.id)
        count = svc.unlike_outfit(user_id, shared.id)
        assert count == 0

    def test_unlike_is_idempotent(self, svc, db, user_id):
        shared = _make_shared_outfit(db)
        count = svc.unlike_outfit(user_id, shared.id)  # never liked
        assert count == 0

    def test_like_missing_outfit_raises_404(self, svc):
        with pytest.raises(HTTPException) as exc_info:
            svc.like_outfit("user-x", "bad-id")
        assert exc_info.value.status_code == 404


# ── Discovery Feed ────────────────────────────────────────────────────────────

class TestDiscoveryFeed:
    def test_returns_public_outfits(self, svc, db):
        _make_shared_outfit(db, public=True)
        _make_shared_outfit(db, public=True)
        _make_shared_outfit(db, public=False)
        results = svc.get_discovery_feed()
        assert len(results) == 2

    def test_filters_by_occasion_tag(self, svc, db):
        _make_shared_outfit(db)  # has tag "casual"
        # Create one with different tag
        o = _make_shared_outfit(db)
        o.occasion_tags = ["formal"]
        db.commit()
        results = svc.get_discovery_feed(occasion_tag="casual")
        assert all("casual" in (r.occasion_tags or []) for r in results)

    def test_filters_by_formality(self, svc, db):
        o_low = _make_shared_outfit(db)
        o_low.formality_score = 0.1
        o_high = _make_shared_outfit(db)
        o_high.formality_score = 0.9
        db.commit()
        results = svc.get_discovery_feed(min_formality=0.5)
        assert all(r.formality_score >= 0.5 for r in results)

    def test_pagination(self, svc, db):
        for _ in range(5):
            _make_shared_outfit(db)
        page1 = svc.get_discovery_feed(page=1, limit=2)
        page2 = svc.get_discovery_feed(page=2, limit=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert {o.id for o in page1}.isdisjoint({o.id for o in page2})


# ── Board CRUD ────────────────────────────────────────────────────────────────

class TestBoardCRUD:
    def test_create_board(self, svc, user_id):
        board = svc.create_board(user_id, "My Board", "A test board")
        assert board.id is not None
        assert board.name == "My Board"
        assert board.user_id == user_id
        assert board.is_active is False

    def test_list_boards(self, svc, user_id):
        svc.create_board(user_id, "Board 1")
        svc.create_board(user_id, "Board 2")
        boards = svc.list_boards(user_id)
        assert len(boards) == 2

    def test_update_board(self, svc, user_id):
        board = svc.create_board(user_id, "Old Name")
        updated = svc.update_board(user_id, board.id, name="New Name")
        assert updated.name == "New Name"

    def test_delete_board(self, svc, db, user_id):
        board = svc.create_board(user_id, "To Delete")
        svc.delete_board(user_id, board.id)
        assert db.query(MoodBoard).filter(MoodBoard.id == board.id).first() is None

    def test_delete_wrong_user_raises_403(self, svc, user_id, other_user_id):
        board = svc.create_board(user_id, "Private Board")
        with pytest.raises(HTTPException) as exc_info:
            svc.delete_board(other_user_id, board.id)
        assert exc_info.value.status_code == 403

    def test_get_missing_board_raises_404(self, svc, user_id):
        with pytest.raises(HTTPException) as exc_info:
            svc.get_board(user_id, "non-existent")
        assert exc_info.value.status_code == 404

    def test_activate_board_deactivates_others(self, svc, db, user_id):
        b1 = svc.create_board(user_id, "Board 1")
        b2 = svc.create_board(user_id, "Board 2")
        svc.activate_board(user_id, b1.id)
        svc.activate_board(user_id, b2.id)

        db.refresh(b1)
        db.refresh(b2)
        assert b1.is_active is False
        assert b2.is_active is True


# ── Items ─────────────────────────────────────────────────────────────────────

class TestItems:
    def test_save_outfit_to_board(self, svc, db, user_id):
        board = svc.create_board(user_id, "Test Board")
        shared = _make_shared_outfit(db)
        item = svc.save_outfit_to_board(user_id, board.id, shared.id, "Love this!")
        assert item.id is not None
        assert item.board_id == board.id
        assert item.personal_note == "Love this!"

    def test_save_outfit_increments_saves_count(self, svc, db, user_id):
        board = svc.create_board(user_id, "Test Board")
        shared = _make_shared_outfit(db)
        svc.save_outfit_to_board(user_id, board.id, shared.id)
        db.refresh(shared)
        assert shared.saves_count == 1

    def test_save_outfit_is_idempotent(self, svc, db, user_id):
        board = svc.create_board(user_id, "Test Board")
        shared = _make_shared_outfit(db)
        item1 = svc.save_outfit_to_board(user_id, board.id, shared.id)
        item2 = svc.save_outfit_to_board(user_id, board.id, shared.id)
        assert item1.id == item2.id
        db.refresh(shared)
        assert shared.saves_count == 1  # not doubled

    def test_save_private_outfit_raises_404(self, svc, db, user_id):
        board = svc.create_board(user_id, "Test Board")
        private = _make_shared_outfit(db, public=False)
        with pytest.raises(HTTPException) as exc_info:
            svc.save_outfit_to_board(user_id, board.id, private.id)
        assert exc_info.value.status_code == 404

    def test_remove_item(self, svc, db, user_id):
        board = svc.create_board(user_id, "Test Board")
        shared = _make_shared_outfit(db)
        item = svc.save_outfit_to_board(user_id, board.id, shared.id)
        svc.remove_item(user_id, board.id, item.id)
        assert db.query(MoodBoardItem).filter(MoodBoardItem.id == item.id).first() is None

    def test_remove_item_decrements_saves_count(self, svc, db, user_id):
        board = svc.create_board(user_id, "Test Board")
        shared = _make_shared_outfit(db)
        item = svc.save_outfit_to_board(user_id, board.id, shared.id)
        svc.remove_item(user_id, board.id, item.id)
        db.refresh(shared)
        assert shared.saves_count == 0

    def test_list_items(self, svc, db, user_id):
        board = svc.create_board(user_id, "Test Board")
        for _ in range(3):
            shared = _make_shared_outfit(db)
            svc.save_outfit_to_board(user_id, board.id, shared.id)
        items = svc.list_items(board.id)
        assert len(items) == 3


# ── Style Profile ─────────────────────────────────────────────────────────────

class TestStyleProfile:
    def test_no_items_returns_none(self, svc, user_id):
        board = svc.create_board(user_id, "Empty Board")
        profile = svc.get_style_profile(user_id, board.id)
        assert profile is None

    def test_profile_built_after_adding_items(self, svc, db, user_id):
        board = svc.create_board(user_id, "My Board")
        shared = _make_shared_outfit(db)
        svc.save_outfit_to_board(user_id, board.id, shared.id)
        profile = svc.get_style_profile(user_id, board.id)
        assert profile is not None
        assert profile.items_count == 1
        assert profile.is_stale is False

    def test_profile_invalidated_on_item_add(self, svc, db, user_id):
        board = svc.create_board(user_id, "My Board")
        shared1 = _make_shared_outfit(db)
        svc.save_outfit_to_board(user_id, board.id, shared1.id)
        # Get profile (marks as fresh)
        p1 = svc.get_style_profile(user_id, board.id)
        assert p1.is_stale is False

        # Add another item — profile should be marked stale
        shared2 = _make_shared_outfit(db)
        svc.save_outfit_to_board(user_id, board.id, shared2.id)
        db.refresh(p1)
        assert p1.is_stale is True


# ── Gap Analysis ──────────────────────────────────────────────────────────────

class TestGapAnalysis:
    def test_empty_board_returns_zero_alignment(self, svc, user_id):
        board = svc.create_board(user_id, "Empty")
        gap = svc.get_gap_analysis(user_id, board.id)
        assert gap["alignment_score"] == pytest.approx(0.0)

    def test_perfect_coverage(self, svc, db, user_id):
        board = svc.create_board(user_id, "My Board")
        shared = _make_shared_outfit(db)  # colors: ["beige", "white"], styles: {"minimalist": 0.8}
        svc.save_outfit_to_board(user_id, board.id, shared.id)

        gap = svc.get_gap_analysis(
            user_id, board.id,
            wardrobe_colors=["beige", "white"],
            wardrobe_styles=["minimalist"],
        )
        assert gap["alignment_score"] > 50.0
        assert gap["missing_colors"] == []
        assert gap["missing_styles"] == []

    def test_partial_coverage(self, svc, db, user_id):
        board = svc.create_board(user_id, "My Board")
        shared = _make_shared_outfit(db)
        svc.save_outfit_to_board(user_id, board.id, shared.id)

        gap = svc.get_gap_analysis(
            user_id, board.id,
            wardrobe_colors=["beige"],  # has beige, missing white
            wardrobe_styles=[],
        )
        assert gap["alignment_score"] < 100.0
        assert "white" in gap["missing_colors"]


# ── Recommendations ───────────────────────────────────────────────────────────

class TestRecommendations:
    def test_no_profile_returns_same_order(self, svc, user_id):
        board = svc.create_board(user_id, "Empty Board")
        candidates = [
            {"garment_ids": ["g1"], "base_score": 0.8},
            {"garment_ids": ["g2"], "base_score": 0.9},
        ]
        result = svc.get_moodboard_recommendations(user_id, board.id, candidates)
        # Without profile, base_score determines order
        assert result[0]["base_score"] == 0.9

    def test_with_profile_boosts_matching_outfits(self, svc, db, user_id):
        board = svc.create_board(user_id, "Minimalist Board")
        shared = _make_shared_outfit(db)  # minimalist style
        svc.save_outfit_to_board(user_id, board.id, shared.id)

        candidates = [
            {
                "garment_ids": ["g1"],
                "base_score": 0.5,
                "dominant_styles": {"minimalist": 0.9},
                "dominant_colors": ["beige"],
                "formality_score": 0.3,
                "embedding_vector": [1.0, 0.0],  # matches shared outfit
            },
            {
                "garment_ids": ["g2"],
                "base_score": 0.5,
                "dominant_styles": {"bohemian": 0.9},
                "dominant_colors": ["red"],
                "formality_score": 0.9,
                "embedding_vector": [0.0, 1.0],
            },
        ]
        result = svc.get_moodboard_recommendations(user_id, board.id, candidates, user_preference=0.5)
        # The minimalist outfit should score higher after boost
        assert result[0]["garment_ids"] == ["g1"]

    def test_all_results_have_required_fields(self, svc, user_id):
        board = svc.create_board(user_id, "Board")
        candidates = [{"garment_ids": ["g1"], "base_score": 0.7}]
        result = svc.get_moodboard_recommendations(user_id, board.id, candidates)
        assert "final_score" in result[0]
        assert "moodboard_score" in result[0]
        assert "boost_applied" in result[0]


# ── Async: get_board_summary ──────────────────────────────────────────────────

class TestBoardSummaryAsync:
    @pytest.mark.asyncio
    async def test_returns_string_no_items(self, svc, user_id):
        board = svc.create_board(user_id, "Empty Board")
        result = await svc.get_board_summary(user_id, board.id)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_returns_string_with_profile(self, svc, db, user_id):
        board = svc.create_board(user_id, "Styled Board")
        shared = _make_shared_outfit(db)
        svc.save_outfit_to_board(user_id, board.id, shared.id)
        result = await svc.get_board_summary(user_id, board.id)
        assert isinstance(result, str)
