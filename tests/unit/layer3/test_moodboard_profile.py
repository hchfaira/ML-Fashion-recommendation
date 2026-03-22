"""Unit tests for MoodBoardProfileBuilder (layer3_context/moodboard_profile.py)."""
import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone

from src.layer3_context.moodboard_profile import MoodBoardProfileBuilder
from src.database.models import MoodBoard, MoodBoardItem, MoodBoardStyleProfile, SharedOutfit


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_shared_outfit(
    dominant_colors=None,
    dominant_styles=None,
    formality_score=None,
    occasion_tags=None,
    embedding_vector=None,
) -> SharedOutfit:
    o = SharedOutfit()
    o.id = str(uuid4())
    o.dominant_colors = dominant_colors or []
    o.dominant_styles = dominant_styles or {}
    o.formality_score = formality_score
    o.occasion_tags = occasion_tags or []
    o.embedding_vector = embedding_vector or []
    return o


def make_item(board_id: str, shared_outfit_id: str) -> MoodBoardItem:
    item = MoodBoardItem()
    item.id = str(uuid4())
    item.board_id = board_id
    item.shared_outfit_id = shared_outfit_id
    return item


def make_board(board_id=None) -> MoodBoard:
    b = MoodBoard()
    b.id = board_id or str(uuid4())
    b.user_id = "user-1"
    b.name = "Test Board"
    b.is_active = False
    return b


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def builder(mock_db):
    return MoodBoardProfileBuilder(mock_db)


# ── Tests: _aggregate_colors ──────────────────────────────────────────────────

class TestAggregateColors:
    def test_empty_outfits_returns_empty(self, builder):
        result = builder._aggregate_colors([])
        assert result == []

    def test_single_outfit_single_color(self, builder):
        outfit = make_shared_outfit(dominant_colors=["beige"])
        result = builder._aggregate_colors([outfit])
        assert len(result) == 1
        assert result[0]["color"] == "beige"
        assert result[0]["frequency"] == 1.0

    def test_counts_are_normalised(self, builder):
        outfits = [
            make_shared_outfit(dominant_colors=["beige", "white"]),
            make_shared_outfit(dominant_colors=["beige"]),
        ]
        result = builder._aggregate_colors(outfits)
        by_color = {r["color"]: r["frequency"] for r in result}
        assert by_color["beige"] == pytest.approx(2 / 3, rel=1e-3)
        assert by_color["white"] == pytest.approx(1 / 3, rel=1e-3)

    def test_returns_max_8_colors(self, builder):
        colors = [f"color_{i}" for i in range(12)]
        outfits = [make_shared_outfit(dominant_colors=colors)]
        result = builder._aggregate_colors(outfits)
        assert len(result) <= 8

    def test_case_insensitive(self, builder):
        outfits = [
            make_shared_outfit(dominant_colors=["Beige"]),
            make_shared_outfit(dominant_colors=["beige"]),
        ]
        result = builder._aggregate_colors(outfits)
        by_color = {r["color"]: r["frequency"] for r in result}
        assert "beige" in by_color
        assert len(result) == 1


# ── Tests: _aggregate_styles ──────────────────────────────────────────────────

class TestAggregateStyles:
    def test_empty(self, builder):
        assert builder._aggregate_styles([]) == {}

    def test_single_outfit(self, builder):
        outfit = make_shared_outfit(dominant_styles={"minimalist": 0.8, "parisian": 0.4})
        result = builder._aggregate_styles([outfit])
        assert result["minimalist"] == pytest.approx(0.8)
        assert result["parisian"] == pytest.approx(0.4)

    def test_averages_across_outfits(self, builder):
        outfits = [
            make_shared_outfit(dominant_styles={"minimalist": 1.0}),
            make_shared_outfit(dominant_styles={"minimalist": 0.0}),
        ]
        result = builder._aggregate_styles(outfits)
        assert result["minimalist"] == pytest.approx(0.5)

    def test_sorted_descending(self, builder):
        outfits = [make_shared_outfit(dominant_styles={"casual": 0.2, "minimalist": 0.9})]
        result = builder._aggregate_styles(outfits)
        keys = list(result.keys())
        assert keys[0] == "minimalist"


# ── Tests: _average_formality ─────────────────────────────────────────────────

class TestAverageFormality:
    def test_none_when_no_scores(self, builder):
        outfits = [make_shared_outfit(), make_shared_outfit()]
        assert builder._average_formality(outfits) is None

    def test_average(self, builder):
        outfits = [
            make_shared_outfit(formality_score=0.4),
            make_shared_outfit(formality_score=0.6),
        ]
        result = builder._average_formality(outfits)
        assert result == pytest.approx(0.5)

    def test_ignores_none_values(self, builder):
        outfits = [
            make_shared_outfit(formality_score=0.8),
            make_shared_outfit(formality_score=None),
        ]
        result = builder._average_formality(outfits)
        assert result == pytest.approx(0.8)


# ── Tests: _aggregate_occasions ───────────────────────────────────────────────

class TestAggregateOccasions:
    def test_empty(self, builder):
        assert builder._aggregate_occasions([]) == {}

    def test_normalised(self, builder):
        outfits = [
            make_shared_outfit(occasion_tags=["casual", "weekend"]),
            make_shared_outfit(occasion_tags=["casual"]),
        ]
        result = builder._aggregate_occasions(outfits)
        assert result["casual"] == pytest.approx(2 / 3, rel=1e-3)
        assert result["weekend"] == pytest.approx(1 / 3, rel=1e-3)


# ── Tests: _compute_centroid ──────────────────────────────────────────────────

class TestComputeCentroid:
    def test_empty_when_no_embeddings(self, builder):
        outfits = [make_shared_outfit(), make_shared_outfit()]
        assert builder._compute_centroid(outfits) == []

    def test_mean_of_vectors(self, builder):
        outfits = [
            make_shared_outfit(embedding_vector=[1.0, 0.0]),
            make_shared_outfit(embedding_vector=[0.0, 1.0]),
        ]
        result = builder._compute_centroid(outfits)
        assert result == pytest.approx([0.5, 0.5])

    def test_skips_mismatched_dimensions(self, builder):
        outfits = [
            make_shared_outfit(embedding_vector=[1.0, 0.0]),
            make_shared_outfit(embedding_vector=[0.0]),  # wrong dim — skipped
        ]
        result = builder._compute_centroid(outfits)
        # The mismatched vector is skipped in the averaging loop,
        # but the dim is determined from the first valid vector so result
        # is based on the one valid 2-dim vector only.
        assert len(result) == 2
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)


# ── Tests: _compute_coherence ─────────────────────────────────────────────────

class TestComputeCoherence:
    def test_single_outfit_returns_1(self, builder):
        outfits = [make_shared_outfit(embedding_vector=[1.0, 0.0])]
        assert builder._compute_coherence(outfits) == 1.0

    def test_identical_vectors_returns_1(self, builder):
        v = [1.0, 0.0]
        outfits = [make_shared_outfit(embedding_vector=v) for _ in range(3)]
        result = builder._compute_coherence(outfits)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_opposite_vectors_returns_0(self, builder):
        outfits = [
            make_shared_outfit(embedding_vector=[1.0, 0.0]),
            make_shared_outfit(embedding_vector=[-1.0, 0.0]),
        ]
        result = builder._compute_coherence(outfits)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_orthogonal_vectors_returns_0_5(self, builder):
        outfits = [
            make_shared_outfit(embedding_vector=[1.0, 0.0]),
            make_shared_outfit(embedding_vector=[0.0, 1.0]),
        ]
        result = builder._compute_coherence(outfits)
        assert result == pytest.approx(0.5, abs=1e-3)

    def test_no_embeddings_returns_1(self, builder):
        outfits = [make_shared_outfit() for _ in range(3)]
        assert builder._compute_coherence(outfits) == 1.0


# ── Tests: invalidate_profile ─────────────────────────────────────────────────

class TestInvalidateProfile:
    def test_marks_existing_profile_stale(self, builder, mock_db):
        profile = MoodBoardStyleProfile()
        profile.is_stale = False
        mock_db.query.return_value.filter.return_value.first.return_value = profile

        builder.invalidate_profile("board-1")

        assert profile.is_stale is True
        mock_db.commit.assert_called_once()

    def test_noop_when_no_profile(self, builder, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        builder.invalidate_profile("board-1")  # should not raise


# ── Tests: get_or_rebuild_profile ─────────────────────────────────────────────

class TestGetOrRebuildProfile:
    def test_returns_cached_when_not_stale(self, builder, mock_db):
        profile = MoodBoardStyleProfile()
        profile.is_stale = False
        mock_db.query.return_value.filter.return_value.first.return_value = profile

        result = builder.get_or_rebuild_profile("board-1")
        assert result is profile

    def test_rebuilds_when_stale(self, builder, mock_db):
        profile = MoodBoardStyleProfile()
        profile.is_stale = True
        mock_db.query.return_value.filter.return_value.first.return_value = profile

        with patch.object(builder, "build_profile", return_value=None) as mock_build:
            builder.get_or_rebuild_profile("board-1")
            mock_build.assert_called_once_with("board-1")

    def test_rebuilds_when_no_profile(self, builder, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with patch.object(builder, "build_profile", return_value=None) as mock_build:
            builder.get_or_rebuild_profile("board-1")
            mock_build.assert_called_once_with("board-1")


# ── Tests: build_profile ──────────────────────────────────────────────────────

class TestBuildProfile:
    def _setup_db(self, mock_db, board, items, outfits, existing_profile=None):
        """Wire up mock_db for build_profile calls."""
        def query_side_effect(model):
            m = MagicMock()
            if model is MoodBoard:
                m.filter.return_value.first.return_value = board
            elif model is MoodBoardItem:
                m.filter.return_value.all.return_value = items
            elif model is SharedOutfit:
                m.filter.return_value.all.return_value = outfits
            elif model is MoodBoardStyleProfile:
                m.filter.return_value.first.return_value = existing_profile
            return m

        mock_db.query.side_effect = query_side_effect

    def test_returns_none_when_board_not_found(self, builder, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        result = builder.build_profile("missing-board")
        assert result is None

    def test_returns_none_when_no_items(self, builder, mock_db):
        board = make_board("b1")

        def query_side(model):
            m = MagicMock()
            if model is MoodBoard:
                m.filter.return_value.first.return_value = board
            elif model is MoodBoardItem:
                m.filter.return_value.all.return_value = []
            return m

        mock_db.query.side_effect = query_side
        result = builder.build_profile("b1")
        assert result is None

    def test_creates_profile_when_none_exists(self, builder, mock_db):
        board = make_board("b1")
        outfit = make_shared_outfit(
            dominant_colors=["beige"],
            dominant_styles={"minimalist": 0.8},
            formality_score=0.4,
            occasion_tags=["casual"],
            embedding_vector=[1.0, 0.0],
        )
        outfit.id = "outfit-1"
        item = make_item("b1", "outfit-1")

        created_profile = None

        def db_add(obj):
            nonlocal created_profile
            if isinstance(obj, MoodBoardStyleProfile):
                created_profile = obj

        mock_db.add.side_effect = db_add

        def query_side(model):
            m = MagicMock()
            if model is MoodBoard:
                m.filter.return_value.first.return_value = board
            elif model is MoodBoardItem:
                m.filter.return_value.all.return_value = [item]
            elif model is SharedOutfit:
                m.filter.return_value.all.return_value = [outfit]
            elif model is MoodBoardStyleProfile:
                m.filter.return_value.first.return_value = None  # not yet created
            return m

        mock_db.query.side_effect = query_side
        mock_db.refresh.side_effect = lambda obj: None

        result = builder.build_profile("b1")

        assert result is not None or created_profile is not None
        mock_db.commit.assert_called()
