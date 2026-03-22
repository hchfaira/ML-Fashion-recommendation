"""Unit tests for MoodBoardScorer (layer2_style/moodboard_scorer.py)."""
import math
import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from src.layer2_style.moodboard_scorer import MoodBoardScorer, _W_EMBEDDING, _W_COLOR, _W_STYLE, _W_FORMALITY
from src.database.models import MoodBoardStyleProfile, SharedOutfit


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_profile(
    embedding_centroid=None,
    dominant_colors=None,
    dominant_styles=None,
    formality_average=None,
) -> MoodBoardStyleProfile:
    p = MoodBoardStyleProfile()
    p.embedding_centroid = embedding_centroid or []
    p.dominant_colors = dominant_colors or []
    p.dominant_styles = dominant_styles or {}
    p.formality_average = formality_average
    return p


def make_shared_outfit(embedding_vector=None) -> SharedOutfit:
    o = SharedOutfit()
    o.id = str(uuid4())
    o.embedding_vector = embedding_vector or []
    o.dominant_colors = []
    return o


@pytest.fixture
def scorer():
    return MoodBoardScorer()


# ── Tests: _cosine_similarity ─────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self, scorer):
        assert scorer._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_opposite_vectors(self, scorer):
        assert scorer._cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self, scorer):
        assert scorer._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_returns_0(self, scorer):
        assert scorer._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ── Tests: _embedding_similarity ─────────────────────────────────────────────

class TestEmbeddingSimilarity:
    def test_returns_neutral_when_empty(self, scorer):
        assert scorer._embedding_similarity([], []) == 0.5

    def test_returns_neutral_when_dim_mismatch(self, scorer):
        assert scorer._embedding_similarity([1.0], [1.0, 0.0]) == 0.5

    def test_identical(self, scorer):
        assert scorer._embedding_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# ── Tests: _color_alignment ───────────────────────────────────────────────────

class TestColorAlignment:
    def test_neutral_when_empty(self, scorer):
        assert scorer._color_alignment([], []) == 0.5

    def test_full_match(self, scorer):
        board = [{"color": "beige"}, {"color": "white"}]
        outfit = ["beige", "white"]
        assert scorer._color_alignment(outfit, board) == 1.0

    def test_partial_match(self, scorer):
        board = [{"color": "beige"}, {"color": "white"}]
        outfit = ["beige", "black"]
        assert scorer._color_alignment(outfit, board) == pytest.approx(0.5)

    def test_no_match(self, scorer):
        board = [{"color": "beige"}]
        outfit = ["black"]
        assert scorer._color_alignment(outfit, board) == 0.0

    def test_case_insensitive(self, scorer):
        board = [{"color": "Beige"}]
        outfit = ["beige"]
        assert scorer._color_alignment(outfit, board) == 1.0


# ── Tests: _style_alignment ───────────────────────────────────────────────────

class TestStyleAlignment:
    def test_neutral_when_empty(self, scorer):
        assert scorer._style_alignment({}, {}) == 0.5

    def test_identical_styles_return_1(self, scorer):
        styles = {"minimalist": 1.0}
        assert scorer._style_alignment(styles, styles) == pytest.approx(1.0)

    def test_no_overlap_returns_0(self, scorer):
        result = scorer._style_alignment({"minimalist": 1.0}, {"casual": 1.0})
        assert result == 0.0

    def test_partial_overlap(self, scorer):
        outfit = {"minimalist": 1.0, "parisian": 0.5}
        board = {"minimalist": 1.0}
        result = scorer._style_alignment(outfit, board)
        assert 0.0 < result < 1.0


# ── Tests: _formality_alignment ───────────────────────────────────────────────

class TestFormalityAlignment:
    def test_neutral_when_none(self, scorer):
        assert scorer._formality_alignment(None, None) == 0.5

    def test_exact_match(self, scorer):
        assert scorer._formality_alignment(0.5, 0.5) == 1.0

    def test_within_tolerance(self, scorer):
        assert scorer._formality_alignment(0.5, 0.6) == 1.0  # diff=0.1 < 0.15

    def test_at_tolerance_boundary(self, scorer):
        assert scorer._formality_alignment(0.5, 0.65) == pytest.approx(1.0, abs=1e-4)

    def test_far_apart_returns_low(self, scorer):
        result = scorer._formality_alignment(0.0, 1.0)
        assert result == pytest.approx(0.0, abs=1e-4)


# ── Tests: score_outfit_vs_board ──────────────────────────────────────────────

class TestScoreOutfitVsBoard:
    def test_returns_float_between_0_and_1(self, scorer):
        profile = make_profile(
            embedding_centroid=[1.0, 0.0],
            dominant_colors=[{"color": "beige"}],
            dominant_styles={"minimalist": 0.8},
            formality_average=0.4,
        )
        outfit = {
            "embedding_vector": [1.0, 0.0],
            "dominant_colors": ["beige"],
            "dominant_styles": {"minimalist": 0.9},
            "formality_score": 0.4,
        }
        score = scorer.score_outfit_vs_board(outfit, profile)
        assert 0.0 <= score <= 1.0

    def test_perfect_match_returns_high_score(self, scorer):
        v = [1.0, 0.0]
        profile = make_profile(
            embedding_centroid=v,
            dominant_colors=[{"color": "beige"}],
            dominant_styles={"minimalist": 1.0},
            formality_average=0.5,
        )
        outfit = {
            "embedding_vector": v,
            "dominant_colors": ["beige"],
            "dominant_styles": {"minimalist": 1.0},
            "formality_score": 0.5,
        }
        score = scorer.score_outfit_vs_board(outfit, profile)
        assert score >= 0.9

    def test_empty_outfit_returns_neutral(self, scorer):
        profile = make_profile()
        score = scorer.score_outfit_vs_board({}, profile)
        assert 0.0 <= score <= 1.0

    def test_weights_sum_to_1(self):
        total = _W_EMBEDDING + _W_COLOR + _W_STYLE + _W_FORMALITY
        assert total == pytest.approx(1.0)


# ── Tests: apply_moodboard_boost ─────────────────────────────────────────────

class TestApplyMoodboardBoost:
    def test_preference_0_returns_base(self, scorer):
        assert scorer.apply_moodboard_boost(0.7, 0.3, user_preference=0.0) == pytest.approx(0.7)

    def test_preference_1_returns_moodboard(self, scorer):
        assert scorer.apply_moodboard_boost(0.7, 0.3, user_preference=1.0) == pytest.approx(0.3)

    def test_default_preference_03(self, scorer):
        result = scorer.apply_moodboard_boost(0.6, 1.0)
        expected = 0.6 * 0.7 + 1.0 * 0.3
        assert result == pytest.approx(expected)

    def test_clamped_to_0_1(self, scorer):
        # Even with out-of-range preference
        result = scorer.apply_moodboard_boost(0.5, 0.5, user_preference=2.0)
        assert 0.0 <= result <= 1.0

    def test_positive_boost(self, scorer):
        # Mood board score is higher than base → final should be > base
        result = scorer.apply_moodboard_boost(0.4, 1.0, user_preference=0.5)
        assert result > 0.4


# ── Tests: find_similar_saved_outfits ────────────────────────────────────────

class TestFindSimilarSavedOutfits:
    def test_empty_returns_empty(self, scorer):
        result = scorer.find_similar_saved_outfits({}, [], top_k=3)
        assert result == []

    def test_returns_at_most_top_k(self, scorer):
        v = [1.0, 0.0]
        saved = [make_shared_outfit(embedding_vector=v) for _ in range(5)]
        result = scorer.find_similar_saved_outfits({"embedding_vector": v}, saved, top_k=2)
        assert len(result) == 2

    def test_sorted_by_similarity_descending(self, scorer):
        candidate = {"embedding_vector": [1.0, 0.0]}
        close = make_shared_outfit(embedding_vector=[1.0, 0.0])
        far = make_shared_outfit(embedding_vector=[0.0, 1.0])
        result = scorer.find_similar_saved_outfits(candidate, [far, close], top_k=2)
        assert result[0][0] == close.id

    def test_color_jaccard_fallback(self, scorer):
        """When no embeddings are available, falls back to color Jaccard."""
        candidate = {"dominant_colors": ["beige", "white"]}
        s1 = make_shared_outfit()
        s1.dominant_colors = ["beige", "white"]
        s2 = make_shared_outfit()
        s2.dominant_colors = ["black"]
        result = scorer.find_similar_saved_outfits(candidate, [s1, s2], top_k=2)
        assert result[0][0] == s1.id

    def test_zero_similarity_for_no_overlap(self, scorer):
        candidate = {"dominant_colors": ["beige"]}
        s1 = make_shared_outfit()
        s1.dominant_colors = ["black"]
        result = scorer.find_similar_saved_outfits(candidate, [s1], top_k=1)
        assert result[0][1] == pytest.approx(0.0)
