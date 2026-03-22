"""
Mood Board Scorer — Layer 2 Style
==================================

Scores how well a candidate outfit matches a user's mood board profile.

Four sub-scores are combined into one final score:
  1. Embedding similarity  (40 %)
  2. Color alignment       (30 %)
  3. Style alignment       (20 %)
  4. Formality alignment   (10 %)

The final score can then be blended with the classic style score via
``apply_moodboard_boost``.

Public API:
    scorer = MoodBoardScorer()
    score  = scorer.score_outfit_vs_board(outfit_dict, profile)
    final  = scorer.apply_moodboard_boost(base_score, moodboard_score, user_preference)
    hits   = scorer.find_similar_saved_outfits(outfit_dict, saved_outfits, top_k=3)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.core import get_logger
from src.database.models import MoodBoardStyleProfile, SharedOutfit

logger = get_logger(__name__)

# Weights must sum to 1.0
_W_EMBEDDING = 0.40
_W_COLOR = 0.30
_W_STYLE = 0.20
_W_FORMALITY = 0.10

# Tolerance for formality match (±0.15 is a perfect match)
_FORMALITY_TOLERANCE = 0.15


class MoodBoardScorer:
    """Computes how well a candidate outfit aligns with a mood board profile."""

    # ------------------------------------------------------------------
    # Main scoring entry-point
    # ------------------------------------------------------------------

    def score_outfit_vs_board(
        self,
        outfit: Dict[str, Any],
        profile: MoodBoardStyleProfile,
    ) -> float:
        """
        Return a score in [0, 1] reflecting how close the outfit is to the
        mood board profile.

        Args:
            outfit: dict with optional keys:
                - ``embedding_vector``: list[float]
                - ``dominant_colors``:  list[str]
                - ``dominant_styles``:  dict[str, float]
                - ``formality_score``:  float
            profile: MoodBoardStyleProfile ORM row (already computed, not stale)
        """
        s_emb = self._embedding_similarity(
            outfit.get("embedding_vector", []),
            profile.embedding_centroid or [],
        )
        s_color = self._color_alignment(
            outfit.get("dominant_colors", []),
            profile.dominant_colors or [],
        )
        s_style = self._style_alignment(
            outfit.get("dominant_styles", {}),
            profile.dominant_styles or {},
        )
        s_form = self._formality_alignment(
            outfit.get("formality_score"),
            profile.formality_average,
        )

        final = (
            _W_EMBEDDING * s_emb
            + _W_COLOR * s_color
            + _W_STYLE * s_style
            + _W_FORMALITY * s_form
        )
        logger.debug(
            f"MoodBoardScorer: emb={s_emb:.3f} color={s_color:.3f} "
            f"style={s_style:.3f} form={s_form:.3f} → {final:.3f}"
        )
        return round(min(1.0, max(0.0, final)), 4)

    # ------------------------------------------------------------------
    # Boost & discovery helpers
    # ------------------------------------------------------------------

    def apply_moodboard_boost(
        self,
        base_score: float,
        moodboard_score: float,
        user_preference: float = 0.3,
    ) -> float:
        """
        Blend the classic style score with the mood board score.

        Args:
            base_score:        Classic style score (0-1).
            moodboard_score:   Score from ``score_outfit_vs_board`` (0-1).
            user_preference:   Weight given to the mood board (0 = ignore, 1 = override).

        Returns:
            Blended score in [0, 1].
        """
        user_preference = max(0.0, min(1.0, user_preference))
        final = base_score * (1.0 - user_preference) + moodboard_score * user_preference
        return round(min(1.0, max(0.0, final)), 4)

    def find_similar_saved_outfits(
        self,
        outfit: Dict[str, Any],
        saved_outfits: List[SharedOutfit],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Return the top-k saved outfits most similar to the candidate outfit,
        as a list of (shared_outfit_id, similarity_score) tuples.

        Similarity is cosine similarity on the embedding vectors when available;
        falls back to color overlap otherwise.
        """
        candidate_emb: List[float] = outfit.get("embedding_vector", [])
        scores: List[Tuple[str, float]] = []

        for saved in saved_outfits:
            saved_emb: List[float] = saved.embedding_vector or []
            if candidate_emb and saved_emb:
                sim = self._cosine_similarity(candidate_emb, saved_emb)
            else:
                # Fallback: Jaccard on color lists
                c_colors = set((outfit.get("dominant_colors") or []))
                s_colors = set(saved.dominant_colors or [])
                union = c_colors | s_colors
                sim = len(c_colors & s_colors) / len(union) if union else 0.0
            scores.append((saved.id, round(sim, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _embedding_similarity(
        self, candidate: List[float], centroid: List[float]
    ) -> float:
        """Cosine similarity between candidate embedding and board centroid."""
        if not candidate or not centroid:
            return 0.5  # neutral when data is missing
        if len(candidate) != len(centroid):
            return 0.5
        return self._cosine_similarity(candidate, centroid)

    def _color_alignment(
        self,
        outfit_colors: List[str],
        board_colors: List[Dict],
    ) -> float:
        """
        Fraction of outfit colors that appear in the board palette.

        board_colors is a list of {"color": str, "frequency": float}.
        """
        if not outfit_colors or not board_colors:
            return 0.5

        palette = {entry["color"].lower() for entry in board_colors if "color" in entry}
        matches = sum(1 for c in outfit_colors if c.lower() in palette)
        return round(matches / len(outfit_colors), 4)

    def _style_alignment(
        self,
        outfit_styles: Dict[str, float],
        board_styles: Dict[str, float],
    ) -> float:
        """
        Dot-product style similarity normalised to [0, 1].
        """
        if not outfit_styles or not board_styles:
            return 0.5

        all_styles = set(outfit_styles) | set(board_styles)
        dot = sum(
            outfit_styles.get(s, 0.0) * board_styles.get(s, 0.0)
            for s in all_styles
        )
        norm_o = math.sqrt(sum(v * v for v in outfit_styles.values()))
        norm_b = math.sqrt(sum(v * v for v in board_styles.values()))
        if norm_o == 0 or norm_b == 0:
            return 0.5
        similarity = dot / (norm_o * norm_b)
        return round(min(1.0, max(0.0, similarity)), 4)

    def _formality_alignment(
        self,
        outfit_formality: Optional[float],
        board_formality: Optional[float],
    ) -> float:
        """
        1.0 when the difference is within tolerance, decays linearly to 0.
        """
        if outfit_formality is None or board_formality is None:
            return 0.5
        diff = abs(outfit_formality - board_formality)
        if diff <= _FORMALITY_TOLERANCE:
            return 1.0
        # Linear decay: diff=0.15 → 1.0, diff=1.0 → 0.0
        score = 1.0 - (diff - _FORMALITY_TOLERANCE) / (1.0 - _FORMALITY_TOLERANCE)
        return round(max(0.0, score), 4)

    # ------------------------------------------------------------------
    # Math utility
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return round(min(1.0, max(-1.0, dot / (norm_a * norm_b))), 6)
