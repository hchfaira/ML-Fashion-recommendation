"""
CF Hybrid Recommender — blend style scores with CF scores.
============================================================

Adds a collaborative-filtering signal to outfit candidates that have
already been scored by the style pipeline (Layers 2 + 3).

The blend formula is:

    combined = style_weight × style_score + cf_weight × cf_score

Default weights: ``style_weight=0.70``, ``cf_weight=0.30``.

When the CF model is not trained, the original ranking is returned
untouched (``personalization_active=False``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core import get_logger
from .collaborative_filter import CollaborativeFilter
from .models import HybridScore

logger = get_logger(__name__)

_DEFAULT_STYLE_WEIGHT = 0.70
_DEFAULT_CF_WEIGHT = 0.30


class CFHybridRecommender:
    """Blend style pipeline scores with CF boost scores.

    Parameters
    ----------
    style_weight : float
        Weight for the style-pipeline score (default 0.70).
    cf_weight : float
        Weight for the collaborative-filtering score (default 0.30).
    """

    def __init__(
        self,
        style_weight: float = _DEFAULT_STYLE_WEIGHT,
        cf_weight: float = _DEFAULT_CF_WEIGHT,
    ) -> None:
        self.style_weight = style_weight
        self.cf_weight = cf_weight

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        user_id: str,
        cf: CollaborativeFilter,
    ) -> List[Dict[str, Any]]:
        """Add CF scores to *candidates* and re-rank.

        Each candidate dict is **mutated in-place** with new keys:

        * ``style_score`` — original score (copied from ``overall_score``)
        * ``cf_score`` — CF boost score for this user × outfit
        * ``combined_score`` — blended score
        * ``personalization_active`` — whether the CF signal contributed

        If the CF model is not trained, each candidate gets
        ``cf_score=0.5`` and ``personalization_active=False``, and the
        original order is preserved.

        Parameters
        ----------
        candidates : list[dict]
            Outfit dicts, each **must** have an ``"overall_score"`` key
            and an ``"id"`` key (or ``"outfit_id"``).
        user_id : str
            The authenticated user.
        cf : CollaborativeFilter
            The CF engine instance.

        Returns
        -------
        list[dict]
            The same list of dicts, sorted by ``combined_score`` descending
            if the CF model is active; original order otherwise.
        """
        active = cf.is_trained
        results: List[Dict[str, Any]] = []

        for cand in candidates:
            outfit_id = cand.get("id") or cand.get("outfit_id", "")
            style_score = float(cand.get("overall_score", 0.5))

            if active:
                boost = cf.get_boost_score(user_id, outfit_id)
                cf_score = boost.score
            else:
                cf_score = 0.5

            combined = (
                self.style_weight * style_score
                + self.cf_weight * cf_score
            )

            cand["style_score"] = round(style_score, 4)
            cand["cf_score"] = round(cf_score, 4)
            cand["combined_score"] = round(combined, 4)
            cand["personalization_active"] = active
            results.append(cand)

        if active:
            results.sort(key=lambda c: c["combined_score"], reverse=True)

        return results
