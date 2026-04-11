"""
CF Hybrid Recommender — blend style, CF, and context scores.
==============================================================

Adds collaborative-filtering and context signals to outfit candidates
that have already been scored by the style pipeline (Layers 2 + 3).

The blend formula is:

    combined = style_weight × style + cf_weight × cf + context_weight × context

Default weights: ``style_weight=0.40``, ``cf_weight=0.40``,
``context_weight=0.20``.

When the CF model is not trained, ``cf_weight`` is redistributed
proportionally to style and context (``personalization_active=False``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core import get_logger
from .collaborative_filter import CollaborativeFilter
from .models import HybridScore

logger = get_logger(__name__)

_DEFAULT_STYLE_WEIGHT = 0.40
_DEFAULT_CF_WEIGHT = 0.40
_DEFAULT_CONTEXT_WEIGHT = 0.20


class CFHybridRecommender:
    """Blend style pipeline scores with CF boost scores and context scores.

    Parameters
    ----------
    style_weight : float
        Weight for the style-pipeline score (default 0.40).
    cf_weight : float
        Weight for the collaborative-filtering score (default 0.40).
    context_weight : float
        Weight for the context score (weather/occasion, default 0.20).
    """

    def __init__(
        self,
        style_weight: float = _DEFAULT_STYLE_WEIGHT,
        cf_weight: float = _DEFAULT_CF_WEIGHT,
        context_weight: float = _DEFAULT_CONTEXT_WEIGHT,
    ) -> None:
        self.style_weight = style_weight
        self.cf_weight = cf_weight
        self.context_weight = context_weight

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        user_id: str,
        cf: CollaborativeFilter,
        context_scores: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Add CF + context scores to *candidates* and re-rank.

        Each candidate dict is **mutated in-place** with new keys:

        * ``style_score`` — original score (copied from ``overall_score``)
        * ``cf_score`` — CF boost score for this user × outfit
        * ``context_score`` — context fit score (weather + occasion)
        * ``combined_score`` — blended score
        * ``personalization_active`` — whether the CF signal contributed

        Parameters
        ----------
        candidates : list[dict]
            Outfit dicts, each **must** have an ``"overall_score"`` key
            and an ``"id"`` key (or ``"outfit_id"``).
        user_id : str
            The authenticated user.
        cf : CollaborativeFilter
            The CF engine instance.
        context_scores : dict, optional
            Mapping of ``outfit_id → context_score`` in ``[0, 1]``.
            If ``None``, context defaults to ``0.5`` for all candidates.

        Returns
        -------
        list[dict]
            The same list of dicts, sorted by ``combined_score`` descending
            if the CF model is active; original order otherwise.
        """
        active = cf.is_trained
        ctx_map = context_scores or {}
        results: List[Dict[str, Any]] = []

        # When CF is inactive, redistribute its weight
        if active:
            sw, cw, xw = self.style_weight, self.cf_weight, self.context_weight
        else:
            # Redistribute cf_weight proportionally to style and context
            total_other = self.style_weight + self.context_weight
            if total_other > 0:
                sw = self.style_weight + self.cf_weight * (self.style_weight / total_other)
                xw = self.context_weight + self.cf_weight * (self.context_weight / total_other)
            else:
                sw, xw = 0.5, 0.5
            cw = 0.0

        for cand in candidates:
            outfit_id = cand.get("id") or cand.get("outfit_id", "")
            style_score = float(cand.get("overall_score", 0.5))
            context_score = float(ctx_map.get(outfit_id, 0.5))

            if active:
                boost = cf.get_boost_score(user_id, outfit_id)
                cf_score = boost.score
            else:
                cf_score = 0.5

            combined = (
                sw * style_score
                + cw * cf_score
                + xw * context_score
            )

            cand["style_score"] = round(style_score, 4)
            cand["cf_score"] = round(cf_score, 4)
            cand["context_score"] = round(context_score, 4)
            cand["combined_score"] = round(combined, 4)
            cand["personalization_active"] = active
            results.append(cand)

        if active:
            results.sort(key=lambda c: c["combined_score"], reverse=True)

        return results
