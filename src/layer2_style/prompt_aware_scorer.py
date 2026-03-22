"""
Prompt-Aware Scorer — Layer 2 Style
=====================================

Scores a candidate outfit against a :class:`WardrobeFilters` derived from a
free-text user prompt.

Three sub-scores are combined into one final relevance score:
  1. Color alignment     (weight from filters, default 35 %)
  2. Style alignment     (weight from filters, default 35 %)
  3. Occasion alignment  (weight from filters, default 30 %)

The final prompt score can then be blended with the classic style score via
``apply_prompt_boost``.

Public API:
    scorer = PromptAwareScorer()
    score  = scorer.score_outfit(outfit_dict, filters)
    final  = scorer.apply_prompt_boost(base_score, prompt_score, preference=0.4)
    weight_overrides = scorer.get_dynamic_weights(filters)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.core import get_logger
from src.layer3_context.prompt_to_filters import PromptToFilters, WardrobeFilters

logger = get_logger(__name__)

_translator = PromptToFilters()

# Occasion → scorer weight multipliers
# key: scorer name  value: multiplier applied to base weight
_OCCASION_WEIGHT_BOOSTS: Dict[str, Dict[str, float]] = {
    "wedding": {"occasion": 2.0, "creativity": 0.5},
    "gala": {"occasion": 2.0, "creativity": 0.5},
    "prom": {"occasion": 1.8, "creativity": 0.6},
    "work": {"formality": 1.6, "occasion": 1.4, "sandwich_rule": 0.7},
    "interview": {"formality": 1.8, "occasion": 1.5},
    "date": {"style": 1.4, "occasion": 1.2},
    "beach": {"color": 1.3, "formality": 0.4},
    "gym": {"formality": 0.2, "color": 0.8},
    "casual": {"creativity": 1.2, "formality": 0.4},
    "funeral": {"formality": 1.8, "occasion": 2.0, "creativity": 0.3},
}

_STYLE_WEIGHT_BOOSTS: Dict[str, Dict[str, float]] = {
    "minimalist": {"three_color": 1.5, "pattern_mixing": 0.4},
    "romantic": {"style": 1.3, "color": 1.2},
    "elegant": {"formality": 1.4, "style": 1.3},
    "sporty": {"formality": 0.5, "comfort": 1.5},
    "edgy": {"creativity": 1.4, "color": 1.2},
    "bohemian": {"creativity": 1.3, "color": 1.1},
}


class PromptAwareScorer:
    """
    Scores outfit candidates against a parsed user prompt.

    All sub-scores are deterministic (no LLM calls).
    """

    # ------------------------------------------------------------------
    # Main scoring entry-point
    # ------------------------------------------------------------------

    def score_outfit(
        self,
        outfit: Dict[str, Any],
        filters: WardrobeFilters,
    ) -> float:
        """
        Return a relevance score in [0, 1] for how well *outfit* matches *filters*.

        outfit dict keys (all optional):
            dominant_colors: list[str]
            dominant_styles: dict[str, float]
            formality_score: float | None
            occasion_tags:   list[str]
            style_tags:      list[str]
        """
        color_score = self._color_score(outfit, filters)
        style_score = self._style_score(outfit, filters)
        occasion_score = self._occasion_score(outfit, filters)

        total = (
            filters.weight_color * color_score
            + filters.weight_style * style_score
            + filters.weight_occasion * occasion_score
        )
        return min(1.0, max(0.0, total))

    # ------------------------------------------------------------------
    # Sub-scores
    # ------------------------------------------------------------------

    def _color_score(self, outfit: Dict[str, Any], filters: WardrobeFilters) -> float:
        """How well the outfit's dominant colors match the filter's color criteria."""
        if not filters.target_colors and not filters.color_families:
            return 0.5  # No colour constraint → neutral

        dominant = [c.lower() for c in (outfit.get("dominant_colors") or [])]
        if not dominant:
            return 0.3  # Unknown colors → slight penalty

        return _translator.garment_matches_color_filter(dominant, filters)

    def _style_score(self, outfit: Dict[str, Any], filters: WardrobeFilters) -> float:
        """How well the outfit's style distribution matches the filter's styles."""
        if not filters.target_styles:
            return 0.5

        styles: Dict[str, float] = outfit.get("dominant_styles") or {}
        if not styles:
            # Fall back to style_tags if present
            style_tags: List[str] = outfit.get("style_tags") or []
            if not style_tags:
                return 0.3
            styles = {t: 1.0 for t in style_tags}

        return _translator.garment_matches_style_filter(styles, filters)

    def _occasion_score(self, outfit: Dict[str, Any], filters: WardrobeFilters) -> float:
        """How well the outfit's occasion tags and formality match the filter."""
        score = 0.5  # Neutral baseline

        # Formality range check
        formality = outfit.get("formality_score")
        if formality is not None:
            in_range = filters.formality_min <= formality <= filters.formality_max
            out_distance = 0.0
            if not in_range:
                out_distance = min(
                    abs(formality - filters.formality_min),
                    abs(formality - filters.formality_max),
                )
            formality_score = 1.0 if in_range else max(0.0, 1.0 - out_distance * 2)
        else:
            formality_score = 0.5

        # Occasion tag check
        occasion_score = 0.5
        if filters.target_occasion:
            tags = [t.lower() for t in (outfit.get("occasion_tags") or [])]
            if filters.target_occasion.lower() in tags:
                occasion_score = 1.0
            elif tags:
                occasion_score = 0.2  # Has tags but wrong occasion

        score = 0.5 * formality_score + 0.5 * occasion_score
        return score

    # ------------------------------------------------------------------
    # Boost application
    # ------------------------------------------------------------------

    def apply_prompt_boost(
        self,
        base_score: float,
        prompt_score: float,
        preference: float = 0.4,
    ) -> float:
        """
        Blend *base_score* (classic style) with *prompt_score* (prompt relevance).

        preference=0.0 → ignore prompt, use base score only
        preference=1.0 → ignore base score, use prompt score only
        """
        preference = max(0.0, min(1.0, preference))
        blended = (1.0 - preference) * base_score + preference * prompt_score
        return max(0.0, min(1.0, blended))

    # ------------------------------------------------------------------
    # Dynamic weight report (for external logging / debugging)
    # ------------------------------------------------------------------

    def get_dynamic_weights(self, filters: WardrobeFilters) -> Dict[str, float]:
        """
        Return a dict of scorer-name → relative multiplier based on the filters.

        This is informational: it shows which scorers should be emphasised when
        re-running the classic pipeline with the prompt's context.
        """
        boosts: Dict[str, float] = {
            "color": 1.0,
            "style": 1.0,
            "occasion": 1.0,
            "formality": 1.0,
            "creativity": 1.0,
            "pattern_mixing": 1.0,
            "three_color": 1.0,
            "sandwich_rule": 1.0,
            "comfort": 1.0,
        }

        if filters.target_occasion and filters.target_occasion in _OCCASION_WEIGHT_BOOSTS:
            for scorer, mult in _OCCASION_WEIGHT_BOOSTS[filters.target_occasion].items():
                boosts[scorer] = boosts.get(scorer, 1.0) * mult

        for style in filters.target_styles:
            if style in _STYLE_WEIGHT_BOOSTS:
                for scorer, mult in _STYLE_WEIGHT_BOOSTS[style].items():
                    boosts[scorer] = boosts.get(scorer, 1.0) * mult

        return boosts

    # ------------------------------------------------------------------
    # Garment-level hard filter
    # ------------------------------------------------------------------

    def passes_hard_filters(
        self,
        garment: Dict[str, Any],
        filters: WardrobeFilters,
    ) -> bool:
        """
        Return False if the garment must be excluded (hard filter).

        Hard filters:
          • garment_type in excluded_types
          • formality_score outside formality range (strict mode)
        """
        garment_type = (garment.get("garment_type") or "").lower()
        if garment_type in [t.lower() for t in filters.excluded_types]:
            return False
        return True

    def rank_outfits(
        self,
        outfits: List[Dict[str, Any]],
        filters: WardrobeFilters,
        base_scores: Optional[List[float]] = None,
        preference: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank *outfits* by blended prompt + base scores.

        Args:
            outfits:      list of outfit dicts
            filters:      WardrobeFilters from the prompt
            base_scores:  optional pre-computed style scores (same order as outfits)
            preference:   0-1, how much weight to give prompt score vs base score

        Returns:
            Outfits sorted descending by blended score, each augmented with
            ``prompt_score`` and ``final_score`` keys.
        """
        if base_scores is None:
            base_scores = [0.5] * len(outfits)

        results = []
        for outfit, base in zip(outfits, base_scores):
            p_score = self.score_outfit(outfit, filters)
            final = self.apply_prompt_boost(base, p_score, preference)
            results.append({**outfit, "prompt_score": round(p_score, 4), "final_score": round(final, 4)})

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results
