"""PackingScoreCalculator — measures how versatile a packing selection is.

Scoring formula (0-100):
    score = versatility_ratio * occasion_coverage * colour_cohesion * 100

Where:
    versatility_ratio  = outfit_combinations / pieces_packed          (target > 2.0, capped at 3.0)
    occasion_coverage  = fraction of required occasion days covered   (0-1)
    colour_cohesion    = garment colour cohesion score                (0-1, from CapsuleAnalyzer)

The score is intentionally a *product* so that a pack that fails any single
dimension cannot artificially inflate the total.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

from src.core.models import Garment
from src.core.travel_models import PackingPlan, PackedGarment

logger = logging.getLogger(__name__)


class PackingScoreCalculator:
    """Compute a versatility score for a packing selection.

    This class is intentionally stateless (no config needed) so it can be
    embedded cheaply inside any planner loop.

    Optimisation notes
    ------------------
    * outfit_combinations is approximated as ``n_tops * n_bottoms + n_dress``
      (same heuristic used by OutfitBuilder) — O(1) per call.
    * No external API calls; all inputs are pre-computed.
    """

    # Caps to prevent runaway scores
    _MAX_VERSATILITY_RATIO: float = 3.0
    _OUTFIT_BONUS_PER_SHOE_PAIR: float = 0.5   # shoes multiply outfit variety
    _OUTFIT_BONUS_PER_OUTERWEAR: float = 0.3

    def __call__(
        self,
        packed: List[Garment],
        occasions: Dict[str, int],
        colour_cohesion: float = 1.0,
    ) -> float:
        """Return packing score in [0, 100].

        Parameters
        ----------
        packed:
            Garments selected for packing.
        occasions:
            ``{occasion_name: days_count}`` mapping (e.g. ``{"casual": 4, "evening": 2}``).
        colour_cohesion:
            Pre-computed cohesion score from ``WardrobeCapsuleAnalyzer`` normalised to
            [0, 1].  Defaults to 1.0 when the analyzer is not available.
        """
        if not packed:
            return 0.0

        n_pieces = len(packed)

        # --- versatility ratio (O(n)) ---
        cat_counts = self._count_categories(packed)
        combos = self._estimate_outfit_combinations(cat_counts)
        raw_ratio = combos / n_pieces
        capped_ratio = min(raw_ratio, self._MAX_VERSATILITY_RATIO)
        versatility_ratio = capped_ratio / self._MAX_VERSATILITY_RATIO  # normalise to [0,1]

        # --- occasion coverage (O(n_occasions)) ---
        occasion_coverage = self._occasion_coverage(packed, occasions)

        # --- colour cohesion: already 0-1 ---
        colour_cohesion = max(0.0, min(1.0, colour_cohesion))

        # --- final product score ---
        raw_score = versatility_ratio * occasion_coverage * colour_cohesion * 100.0
        # Small bonus for larger packs that still remain efficient (avoid harsh penalty)
        efficiency_bonus = math.log1p(n_pieces) * 0.5
        score = min(100.0, raw_score + efficiency_bonus)

        logger.debug(
            "PackingScore: ratio=%.2f cov=%.2f cohesion=%.2f => %.1f",
            versatility_ratio,
            occasion_coverage,
            colour_cohesion,
            score,
        )
        return round(score, 1)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def score_label(score: float) -> str:
        """Human-readable label for a packing score."""
        if score >= 80:
            return "excellent"
        if score >= 60:
            return "good"
        if score >= 40:
            return "fair"
        return "poor"

    def versatility_ratio(self, packed: List[Garment]) -> float:
        """Return raw outfit_combinations / pieces ratio (uncapped)."""
        if not packed:
            return 0.0
        cat_counts = self._count_categories(packed)
        combos = self._estimate_outfit_combinations(cat_counts)
        return round(combos / len(packed), 2)

    # ------------------------------------------------------------------
    # private
    # ------------------------------------------------------------------

    @staticmethod
    def _count_categories(packed: List[Garment]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for g in packed:
            cat = g.attributes.category.value if hasattr(g.attributes, "category") else "unknown"
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _estimate_outfit_combinations(self, cat_counts: Dict[str, int]) -> float:
        """Estimate number of outfits without enumerating all combinations.

        Formula:
            base = tops * bottoms + dresses
            multiplied by shoe_bonus^n_shoe_pairs * outerwear_bonus^n_outerwear
        """
        tops = cat_counts.get("top", 0)
        bottoms = cat_counts.get("bottom", 0)
        dresses = cat_counts.get("dress", 0)
        shoes = cat_counts.get("shoes", 0)
        outerwear = cat_counts.get("outerwear", 0)

        base = tops * bottoms + dresses
        if base == 0:
            base = max(1, sum(cat_counts.values()) // 2)

        # shoes and outerwear multiply variety
        shoe_factor = 1.0 + (shoes - 1) * self._OUTFIT_BONUS_PER_SHOE_PAIR if shoes > 1 else 1.0
        outer_factor = 1.0 + outerwear * self._OUTFIT_BONUS_PER_OUTERWEAR

        return base * shoe_factor * outer_factor

    @staticmethod
    def _occasion_coverage(packed: List[Garment], occasions: Dict[str, int]) -> float:
        """Fraction of occasion types covered by at least one garment."""
        if not occasions:
            return 1.0

        occasion_styles: Dict[str, List[str]] = {
            "casual": ["casual", "relaxed", "everyday"],
            "beach": ["casual", "summer", "beach"],
            "tourism": ["casual", "comfortable"],
            "business": ["business", "professional", "formal"],
            "formal": ["formal", "elegant", "evening"],
            "evening": ["smart_casual", "evening", "formal"],
            "sport": ["sport", "athletic", "outdoor"],
            "outdoor": ["outdoor", "casual", "sport"],
        }

        covered = 0
        total = len(occasions)

        for occasion in occasions:
            target_styles = set(occasion_styles.get(occasion, [occasion]))
            for g in packed:
                garment_styles = set(g.attributes.style_tags or [])
                garment_formality = g.attributes.formality_level.value if hasattr(g.attributes, "formality_level") else ""
                if target_styles & garment_styles or garment_formality in target_styles:
                    covered += 1
                    break
            else:
                # Fallback: count as covered if there's at least one garment at the right formality level
                _formality_map = {
                    "casual": ["very_casual", "casual"],
                    "beach": ["very_casual", "casual"],
                    "tourism": ["casual", "smart_casual"],
                    "business": ["business_casual", "business"],
                    "formal": ["formal", "black_tie"],
                    "evening": ["smart_casual", "business_casual", "formal"],
                    "sport": ["very_casual"],
                    "outdoor": ["casual", "very_casual"],
                }
                ok_formalities = set(_formality_map.get(occasion, ["casual"]))
                for g in packed:
                    fv = g.attributes.formality_level.value if hasattr(g.attributes, "formality_level") else "casual"
                    if fv in ok_formalities:
                        covered += 1
                        break

        return covered / total if total > 0 else 1.0
