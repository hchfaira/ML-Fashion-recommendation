"""
Wardrobe Capsule Analyzer — Fonctionnalité 1
=============================================

Évalue à quel point une garde-robe est "capsule-ready" :
  • Score de cohésion 0-100 (pondéré : couleur, polyvalence, redondance, orphelins)
  • Identification des pièces clés / orphelins / redondants
  • Palette chromatique dominante

Optimisations :
  • Centralité Neo4j lue une seule fois (si disponible) → polyvalence en <10 ms
  • Fallback heuristique local si Neo4j absent (0 appel DB)
  • Cache basé sur le hash de la liste de garments (TTL configurable)
  • Aucun appel LLM — tout le calcul est déterministe
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core import get_logger
from src.core.models import (
    CapsuleAnalysisResult,
    CapsuleGarmentScore,
    GarmentCapsuleRole,
    Garment,
    GarmentCategory,
    RedundantPair,
)

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "data" / "capsule_config.json"

# Neutral color families that combine with everything
_NEUTRAL_FAMILIES = {"black", "white", "grey", "gray", "navy", "beige", "camel", "ivory", "cream"}

# Color families for cohesion grouping
_COLOR_FAMILIES: Dict[str, str] = {
    "black": "dark_neutral", "charcoal": "dark_neutral", "dark_grey": "dark_neutral",
    "white": "light_neutral", "ivory": "light_neutral", "cream": "light_neutral", "off-white": "light_neutral",
    "grey": "neutral", "gray": "neutral", "beige": "neutral", "camel": "neutral", "tan": "neutral",
    "navy": "cool_classic", "cobalt": "cool_classic", "royal_blue": "cool_classic", "blue": "cool_classic",
    "burgundy": "warm_deep", "rust": "warm_deep", "terracotta": "warm_deep", "oxblood": "warm_deep",
    "mustard": "warm_light", "olive": "warm_light", "khaki": "warm_light",
    "blush": "pastel", "lavender": "pastel", "sage": "pastel", "mint": "pastel", "peach": "pastel",
    "red": "bold", "green": "bold", "yellow": "bold", "orange": "bold", "pink": "bold", "purple": "bold",
}


def _load_config() -> Dict[str, Any]:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {
            "versatility_thresholds": {"orphan_max": 0.05, "key_piece_min": 0.15},
            "redundancy": {"same_category_color_similarity_threshold": 0.85},
            "cohesion_weights": {
                "color_cohesion": 0.30, "versatility_ratio": 0.30,
                "redundancy_penalty": 0.20, "orphan_penalty": 0.20,
            },
        }


class WardrobeCapsuleAnalyzer:
    """
    Analyzes a wardrobe for capsule readiness.

    Usage (sync, all local — zero DB calls):
        analyzer = WardrobeCapsuleAnalyzer()
        result = analyzer.analyze(garments)

    Usage with Neo4j centrality (richer versatility scores):
        centrality_map = await fetch_centrality_from_neo4j(client, garments)
        result = analyzer.analyze(garments, centrality_map=centrality_map)
    """

    def __init__(self) -> None:
        self._cfg = _load_config()
        self._thresholds = self._cfg.get("versatility_thresholds", {})
        self._weights = self._cfg.get("cohesion_weights", {})
        self._redundancy_cfg = self._cfg.get("redundancy", {})

        self._orphan_max: float = self._thresholds.get("orphan_max", 0.05)
        self._key_min: float = self._thresholds.get("key_piece_min", 0.15)
        self._redundancy_threshold: float = self._redundancy_cfg.get(
            "same_category_color_similarity_threshold", 0.85
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        garments: List[Garment],
        centrality_map: Optional[Dict[str, float]] = None,
        outfit_count_map: Optional[Dict[str, int]] = None,
    ) -> CapsuleAnalysisResult:
        """
        Full capsule analysis.

        Parameters
        ----------
        garments:
            All garments in the wardrobe.
        centrality_map:
            Optional dict {garment_id: centrality_score (0–1)} from Neo4j.
            When provided, used directly as the versatility signal (fastest path).
        outfit_count_map:
            Optional dict {garment_id: nb_outfits_containing_it}.
            Pre-computed by the pipeline to avoid re-generating all combinations.
        """
        if not garments:
            return CapsuleAnalysisResult(
                cohesion_score=0.0,
                color_cohesion_score=0.0,
                versatility_ratio=0.0,
                redundancy_penalty=0.0,
                orphan_penalty=0.0,
                dominant_colors=[],
                color_coverage_pct=0.0,
                recommendation="Your wardrobe is empty — start building!",
            )

        logger.info("CapsuleAnalyzer: analysing %d garments", len(garments))

        total_outfits = max(sum(outfit_count_map.values()) // max(len(garments), 1), 1) if outfit_count_map else 1

        # 1. Score each garment's versatility
        garment_scores = self._score_garments(garments, centrality_map, outfit_count_map, total_outfits)

        # 2. Detect redundant pairs
        redundant_pairs = self._detect_redundant_pairs(garments)

        # 3. Colour palette analysis
        dominant_colors, color_coverage_pct = self._analyse_color_palette(garments)
        color_cohesion = self._compute_color_cohesion(garments)

        # 4. Sub-scores
        key_ids = [gs.garment_id for gs in garment_scores if gs.capsule_role == GarmentCapsuleRole.KEY_PIECE]
        orphan_ids = [gs.garment_id for gs in garment_scores if gs.capsule_role == GarmentCapsuleRole.ORPHAN]
        redundant_ids = {p.garment_a_id for p in redundant_pairs} | {p.garment_b_id for p in redundant_pairs}

        n = len(garments)
        versatility_ratio = len(key_ids) / n if n else 0.0
        redundancy_penalty = min(len(redundant_pairs) / max(n / 2, 1), 1.0)
        orphan_penalty = len(orphan_ids) / n if n else 0.0

        # 5. Cohesion score (0-100)
        w = self._weights
        raw = (
            color_cohesion * w.get("color_cohesion", 0.30)
            + versatility_ratio * w.get("versatility_ratio", 0.30)
            + (1.0 - redundancy_penalty) * w.get("redundancy_penalty", 0.20)
            + (1.0 - orphan_penalty) * w.get("orphan_penalty", 0.20)
        )
        cohesion_score = round(raw * 100, 1)

        # 6. Projected score after removing orphans + redundant losers
        orphan_plus_redundant = len(orphan_ids) + max(len(redundant_pairs) - 1, 0)
        cleaned_n = max(n - orphan_plus_redundant, 1)
        new_orphan_ratio = max(orphan_penalty - orphan_plus_redundant / n, 0.0)
        new_redundancy_penalty = max(redundancy_penalty - len(redundant_pairs) / max(n, 1), 0.0)
        projected_raw = (
            color_cohesion * w.get("color_cohesion", 0.30)
            + min(versatility_ratio * 1.2, 1.0) * w.get("versatility_ratio", 0.30)
            + (1.0 - new_redundancy_penalty) * w.get("redundancy_penalty", 0.20)
            + (1.0 - new_orphan_ratio) * w.get("orphan_penalty", 0.20)
        )
        projected_cohesion = round(projected_raw * 100, 1)

        # 7. Determine capsule profile
        if n <= 20:
            profile = "minimalist"
        elif n <= 35:
            profile = "standard"
        else:
            profile = "rich"

        recommendation = self._build_recommendation(
            cohesion_score, len(orphan_ids), len(redundant_pairs), key_ids, n
        )

        return CapsuleAnalysisResult(
            cohesion_score=cohesion_score,
            color_cohesion_score=round(color_cohesion, 3),
            versatility_ratio=round(versatility_ratio, 3),
            redundancy_penalty=round(redundancy_penalty, 3),
            orphan_penalty=round(orphan_penalty, 3),
            dominant_colors=dominant_colors,
            color_coverage_pct=round(color_coverage_pct, 3),
            garment_scores=garment_scores,
            key_pieces=key_ids,
            orphan_pieces=orphan_ids,
            redundant_pairs=redundant_pairs,
            total_garments=n,
            total_outfits=total_outfits,
            capsule_profile=profile,
            recommendation=recommendation,
            projected_score_after_cleanup=projected_cohesion,
        )

    # ------------------------------------------------------------------
    # Versatility scoring
    # ------------------------------------------------------------------

    def _score_garments(
        self,
        garments: List[Garment],
        centrality_map: Optional[Dict[str, float]],
        outfit_count_map: Optional[Dict[str, int]],
        total_outfits: int,
    ) -> List[CapsuleGarmentScore]:
        """Score each garment's versatility using the best available signal."""
        scores: List[CapsuleGarmentScore] = []

        for g in garments:
            desc = g.attributes.subcategory or g.attributes.category.value
            color = g.attributes.color.primary if g.attributes.color else "unknown"
            full_desc = f"{color} {desc}"

            # Signal priority: Neo4j centrality > outfit_count_map > local heuristic
            if centrality_map and g.id in centrality_map:
                versatility = centrality_map[g.id]
                outfit_count = int(versatility * total_outfits)
            elif outfit_count_map and g.id in outfit_count_map:
                outfit_count = outfit_count_map[g.id]
                versatility = outfit_count / max(total_outfits, 1)
            else:
                versatility = self._heuristic_versatility(g)
                outfit_count = int(versatility * total_outfits)

            # Classify role
            if versatility >= self._key_min:
                role = GarmentCapsuleRole.KEY_PIECE
            elif versatility <= self._orphan_max:
                role = GarmentCapsuleRole.ORPHAN
            else:
                role = GarmentCapsuleRole.ACCEPTABLE

            scores.append(CapsuleGarmentScore(
                garment_id=g.id,
                garment_description=full_desc,
                versatility_score=round(versatility, 4),
                outfit_count=outfit_count,
                total_outfits=total_outfits,
                capsule_role=role,
            ))

        # Sort by versatility descending
        scores.sort(key=lambda s: s.versatility_score, reverse=True)
        return scores

    def _heuristic_versatility(self, g: Garment) -> float:
        """
        Fast local heuristic when no Neo4j data is available.

        Combines:
          • Color neutrality (0.4 weight)
          • Formality range / coverage (0.3 weight)
          • Season coverage (0.3 weight)
        """
        color = (g.attributes.color.primary or "").lower() if g.attributes.color else ""
        color_score = 0.8 if color in _NEUTRAL_FAMILIES else 0.3

        from src.core.models import FormalityLevel
        _MID_FORMALITY = {
            FormalityLevel.SMART_CASUAL,
            FormalityLevel.BUSINESS_CASUAL,
            FormalityLevel.CASUAL,
        }
        formality_score = 0.7 if g.attributes.formality_level in _MID_FORMALITY else 0.3

        seasons = g.attributes.season_suitable or []
        if g.attributes.seasonality and g.attributes.seasonality.seasons:
            seasons = g.attributes.seasonality.seasons
        season_score = min(len(seasons) / 4.0, 1.0) if seasons else 0.5

        return round(color_score * 0.4 + formality_score * 0.3 + season_score * 0.3, 4)

    # ------------------------------------------------------------------
    # Colour analysis
    # ------------------------------------------------------------------

    def _analyse_color_palette(self, garments: List[Garment]) -> tuple[List[str], float]:
        """Return (top-5 colors, coverage percentage of those 5 colors)."""
        counter: Counter = Counter(
            (g.attributes.color.primary or "unknown").lower()
            for g in garments
            if g.attributes.color
        )
        total = len(garments)
        if total == 0:
            return [], 0.0
        top5 = [c for c, _ in counter.most_common(5)]
        top5_count = sum(counter[c] for c in top5)
        return top5, round(top5_count / total, 3)

    def _compute_color_cohesion(self, garments: List[Garment]) -> float:
        """
        Color cohesion = % of garments whose color family matches the dominant family.

        A wardrobe where most pieces share 1-2 color families scores high.
        A rainbow wardrobe scores low.
        """
        if not garments:
            return 0.0

        family_counter: Counter = Counter()
        for g in garments:
            color = (g.attributes.color.primary or "unknown").lower() if g.attributes.color else "unknown"
            family = _COLOR_FAMILIES.get(color, "other")
            family_counter[family] += 1

        total = len(garments)
        # Top-2 families
        top2_count = sum(cnt for _, cnt in family_counter.most_common(2))
        # Bonus if mostly neutrals
        neutral_count = sum(
            cnt for fam, cnt in family_counter.items()
            if "neutral" in fam or fam == "cool_classic"
        )
        base_cohesion = top2_count / total
        neutral_bonus = min(neutral_count / total * 0.2, 0.2)
        return min(base_cohesion + neutral_bonus, 1.0)

    # ------------------------------------------------------------------
    # Redundancy detection
    # ------------------------------------------------------------------

    def _detect_redundant_pairs(self, garments: List[Garment]) -> List[RedundantPair]:
        """
        Detect pairs of garments in the same (sub)category with similar colours.

        Uses only local data — zero DB calls.
        """
        pairs: List[RedundantPair] = []
        threshold = self._redundancy_threshold

        # Group by category
        by_cat: Dict[str, List[Garment]] = {}
        for g in garments:
            key = g.attributes.subcategory or g.attributes.category.value
            by_cat.setdefault(key, []).append(g)

        for cat_key, items in by_cat.items():
            if len(items) < 2:
                continue
            # Compare all pairs within same subcategory
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i], items[j]
                    sim = self._color_similarity(a, b)
                    if sim >= threshold:
                        desc_a = f"{(a.attributes.color.primary or '') } {a.attributes.subcategory or a.attributes.category.value}"
                        desc_b = f"{(b.attributes.color.primary or '')} {b.attributes.subcategory or b.attributes.category.value}"
                        pairs.append(RedundantPair(
                            garment_a_id=a.id,
                            garment_a_description=desc_a.strip(),
                            garment_b_id=b.id,
                            garment_b_description=desc_b.strip(),
                            similarity_score=round(sim, 3),
                            reason=f"Both are {cat_key} with similar colour ({a.attributes.color.primary if a.attributes.color else 'unknown'} ≈ {b.attributes.color.primary if b.attributes.color else 'unknown'})",
                        ))
        return pairs

    @staticmethod
    def _color_similarity(a: Garment, b: Garment) -> float:
        """Simple color similarity: 1.0 if same primary color, 0.7 if same family, else 0.0."""
        ca = (a.attributes.color.primary or "").lower() if a.attributes.color else ""
        cb = (b.attributes.color.primary or "").lower() if b.attributes.color else ""
        if not ca or not cb:
            return 0.0
        if ca == cb:
            return 1.0
        fa = _COLOR_FAMILIES.get(ca, ca)
        fb = _COLOR_FAMILIES.get(cb, cb)
        if fa == fb and fa != "other":
            return 0.75
        return 0.0

    # ------------------------------------------------------------------
    # Human-readable recommendation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recommendation(
        score: float,
        orphan_count: int,
        redundant_count: int,
        key_ids: List[str],
        total: int,
    ) -> str:
        parts: List[str] = []
        if score >= 80:
            parts.append("Your wardrobe is capsule-ready 🎉")
        elif score >= 60:
            parts.append("Good foundation — a few tweaks will unlock many more outfits.")
        elif score >= 40:
            parts.append("Your wardrobe has potential but needs some restructuring.")
        else:
            parts.append("Your wardrobe is quite scattered — focus on versatile basics first.")

        if orphan_count:
            parts.append(f"Remove or replace the {orphan_count} orphan piece(s) that barely combine with anything.")
        if redundant_count:
            parts.append(f"You have {redundant_count} near-duplicate pair(s) — keep the more versatile one.")
        if len(key_ids) < 3:
            parts.append("Invest in 2-3 high-versatility basics (neutral blazer, classic bottom, go-to shoes).")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Cache helpers (used by pipeline)
    # ------------------------------------------------------------------

    @staticmethod
    def wardrobe_hash(garments: List[Garment]) -> str:
        """Stable hash of garment ID list — used for cache invalidation."""
        key = ",".join(sorted(g.id for g in garments))
        return hashlib.md5(key.encode()).hexdigest()
