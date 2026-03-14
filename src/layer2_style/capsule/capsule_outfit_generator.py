"""
Capsule Outfit Generator — Fonctionnalité 4
============================================

Filtre et classe les tenues déjà calculées en Phase 3 selon leur
score capsule (% de pièces clés × versatilité moyenne).

Optimisations :
  • Zéro appel OutfitBuilder — on réutilise les OutfitCandidate de Phase 3
  • Zéro appel DB
  • Tier assignment via seuils de capsule_config.json
  • Occasions déduites localement depuis les attributs des pièces
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.core import get_logger
from src.core.models import (
    CapsuleAnalysisResult,
    CapsuleGarmentScore,
    CapsuleGarmentRole,
    CapsuleOutfit,
    CapsuleOutfitsResult,
    FormalityLevel,
)
from src.layer2_style.outfit_builder import OutfitCandidate

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "data" / "capsule_config.json"

# Tier thresholds (overridden by config if present)
_TIER_BASIC = 85.0
_TIER_SEMI = 70.0

_FORMALITY_TO_OCCASIONS: Dict[str, List[str]] = {
    "ultra_casual": ["weekend", "errands"],
    "casual": ["daily_wear", "weekend"],
    "smart_casual": ["work", "casual_dinner", "weekend"],
    "business": ["work", "meeting", "interview"],
    "formal": ["event", "cocktail", "evening"],
    "black_tie": ["gala", "evening", "formal_event"],
}


def _load_config() -> Dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


class CapsuleOutfitGenerator:
    """
    Scores existing OutfitCandidates (from Phase 3) using capsule criteria
    and returns structured CapsuleOutfitsResult.

    capsule_score formula:
        capsule_score = (0.6 × pct_key_pieces + 0.4 × avg_versatility) × 100

    Tier assignment (from config or defaults):
        basic        → score ≥ 85
        semi_creative → 70 ≤ score < 85
        creative     → score < 70
    """

    def __init__(self) -> None:
        self._cfg = _load_config()
        viz_cfg = self._cfg.get("visualization", {})
        self._tier_basic = float(viz_cfg.get("tier_basic_threshold", _TIER_BASIC))
        self._tier_semi = float(viz_cfg.get("tier_semi_threshold", _TIER_SEMI))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        candidates: List[OutfitCandidate],
        analysis: CapsuleAnalysisResult,
        top_n: int = 12,
    ) -> CapsuleOutfitsResult:
        """
        Score and rank existing outfit candidates.

        Parameters
        ----------
        candidates:
            OutfitCandidate list from Phase 3 — NOT regenerated.
        analysis:
            CapsuleAnalysisResult from WardrobeCapsuleAnalyzer.
        top_n:
            Maximum number of capsule outfits to return.
        """
        if not candidates:
            logger.warning("CapsuleOutfitGenerator: no candidates provided")
            return CapsuleOutfitsResult(outfits=[], basic_count=0, semi_creative_count=0, creative_count=0)

        score_map: Dict[str, CapsuleGarmentScore] = {
            gs.garment_id: gs for gs in analysis.garment_scores
        }
        key_ids: Set[str] = {gs.garment_id for gs in analysis.garment_scores
                             if gs.capsule_role == CapsuleGarmentRole.KEY_PIECE}

        capsule_outfits: List[CapsuleOutfit] = []
        for candidate in candidates:
            outfit = self._score_candidate(candidate, score_map, key_ids)
            if outfit is not None:
                capsule_outfits.append(outfit)

        # Sort by capsule_score desc, take top_n
        capsule_outfits.sort(key=lambda o: o.capsule_score, reverse=True)
        capsule_outfits = capsule_outfits[:top_n]

        # Assign ranks
        for rank, outfit in enumerate(capsule_outfits, start=1):
            outfit.rank = rank

        basic = sum(1 for o in capsule_outfits if o.tier == "basic")
        semi = sum(1 for o in capsule_outfits if o.tier == "semi_creative")
        creative = sum(1 for o in capsule_outfits if o.tier == "creative")

        logger.info(
            "CapsuleOutfitGenerator: %d outfits (basic=%d, semi=%d, creative=%d)",
            len(capsule_outfits), basic, semi, creative,
        )

        return CapsuleOutfitsResult(
            outfits=capsule_outfits,
            basic_count=basic,
            semi_creative_count=semi,
            creative_count=creative,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_candidate(
        self,
        candidate: OutfitCandidate,
        score_map: Dict[str, CapsuleGarmentScore],
        key_ids: Set[str],
    ) -> Optional[CapsuleOutfit]:
        garment_ids = [g.id for g in candidate.garments if hasattr(g, "id")]
        if not garment_ids:
            # Fallback: try garment name as id
            garment_ids = [g.name for g in candidate.garments if hasattr(g, "name")]
        if not garment_ids:
            return None

        # Versatility scores — fallback to overall_score if not in map
        versatility_values = []
        for gid in garment_ids:
            if gid in score_map:
                versatility_values.append(score_map[gid].versatility_score)
            else:
                versatility_values.append(candidate.overall_score)

        avg_versatility = sum(versatility_values) / len(versatility_values) if versatility_values else 0.5
        pct_key = sum(1 for gid in garment_ids if gid in key_ids) / len(garment_ids)
        capsule_score = (0.6 * pct_key + 0.4 * avg_versatility) * 100

        tier = self._assign_tier(capsule_score)
        occasions = self._infer_occasions(candidate)

        return CapsuleOutfit(
            rank=0,  # set after sorting
            garment_ids=garment_ids,
            capsule_score=round(capsule_score, 1),
            pct_key_pieces=round(pct_key, 3),
            avg_versatility=round(avg_versatility, 3),
            tier=tier,
            occasions=occasions,
        )

    def _assign_tier(self, score: float) -> str:
        if score >= self._tier_basic:
            return "basic"
        if score >= self._tier_semi:
            return "semi_creative"
        return "creative"

    @staticmethod
    def _infer_occasions(candidate: OutfitCandidate) -> List[str]:
        """Derive occasion tags from garment formality levels."""
        seen: Set[str] = set()
        for garment in candidate.garments:
            attrs = getattr(garment, "attributes", None)
            if attrs is None:
                continue
            occ_profile = getattr(attrs, "occasion_profile", None)
            if occ_profile is None:
                continue
            formality = getattr(occ_profile, "formality_level", None)
            if formality is None:
                continue
            level_key = formality.value if hasattr(formality, "value") else str(formality)
            for occ in _FORMALITY_TO_OCCASIONS.get(level_key, ["daily_wear"]):
                seen.add(occ)
        return sorted(seen) if seen else ["daily_wear"]
