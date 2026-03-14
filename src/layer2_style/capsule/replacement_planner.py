"""
Replacement Planner — Fonctionnalité 3
=======================================

Pour chaque paire redondante, décide laquelle des deux pièces garder
et planifie un calendrier de remplacement progressif.

Optimisations :
  • Utilise les versatility_scores déjà calculés par WardrobeCapsuleAnalyzer
    → 0 appel DB supplémentaire
  • Un seul appel LLM en batch via CapsuleExplainer (appelé séparément)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.core import get_logger
from src.core.models import (
    CapsuleGarmentScore,
    RedundantPair,
    ReplacementPlanResult,
    ReplacementVerdict,
)

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "data" / "capsule_config.json"

# Minimum versatility gain to warrant a replacement recommendation
_MIN_GAIN_THRESHOLD = 0.05

_TIMING_RULES = [
    # (min_confidence, season_hint) → transition message
    (0.85, "Replace at next seasonal refresh — strong quality difference."),
    (0.65, "Replace within 2-3 months — meaningful versatility gain."),
    (0.45, "Consider replacing when the weaker piece shows wear."),
    (0.00, "Keep both for now; differences are marginal."),
]


def _load_config() -> Dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


class ReplacementPlanner:
    """
    Recommends which piece in each redundant pair should be kept or removed.

    Decision logic (all local — no DB calls):
      • winner  = garment with higher versatility_score
      • gain    = winner.versatility_score − loser.versatility_score
      • confidence scaled to [0,1] by dividing by max observed gain
      • If gain < threshold, verdict is "keep both"

    Usage:
        planner = ReplacementPlanner()
        result = planner.plan(redundant_pairs, garment_scores)
    """

    def __init__(self) -> None:
        self._cfg = _load_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        redundant_pairs: List[RedundantPair],
        garment_scores: List[CapsuleGarmentScore],
    ) -> ReplacementPlanResult:
        """
        Generate a replacement plan.

        Parameters
        ----------
        redundant_pairs:
            From CapsuleAnalysisResult.redundant_pairs.
        garment_scores:
            From CapsuleAnalysisResult.garment_scores — provides
            versatility_score for each garment.
        """
        if not redundant_pairs:
            logger.info("ReplacementPlanner: no redundant pairs — nothing to plan")
            return ReplacementPlanResult(verdicts=[], total_outfits_gained=0, summary="No redundant pairs detected.")

        score_map: Dict[str, CapsuleGarmentScore] = {gs.garment_id: gs for gs in garment_scores}

        verdicts: List[ReplacementVerdict] = []
        total_gain = 0

        for pair in redundant_pairs:
            verdict = self._evaluate_pair(pair, score_map)
            if verdict is not None:
                verdicts.append(verdict)
                total_gain += verdict.versatility_gain

        verdicts.sort(key=lambda v: v.confidence, reverse=True)

        if not verdicts:
            summary = "All redundant pairs have similar versatility — no replacements needed yet."
        else:
            summary = (
                f"{len(verdicts)} replacement(s) recommended. "
                f"Total estimated versatility gain: {total_gain:.2f} across your wardrobe."
            )

        return ReplacementPlanResult(
            verdicts=verdicts,
            total_outfits_gained=self._estimate_outfit_gain(verdicts),
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_pair(
        self,
        pair: RedundantPair,
        score_map: Dict[str, CapsuleGarmentScore],
    ) -> Optional[ReplacementVerdict]:
        score_a = score_map.get(pair.garment_a_id)
        score_b = score_map.get(pair.garment_b_id)

        if score_a is None or score_b is None:
            logger.debug("Missing garment score for pair (%s, %s)", pair.garment_a_id, pair.garment_b_id)
            return None

        v_a = score_a.versatility_score
        v_b = score_b.versatility_score
        gain = abs(v_a - v_b)

        if gain < _MIN_GAIN_THRESHOLD:
            return None  # Too close to call — keep both

        keep_id = pair.garment_a_id if v_a >= v_b else pair.garment_b_id
        remove_id = pair.garment_b_id if v_a >= v_b else pair.garment_a_id

        confidence = self._confidence_from_gain(gain)
        timing = self._timing_for_confidence(confidence)

        return ReplacementVerdict(
            garment_keep_id=keep_id,
            garment_remove_id=remove_id,
            versatility_gain=round(gain, 4),
            confidence=round(confidence, 3),
            transition_timing=timing,
        )

    @staticmethod
    def _confidence_from_gain(gain: float) -> float:
        """Map versatility gain to a [0, 1] confidence score."""
        # gain is in [0, 1]; 0.40+ is nearly certain
        return min(gain / 0.40, 1.0)

    @staticmethod
    def _timing_for_confidence(confidence: float) -> str:
        for threshold, message in _TIMING_RULES:
            if confidence >= threshold:
                return message
        return _TIMING_RULES[-1][1]

    @staticmethod
    def _estimate_outfit_gain(verdicts: List[ReplacementVerdict]) -> int:
        """
        Very rough estimate: each high-confidence replacement typically
        frees up 3-5 new outfit combinations.
        """
        total = 0
        for v in verdicts:
            if v.confidence >= 0.85:
                total += 5
            elif v.confidence >= 0.65:
                total += 3
            else:
                total += 1
        return total
