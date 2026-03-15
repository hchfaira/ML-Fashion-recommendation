"""SeasonTransitionAdvisor — build a phased wardrobe transition plan.

The advisor reads ``season_transition_data.json`` and produces a 3-phase plan:

  Phase 1 — Immediate swaps  (week 1): store explicitly off-season items,
                                        pull out obvious season-starters.
  Phase 2 — Color refresh    (week 2): retire clashing colors, introduce
                                        seasonal palette.
  Phase 3 — Gap filling      (week 3): identify and shop for missing pieces.

Each garment is mapped to an action (``store`` | ``keep`` | ``buy`` | ``layer``)
based on material, color, and the transition rules from config.

Complexity: O(n) per call — single pass over the wardrobe.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from config import get_config
from src.core.models import Garment
from src.core.travel_models import (
    SeasonTransitionPlan,
    TransitionAction,
    TransitionPhase,
)

logger = logging.getLogger(__name__)

# Map (from_season, to_season) → key in season_pairs
_PAIR_KEY: Dict[Tuple[str, str], str] = {
    ("summer", "fall"):   "summer_to_fall",
    ("fall", "winter"):   "fall_to_winter",
    ("winter", "spring"): "winter_to_spring",
    ("spring", "summer"): "spring_to_summer",
    # Allow "autumn" as alias
    ("summer", "autumn"): "summer_to_fall",
    ("autumn", "winter"): "fall_to_winter",
}


class SeasonTransitionAdvisor:
    """Produce a phased wardrobe transition plan between two seasons.

    Usage
    -----
    ::

        advisor = SeasonTransitionAdvisor()
        plan = advisor.advise(
            garments=wardrobe,
            current_season="summer",
            target_season="fall",
        )
    """

    def __init__(self, config_path=None):
        cfg = get_config()
        raw = cfg.get_data("season_transition_data", default={})
        self._pairs: Dict = raw.get("season_pairs", {})
        self._readiness: Dict = raw.get("season_readiness", {})
        self._phases: Dict = raw.get("transition_phases", {})
        self._adaptability: Dict = raw.get("adaptability_score", {})
        logger.debug("SeasonTransitionAdvisor loaded — pairs: %s", list(self._pairs.keys()))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advise(
        self,
        garments: List[Garment],
        current_season: str,
        target_season: str,
    ) -> SeasonTransitionPlan:
        """Build transition plan.

        Parameters
        ----------
        garments:
            Full wardrobe.
        current_season:
            Season currently active (``spring``, ``summer``, ``fall`` / ``autumn``, ``winter``).
        target_season:
            Season to transition into.

        Returns
        -------
        SeasonTransitionPlan
        """
        cur = current_season.lower()
        tgt = target_season.lower()

        pair_key = _PAIR_KEY.get((cur, tgt))
        pair_rules = self._pairs.get(pair_key, {}) if pair_key else {}

        if not pair_rules:
            logger.warning(
                "No transition rules found for %s → %s; using generic plan.", cur, tgt
            )

        target_rules = self._readiness.get(tgt, {})

        actions: List[TransitionAction] = []
        store_ids: List[str] = []
        keep_ids: List[str] = []

        # --- Phase 1: Per-garment store/keep decisions ---
        for garment in garments:
            action = self._classify_garment(garment, pair_rules, target_rules)
            actions.append(action)
            if action.action == "store":
                store_ids.append(garment.id)
            else:
                keep_ids.append(garment.id)

        # --- Phase 2: Color shift notes ---
        color_notes = self._color_shift_notes(pair_rules)
        for note in color_notes:
            actions.append(TransitionAction(
                action="layer",
                description=note,
                phase=TransitionPhase.PHASE_2,
                priority=3,
            ))

        # --- Phase 3: Shopping suggestions ---
        buy_descriptions = pair_rules.get("shopping_priority", [])
        for desc in buy_descriptions:
            actions.append(TransitionAction(
                action="buy",
                description=f"Consider buying: {desc}",
                phase=TransitionPhase.PHASE_3,
                priority=2,
            ))

        # Sort actions: phase ascending, priority ascending
        actions.sort(key=lambda a: (a.phase.value, a.priority))

        phase_summary = {
            TransitionPhase.PHASE_1.value: self._phases.get("phase_1", {}).get("description", "Store off-season items"),
            TransitionPhase.PHASE_2.value: self._phases.get("phase_2", {}).get("description", "Refresh color palette"),
            TransitionPhase.PHASE_3.value: self._phases.get("phase_3", {}).get("description", "Fill wardrobe gaps"),
        }

        return SeasonTransitionPlan(
            current_season=cur,
            target_season=tgt,
            actions=actions,
            store_ids=store_ids,
            keep_ids=keep_ids,
            buy_descriptions=buy_descriptions,
            color_shift_notes=color_notes,
            phase_summary=phase_summary,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_garment(
        self, garment: Garment, pair_rules: Dict, target_rules: Dict
    ) -> TransitionAction:
        """Classify one garment as store, keep, or layer (Phase 1)."""
        mat = self._get_material(garment).lower()
        color = self._get_color(garment).lower()
        cat_str = self._get_category(garment).lower()

        # Check explicit store/keep lists from pair rules
        store_keywords = [k.lower() for k in pair_rules.get("store", [])]
        keep_keywords = [k.lower() for k in pair_rules.get("keep_adaptable", [])]

        for kw in store_keywords:
            if kw in mat or kw in color or kw in cat_str:
                return TransitionAction(
                    action="store",
                    garment_id=garment.id,
                    description=f"Store: {self._desc(garment)} (off-season)",
                    phase=TransitionPhase.PHASE_1,
                    priority=1,
                )

        for kw in keep_keywords:
            if kw in mat or kw in color or kw in cat_str:
                return TransitionAction(
                    action="keep",
                    garment_id=garment.id,
                    description=f"Keep: {self._desc(garment)} (versatile, adaptable)",
                    phase=TransitionPhase.PHASE_1,
                    priority=2,
                )

        # Fallback: check against target season boost/penalise
        boost_mats = [m.lower() for m in target_rules.get("boost_materials", [])]
        penalise_mats = [m.lower() for m in target_rules.get("penalise_materials", [])]

        if any(bm in mat for bm in boost_mats):
            return TransitionAction(
                action="keep",
                garment_id=garment.id,
                description=f"Keep: {self._desc(garment)} (material suits {self._target_for(target_rules)})",
                phase=TransitionPhase.PHASE_1,
                priority=2,
            )

        if any(pm in mat for pm in penalise_mats):
            return TransitionAction(
                action="store",
                garment_id=garment.id,
                description=f"Store: {self._desc(garment)} (material unsuitable for next season)",
                phase=TransitionPhase.PHASE_1,
                priority=1,
            )

        # Check multi-season adaptability
        neutral_colors = [c.lower() for c in self._adaptability.get("neutral_colors", [])]
        multi_mats = [m.lower() for m in self._adaptability.get("multi_season_materials", [])]

        if any(nc in color for nc in neutral_colors) or any(mm in mat for mm in multi_mats):
            return TransitionAction(
                action="keep",
                garment_id=garment.id,
                description=f"Keep: {self._desc(garment)} (neutral/versatile — works multiple seasons)",
                phase=TransitionPhase.PHASE_1,
                priority=3,
            )

        return TransitionAction(
            action="keep",
            garment_id=garment.id,
            description=f"Keep: {self._desc(garment)} (no strong reason to store)",
            phase=TransitionPhase.PHASE_1,
            priority=4,
        )

    @staticmethod
    def _color_shift_notes(pair_rules: Dict) -> List[str]:
        """Build human-readable color shift notes from config."""
        color_shift = pair_rules.get("color_shift", {})
        notes = []
        retire = color_shift.get("retire", [])
        introduce = color_shift.get("introduce", [])
        if retire:
            notes.append(f"Retire colors: {', '.join(retire)}")
        if introduce:
            notes.append(f"Introduce colors: {', '.join(introduce)}")
        strategy = pair_rules.get("layering_strategy")
        if strategy:
            notes.append(f"Layering strategy: {strategy.replace('_', ' ')}")
        key_pieces = pair_rules.get("key_transition_pieces", [])
        if key_pieces:
            notes.append(f"Key transition pieces: {', '.join(key_pieces)}")
        return notes

    # --- attribute getters ---

    @staticmethod
    def _get_material(garment: Garment) -> str:
        try:
            return (garment.attributes.material.primary or "").lower()
        except Exception:
            return ""

    @staticmethod
    def _get_color(garment: Garment) -> str:
        try:
            return (garment.attributes.color.primary or "").lower()
        except Exception:
            return ""

    @staticmethod
    def _get_category(garment: Garment) -> str:
        try:
            return garment.attributes.category.value
        except Exception:
            return "unknown"

    @staticmethod
    def _desc(garment: Garment) -> str:
        try:
            color = (garment.attributes.color.primary or "").capitalize()
            cat = garment.attributes.category.value
            return f"{color} {cat}".strip()
        except Exception:
            return garment.id

    @staticmethod
    def _target_for(target_rules: Dict) -> str:
        """Extract a season string from target rules dict (best effort)."""
        return target_rules.get("_season", "next season")
