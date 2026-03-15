"""TravelWardrobePlanner — greedy packing list generator.

Algorithm (O(n²) worst-case, fast in practice)
-----------------------------------------------
1. Score every garment on 4 dimensions using pre-built advisors:
       score_i = W_v * versatility
               + W_o * occasion_match
               + W_w * weather_fit
               + W_m * morphology_fit

2. Select pieces one at a time (greedy), adding a *synergy bonus* each time
   a new piece pairs well with the already-selected set:
       synergy_bonus = color_family_match  * B_color
                     + style_tag_overlap   * B_style
                     + formality_match     * B_formality

3. After selection, generate a simple day-by-day schedule (assign outfit per day).

4. Calculate ``PackingScore`` using ``PackingScoreCalculator``.

Complexity
----------
* Scoring phase: O(n) — one pass per garment.
* Selection phase: O(n × max_pieces) ≈ O(n²) in the worst case.  With typical
  values of n < 100 and max_pieces ≤ 20 this is sub-millisecond.
* Synergy bonus computation: O(|selected|) per candidate — at most O(max_pieces²).

No DB calls; all advisors are in-process.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from config import get_config
from src.core.models import Garment
from src.core.travel_models import (
    DayPlan,
    PackedGarment,
    PackingPlan,
    TravelConstraints,
)
from src.layer2_style.capsule.wardrobe_capsule_analyzer import WardrobeCapsuleAnalyzer
from src.layer2_style.travel.packing_score_calculator import PackingScoreCalculator
from src.layer3_context.morphology_advisor import MorphologyAdvisor

logger = logging.getLogger(__name__)


class TravelWardrobePlanner:
    """Plan a packing list for a trip.

    Parameters
    ----------
    weather_advisor:
        Optional pre-built ``WeatherOutfitAdvisor`` instance (from
        ``WeatherAPIClient``).  If None, weather scoring is skipped.

    Usage
    -----
    ::

        planner = TravelWardrobePlanner()
        plan = planner.plan(garments, constraints)
    """

    def __init__(self, weather_advisor=None):
        cfg = get_config()
        travel_data = cfg.get_data("travel_data", default={})
        self._weights: Dict = travel_data.get("scoring_weights", {
            "versatility": 0.30,
            "occasion_match": 0.30,
            "weather_fit": 0.25,
            "morphology": 0.15,
        })
        self._synergy_bonuses: Dict = travel_data.get("versatility_synergy_bonus", {
            "color_family_match": 0.10,
            "style_tag_overlap": 0.05,
            "formality_match": 0.08,
        })
        self._score_thresholds: Dict = travel_data.get("packing_score_thresholds", {
            "excellent": 80, "good": 60, "fair": 40, "poor": 0,
        })
        self._occasion_data: Dict = cfg.get_data("occasion_data", default={})
        self._capsule_analyzer = WardrobeCapsuleAnalyzer()
        self._morphology_advisor = MorphologyAdvisor()
        self._packing_scorer = PackingScoreCalculator()
        self._weather_advisor = weather_advisor
        logger.debug("TravelWardrobePlanner initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        garments: List[Garment],
        constraints: TravelConstraints,
    ) -> PackingPlan:
        """Generate a complete packing plan.

        Parameters
        ----------
        garments:
            Full wardrobe.
        constraints:
            Trip parameters (destination, days, occasions, max_pieces, …).

        Returns
        -------
        PackingPlan
        """
        if not garments:
            return PackingPlan(
                destination=constraints.destination,
                days=constraints.days,
                max_pieces=constraints.max_pieces,
                packed_garments=[],
                day_plans=[],
                packing_score=0.0,
                score_label="poor",
                warnings=["Wardrobe is empty"],
            )

        # 1. Run capsule analysis once (gives versatility scores)
        analysis = self._capsule_analyzer.analyze(garments)
        versatility_map: Dict[str, float] = {
            gs.garment_id: gs.versatility_score
            for gs in (analysis.garment_scores or [])
        }

        # 2. Score every garment (O(n))
        scored: List[Tuple[float, Garment]] = [
            (self._base_score(g, versatility_map, constraints), g)
            for g in garments
        ]

        # 3. Greedy selection with synergy bonus (O(n * max_pieces))
        selected_garments = self._greedy_select(
            scored, constraints.max_pieces, constraints
        )

        # 4. Build PackedGarment records
        packed = self._build_packed_garments(selected_garments, versatility_map, constraints)

        # 5. Generate day schedule
        day_plans = self._build_day_plans(selected_garments, constraints)

        # 6. Compute packing score
        cohesion = (analysis.cohesion_score or 0.0) / 100.0
        raw_score = self._packing_scorer(selected_garments, constraints.occasions, cohesion)
        label = PackingScoreCalculator.score_label(raw_score)
        vr = self._packing_scorer.versatility_ratio(selected_garments)

        # 7. Occasion coverage stats
        occ_cov = self._occasion_coverage_stats(selected_garments, constraints.occasions)

        warnings = self._compute_warnings(selected_garments, constraints)

        return PackingPlan(
            destination=constraints.destination,
            days=constraints.days,
            max_pieces=constraints.max_pieces,
            packed_garments=packed,
            day_plans=day_plans,
            packing_score=raw_score,
            score_label=label,
            versatility_ratio=vr,
            occasion_coverage=occ_cov,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _base_score(
        self,
        garment: Garment,
        versatility_map: Dict[str, float],
        constraints: TravelConstraints,
    ) -> float:
        """Compute base packing score for a single garment (O(1))."""
        w = self._weights

        # Versatility (from capsule analyzer)
        v_score = versatility_map.get(garment.id, 0.5)

        # Occasion match
        occ_score = self._occasion_match_score(garment, constraints.occasions)

        # Weather fit (1 - penalty)
        weather_score = 1.0
        if self._weather_advisor:
            try:
                penalty = self._weather_advisor.penalise_item(
                    {"material": self._get_material(garment), "category": self._get_category(garment)}
                )
                weather_score = 1.0 - (penalty * 0.5)
            except Exception:
                pass

        # Morphology fit
        morph_score = 0.5  # neutral default
        if constraints.body_shape:
            try:
                morph_score = self._morphology_advisor.score_for_body_type(
                    [garment], constraints.body_shape
                )
            except Exception:
                pass

        base = (
            w.get("versatility", 0.30) * v_score
            + w.get("occasion_match", 0.30) * occ_score
            + w.get("weather_fit", 0.25) * weather_score
            + w.get("morphology", 0.15) * morph_score
        )
        return max(0.0, min(1.0, base))

    def _occasion_match_score(self, garment: Garment, occasions: Dict[str, int]) -> float:
        """How well this garment covers the trip occasions (O(occasions))."""
        if not occasions:
            return 0.7  # neutral

        formality_map = self._occasion_data.get("formality_numeric_mapping", {})
        occasion_formality = self._occasion_data.get("occasion_formality", {})

        # Numeric formality of garment
        garment_formality_str = ""
        if hasattr(garment.attributes, "formality_level"):
            garment_formality_str = garment.attributes.formality_level.value.lower()
        garment_num = formality_map.get(garment_formality_str, 2)

        total_days = sum(occasions.values()) or 1
        weighted_score = 0.0

        for occ, days in occasions.items():
            occ_formality_str = occasion_formality.get(occ, "casual")
            occ_num = formality_map.get(occ_formality_str, 2)

            # Gap between garment and occasion formality — penalise if too far
            gap = abs(garment_num - occ_num)
            match = max(0.0, 1.0 - gap * 0.2)
            weighted_score += match * (days / total_days)

        return weighted_score

    # ------------------------------------------------------------------
    # Greedy selection
    # ------------------------------------------------------------------

    def _greedy_select(
        self,
        scored: List[Tuple[float, Garment]],
        max_pieces: int,
        constraints: TravelConstraints,
    ) -> List[Garment]:
        """Greedy selection with synergy bonus.

        Time: O(n * max_pieces) where n = len(garments).
        """
        # Sort descending by base score for initial ordering
        scored.sort(key=lambda x: x[0], reverse=True)

        selected: List[Garment] = []
        selected_ids: Set[str] = set()

        # Track category counts to ensure basic coverage
        cat_counts: Dict[str, int] = {}

        for _ in range(min(max_pieces, len(scored))):
            best_score = -1.0
            best_garment: Optional[Garment] = None

            for base_score, garment in scored:
                if garment.id in selected_ids:
                    continue

                # Apply synergy bonus based on already-selected pieces
                synergy = self._synergy_bonus(garment, selected)

                # Small bonus if the category is under-represented
                cat = self._get_category(garment)
                cat_min = self._min_category_needed(cat, constraints, cat_counts)
                coverage_bonus = 0.15 if cat_min > 0 else 0.0

                total = base_score + synergy + coverage_bonus

                if total > best_score:
                    best_score = total
                    best_garment = garment

            if best_garment is None:
                break

            selected.append(best_garment)
            selected_ids.add(best_garment.id)
            cat = self._get_category(best_garment)
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        return selected

    def _synergy_bonus(self, candidate: Garment, selected: List[Garment]) -> float:
        """Compute synergy bonus of *candidate* given already-selected pieces (O(|selected|))."""
        if not selected:
            return 0.0

        bonuses = self._synergy_bonuses
        bonus = 0.0

        cand_color_family = self._color_family(candidate)
        cand_styles = set(candidate.attributes.style_tags or [])
        cand_formality = self._get_formality_num(candidate)

        for sel in selected:
            sel_color_family = self._color_family(sel)
            if cand_color_family and cand_color_family == sel_color_family:
                bonus += bonuses.get("color_family_match", 0.10)
                break  # count once

        for sel in selected:
            sel_styles = set(sel.attributes.style_tags or [])
            if cand_styles & sel_styles:
                bonus += bonuses.get("style_tag_overlap", 0.05)
                break

        for sel in selected:
            sel_formality = self._get_formality_num(sel)
            if abs(cand_formality - sel_formality) <= 1:
                bonus += bonuses.get("formality_match", 0.08)
                break

        return min(bonus, 0.25)  # cap synergy so it doesn't overwhelm base score

    # ------------------------------------------------------------------
    # Day schedule builder
    # ------------------------------------------------------------------

    def _build_day_plans(
        self, selected: List[Garment], constraints: TravelConstraints
    ) -> List[DayPlan]:
        """Simple round-robin day assignment.

        Expand occasions to a day list, then assign a garment subset per day.
        """
        day_occasions = self._expand_occasions(constraints.occasions, constraints.days)

        # Group garments by rough occasion suitability
        occ_garments: Dict[str, List[str]] = {}
        for g in selected:
            for occ in day_occasions:
                occ_garments.setdefault(occ, [])
            formality_str = g.attributes.formality_level.value if hasattr(g.attributes, "formality_level") else "casual"
            # Assign garment to best matching occasion
            best_occ = self._best_occasion_for_garment(formality_str, day_occasions)
            occ_garments.setdefault(best_occ, []).append(g.id)

        plans: List[DayPlan] = []
        # Track usage to avoid over-repeating
        usage: Dict[str, int] = {g.id: 0 for g in selected}
        occ_idx: Dict[str, int] = {}

        for day_num, occ in enumerate(day_occasions, start=1):
            pool = occ_garments.get(occ, [g.id for g in selected])
            # Pick 2-3 garments: least-used first
            pool_sorted = sorted(pool, key=lambda gid: usage.get(gid, 0))
            outfit_ids = pool_sorted[:3]

            for gid in outfit_ids:
                usage[gid] = usage.get(gid, 0) + 1

            plans.append(DayPlan(
                day=day_num,
                occasion=occ,
                garment_ids=outfit_ids,
                outfit_name=f"Day {day_num} — {occ.capitalize()}",
            ))

        return plans

    # ------------------------------------------------------------------
    # PackedGarment builder
    # ------------------------------------------------------------------

    def _build_packed_garments(
        self,
        selected: List[Garment],
        versatility_map: Dict[str, float],
        constraints: TravelConstraints,
    ) -> List[PackedGarment]:
        packed = []
        for g in selected:
            cat = self._get_category(g)
            color = ""
            if hasattr(g.attributes, "color") and g.attributes.color:
                color = getattr(g.attributes.color, "primary", "") or ""
            desc = f"{color} {cat}".strip()

            weather_penalty = 0.0
            if self._weather_advisor:
                try:
                    p = self._weather_advisor.penalise_item({
                        "material": self._get_material(g),
                        "category": cat,
                    })
                    weather_penalty = p * 0.5
                except Exception:
                    pass

            score = self._base_score(g, versatility_map, constraints)
            styles = g.attributes.style_tags or []

            packed.append(PackedGarment(
                garment_id=g.id,
                category=cat,
                description=desc,
                packing_score=round(score, 3),
                occasion_tags=styles[:5],
                weather_penalty=round(weather_penalty, 3),
            ))
        return packed

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_category(garment: Garment) -> str:
        try:
            return garment.attributes.category.value
        except Exception:
            return "unknown"

    @staticmethod
    def _get_material(garment: Garment) -> str:
        try:
            return (garment.attributes.material.primary or "").lower()
        except Exception:
            return ""

    @staticmethod
    def _get_formality_num(garment: Garment) -> int:
        _map = {
            "very_casual": 1, "casual": 2, "smart_casual": 3,
            "business_casual": 4, "business": 5, "formal": 6, "black_tie": 7,
        }
        try:
            return _map.get(garment.attributes.formality_level.value.lower(), 2)
        except Exception:
            return 2

    @staticmethod
    def _color_family(garment: Garment) -> Optional[str]:
        _warm = {"orange", "coral", "peach", "gold", "rust", "terracotta", "cream",
                 "mustard", "warm_brown", "camel", "tan", "salmon", "amber"}
        _cool = {"blue", "navy", "purple", "pink", "silver", "gray", "burgundy",
                 "plum", "teal", "emerald", "lavender", "periwinkle"}
        try:
            c = (garment.attributes.color.primary or "").lower()
            if any(w in c for w in _warm):
                return "warm"
            if any(w in c for w in _cool):
                return "cool"
            return "neutral"
        except Exception:
            return None

    @staticmethod
    def _expand_occasions(occasions: Dict[str, int], days: int) -> List[str]:
        """Expand ``{occasion: days}`` dict into an ordered day list."""
        result: List[str] = []
        for occ, count in occasions.items():
            result.extend([occ] * count)
        # Pad with 'casual' if total days < requested
        while len(result) < days:
            result.append("casual")
        return result[:days]

    @staticmethod
    def _best_occasion_for_garment(formality_str: str, available_occasions: List[str]) -> str:
        """Pick the best matching occasion from the list for a given formality."""
        _formality_preference = {
            "very_casual": ["casual", "beach", "sport", "outdoor"],
            "casual": ["casual", "tourism", "outdoor", "beach"],
            "smart_casual": ["tourism", "casual", "evening", "business"],
            "business_casual": ["business", "tourism", "evening"],
            "business": ["business", "formal", "evening"],
            "formal": ["formal", "evening", "business"],
            "black_tie": ["formal", "evening"],
        }
        preferred = _formality_preference.get(formality_str, ["casual"])
        occ_set = set(available_occasions)
        for pref in preferred:
            if pref in occ_set:
                return pref
        return available_occasions[0] if available_occasions else "casual"

    @staticmethod
    def _min_category_needed(
        category: str, constraints: TravelConstraints, current_counts: Dict[str, int]
    ) -> int:
        """Return how many more of this category are needed for basic coverage."""
        # Rough minimum: 1 top, 1 bottom, 1 shoes per trip
        _min_counts = {"top": 2, "bottom": 2, "shoes": 1, "outerwear": 0}
        minimum = _min_counts.get(category, 0)
        have = current_counts.get(category, 0)
        return max(0, minimum - have)

    def _occasion_coverage_stats(
        self, selected: List[Garment], occasions: Dict[str, int]
    ) -> Dict[str, float]:
        """Fraction of days per occasion covered by selected garments."""
        stats: Dict[str, float] = {}
        for occ, days in occasions.items():
            covered = 0
            for g in selected:
                if self._occasion_match_score(g, {occ: 1}) >= 0.5:
                    covered += 1
            stats[occ] = round(min(1.0, covered / days), 2) if days > 0 else 0.0
        return stats

    def _compute_warnings(
        self, selected: List[Garment], constraints: TravelConstraints
    ) -> List[str]:
        warnings: List[str] = []
        cat_counts: Dict[str, int] = {}
        for g in selected:
            c = self._get_category(g)
            cat_counts[c] = cat_counts.get(c, 0) + 1

        if cat_counts.get("top", 0) + cat_counts.get("dress", 0) == 0:
            warnings.append("No tops or dresses in packing list — outfit impossible.")
        if cat_counts.get("shoes", 0) == 0:
            warnings.append("No shoes selected — consider packing at least one pair.")
        if len(selected) < 4:
            warnings.append("Very few items packed — low outfit variety expected.")
        if len(selected) >= constraints.max_pieces:
            warnings.append(
                f"Reached the {constraints.max_pieces}-piece limit; consider increasing max_pieces."
            )
        return warnings
