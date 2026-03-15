"""WeeklyRotationPlanner — generate a 7-day outfit schedule with anti-repetition.

Algorithm
---------
1. Group garments by formality bucket.
2. For each day-slot in the week, pick the outfit with the least-recently-used
   garments (anti-repetition via a usage counter and a *cooldown* set).
3. If the wardrobe is too small to avoid repetition, emit a warning and allow
   a repeat only after ``cooldown_days`` days.

Complexity: O(days * n_garments) — fast for typical wardrobe sizes.

The planner reads the weekly schedule definition from the user input (list of
``{day, session, occasion}`` slots) so it works for any schedule length,
not just 7 days.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional

from config import get_config
from src.core.models import Garment
from src.core.travel_models import RotationSlot, WeeklyRotationPlan

logger = logging.getLogger(__name__)

_DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_FORMALITY_MAP = {
    "very_casual": 1, "casual": 2, "smart_casual": 3,
    "business_casual": 4, "business": 5, "formal": 6, "black_tie": 7,
}


class WeeklyRotationPlanner:
    """Build a 7-day (or custom) outfit rotation plan.

    Parameters
    ----------
    cooldown_days:
        Minimum days before the same garment is re-used.  Defaults to 3.

    Usage
    -----
    ::

        planner = WeeklyRotationPlanner()
        plan = planner.plan(
            garments=wardrobe,
            schedule=[
                {"day": "Monday", "occasion": "work"},
                {"day": "Tuesday", "occasion": "work"},
                {"day": "Wednesday", "occasion": "casual"},
                ...
            ],
        )
    """

    def __init__(self, cooldown_days: int = 3):
        cfg = get_config()
        schedule_data = cfg.get_data("schedule_data", default={})
        self._formality_mapping: Dict[str, int] = schedule_data.get(
            "occasion_formality_mapping", {}
        )
        self._cooldown_days = cooldown_days
        logger.debug("WeeklyRotationPlanner initialised (cooldown=%d)", cooldown_days)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        garments: List[Garment],
        schedule: Optional[List[Dict]] = None,
        work_days: int = 5,
        weekend_days: int = 2,
        work_occasion: str = "work",
        weekend_occasion: str = "casual",
    ) -> WeeklyRotationPlan:
        """Generate weekly rotation plan.

        Parameters
        ----------
        garments:
            Full wardrobe list.
        schedule:
            Explicit schedule as list of ``{"day": str, "session": str, "occasion": str}``.
            If None, a default work+weekend schedule is inferred from
            *work_days* / *weekend_days*.
        work_days:
            Number of work days per week (used when *schedule* is None).
        weekend_days:
            Number of weekend days per week (used when *schedule* is None).
        work_occasion:
            Occasion label for work days.
        weekend_occasion:
            Occasion label for weekend days.

        Returns
        -------
        WeeklyRotationPlan
        """
        if not schedule:
            schedule = self._build_default_schedule(
                work_days, weekend_days, work_occasion, weekend_occasion
            )

        if not garments:
            return WeeklyRotationPlan(
                slots=[],
                repeat_rate=1.0,
                coverage_pct=0.0,
                warnings=["Wardrobe is empty — cannot plan rotation."],
            )

        # Group garments by formality bucket for fast lookup
        garment_groups = self._group_by_formality(garments)

        # Anti-repetition tracker: garment_id → last day index used
        last_used: Dict[str, int] = {}
        # utilisation counter
        utilisation: Dict[str, int] = defaultdict(int)

        slots: List[RotationSlot] = []
        repeated_days = 0

        for day_idx, slot_def in enumerate(schedule):
            day = slot_def.get("day", _DAYS_OF_WEEK[day_idx % 7])
            session = slot_def.get("session", "full_day")
            occasion = slot_def.get("occasion", "casual")
            formality_num = self._formality_mapping.get(occasion.lower(), 2)

            # Pick garments for this slot
            outfit_ids, had_repeat = self._pick_outfit(
                day_idx=day_idx,
                formality_num=formality_num,
                garment_groups=garment_groups,
                last_used=last_used,
                utilisation=utilisation,
            )

            if had_repeat:
                repeated_days += 1

            slots.append(RotationSlot(
                day=day,
                session=session,
                occasion=occasion,
                garment_ids=outfit_ids,
                outfit_name=f"{day} {session.replace('_', ' ').title()} — {occasion.capitalize()}",
                formality_level=formality_num,
            ))

        total_days = len(slots)
        repeat_rate = repeated_days / total_days if total_days else 0.0

        coverage_pct = self._compute_coverage(slots, schedule)
        warnings = self._compute_warnings(garments, slots, repeat_rate)

        return WeeklyRotationPlan(
            week_label="Week 1",
            slots=slots,
            repeat_rate=round(repeat_rate, 3),
            coverage_pct=round(coverage_pct, 3),
            garment_utilisation=dict(utilisation),
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_outfit(
        self,
        day_idx: int,
        formality_num: int,
        garment_groups: Dict[int, List[Garment]],
        last_used: Dict[str, int],
        utilisation: Dict[str, int],
    ):
        """Pick 2-3 garments for one day slot.

        Returns
        -------
        (outfit_ids, had_repeat)
        """
        # Acceptable formality range ±1
        candidates: List[Garment] = []
        for fnum in [formality_num, formality_num - 1, formality_num + 1, formality_num - 2]:
            candidates.extend(garment_groups.get(fnum, []))
            if len(candidates) >= 6:
                break

        if not candidates:
            # Fallback: use any garment
            candidates = [g for group in garment_groups.values() for g in group]

        # Sort by: not on cooldown first, then least-used
        def sort_key(g: Garment):
            days_since = day_idx - last_used.get(g.id, -100)
            on_cooldown = days_since < self._cooldown_days
            return (int(on_cooldown), utilisation.get(g.id, 0))

        candidates_sorted = sorted(candidates, key=sort_key)

        # Pick top 3
        had_repeat = False
        outfit: List[Garment] = []
        for g in candidates_sorted[:3]:
            days_since = day_idx - last_used.get(g.id, -100)
            if days_since < self._cooldown_days:
                had_repeat = True
            outfit.append(g)
            last_used[g.id] = day_idx
            utilisation[g.id] += 1

        return [g.id for g in outfit], had_repeat

    @staticmethod
    def _group_by_formality(garments: List[Garment]) -> Dict[int, List[Garment]]:
        groups: Dict[int, List[Garment]] = defaultdict(list)
        for g in garments:
            try:
                fv = g.attributes.formality_level.value.lower()
                num = _FORMALITY_MAP.get(fv, 2)
            except Exception:
                num = 2
            groups[num].append(g)
        return groups

    @staticmethod
    def _build_default_schedule(
        work_days: int,
        weekend_days: int,
        work_occasion: str,
        weekend_occasion: str,
    ) -> List[Dict]:
        schedule = []
        for i in range(min(work_days, 5)):
            schedule.append({"day": _DAYS_OF_WEEK[i], "session": "full_day", "occasion": work_occasion})
        for i in range(min(weekend_days, 2)):
            schedule.append({"day": _DAYS_OF_WEEK[5 + i], "session": "full_day", "occasion": weekend_occasion})
        return schedule

    @staticmethod
    def _compute_coverage(slots: List[RotationSlot], schedule: List[Dict]) -> float:
        """Fraction of schedule occasions that have at least one garment assigned."""
        if not schedule:
            return 0.0
        filled = sum(1 for s in slots if s.garment_ids)
        return filled / len(schedule)

    @staticmethod
    def _compute_warnings(
        garments: List[Garment], slots: List[RotationSlot], repeat_rate: float
    ) -> List[str]:
        warnings = []
        if repeat_rate > 0.3:
            warnings.append(
                f"High repeat rate ({repeat_rate:.0%}). Consider expanding your wardrobe."
            )
        empty_slots = [s for s in slots if not s.garment_ids]
        if empty_slots:
            warnings.append(f"{len(empty_slots)} day(s) could not be filled.")
        if len(garments) < 7:
            warnings.append("Small wardrobe: outfit variety will be limited for a full week.")
        return warnings
