"""
Smart Removal Analyzer
======================

Replaces hard-threshold removal logic with a continuous **regret-risk** score
(0–1) that integrates:

1. Time-aware usage (recency decay)
2. Quality-weighted outfit impact
3. Validated replacement quality (not just same-category)
4. True versatility (context diversity, not popularity)
5. Usage frequency vs. age
6. User-goal personalisation (minimalist / maximize_options / style_upgrade)
7. Explainability (top reasons, what-you-lose, data gaps, confidence)

Design constraints
------------------
* ``Garment.history`` is the single source of truth — no outfit data is
  duplicated.
* No ML models — pure deterministic, interpretable logic.
* Produces stable results (small data changes don't flip verdicts).
* All parameters are configurable via smart_removal_config.json (no hardcoded thresholds).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.core import get_logger
from src.core.models import (
    DataGap,
    FormalityLevel,
    Garment,
    GarmentCategory,
    Occasion,
    ReplacementCandidate,
    ReplacementQuality,
    Season,
    SmartRemovalReason,
    SmartRemovalVerdict,
    UserContext,
    UserRemovalGoal,
    WearRecord,
)
from src.layer2_style.smart_removal_config import SmartRemovalProfile, SmartRemovalConfigLoader

logger = get_logger(__name__)

# ============================================================================
# Constants (kept for backwards compatibility; overridable via config)
# ============================================================================

FORMALITY_ORDER: Dict[FormalityLevel, int] = {
    FormalityLevel.VERY_CASUAL: 0,
    FormalityLevel.CASUAL: 1,
    FormalityLevel.SMART_CASUAL: 2,
    FormalityLevel.BUSINESS_CASUAL: 3,
    FormalityLevel.BUSINESS: 4,
    FormalityLevel.FORMAL: 5,
    FormalityLevel.BLACK_TIE: 6,
}


class SmartRemovalAnalyzer:
    """Produces an explainable, continuous regret-risk verdict for each garment.

    All scoring parameters are loaded from smart_removal_config.json and customizable
    per profile (default, minimalist_eco, fashion_addict, budget_conscious, work_professional, data_driven).
    """

    def __init__(self, profile_name: str = "default"):
        """Initialize with a named profile from config.

        Args:
            profile_name: Name of the profile to use (e.g., 'default', 'minimalist_eco').
                         Must exist in config/data/smart_removal_config.json.

        Raises:
            ValueError: If profile not found.
        """
        self.profile = SmartRemovalConfigLoader.get_profile(profile_name)
        self.profile_name = profile_name
        logger.info(f"SmartRemovalAnalyzer initialized with profile: {profile_name}")

        # Expose profile params as instance attributes for quick access
        self.recency_half_life = self.profile.recency_half_life_days
        self.formality_flexibility = self.profile.formality_flexibility
        self.replacement_threshold = self.profile.replacement_coverage_threshold
        self.neutral_colors = frozenset(self.profile.neutral_colors)
        self.occasion_importance = self.profile.occasion_importance
        self.goal_weights = {
            goal_name: weights.to_dict()
            for goal_name, weights in self.profile.goal_weights.items()
        }

    def analyze(
        self,
        garment_id: str,
        wardrobe: List[Garment],
        *,
        outfits: Optional[List] = None,
        context: Optional[UserContext] = None,
        user_goal: UserRemovalGoal = UserRemovalGoal.MAXIMIZE_OPTIONS,
    ) -> SmartRemovalVerdict:
        """Analyze a single garment and return a SmartRemovalVerdict."""
        target = self._find(garment_id, wardrobe)
        if target is None:
            return self._not_found(garment_id, user_goal)

        now = datetime.now(timezone.utc)
        data_gaps: List[DataGap] = []

        # ── 1. Recency score (higher = worn recently → more regret to remove) ──
        recency_score, rec_gaps = self._recency_score(target, now)
        data_gaps.extend(rec_gaps)

        # ── 2. Frequency score (time-adjusted) ──
        frequency_score, freq_gaps = self._frequency_score(target, now)
        data_gaps.extend(freq_gaps)

        # ── 3. Versatility (context diversity) ──
        versatility_score = self._versatility_score(target, wardrobe)

        # ── 4. Quality-weighted outfit impact ──
        impact_weighted, impact_by_occasion = self._outfit_impact(
            target, wardrobe, outfits
        )

        # ── 5. Replacement quality ──
        replacement = self._replacement_quality(target, wardrobe)

        # ── 6. Rating signal ──
        rating_score, rating_gaps = self._rating_score(target)
        data_gaps.extend(rating_gaps)

        # ── 7. Newness guard (recently acquired → careful) ──
        newness_score, new_gaps = self._newness_score(target, now)
        data_gaps.extend(new_gaps)

        # ── Aggregate into regret_risk ──
        raw_signals = {
            "recency":       recency_score,
            "frequency":     frequency_score,
            "versatility":   versatility_score,
            "outfit_impact": min(1.0, impact_weighted / 10.0),  # normalise
            "replacement":   1.0 - replacement.valid_count / max(1, len(wardrobe)),
            "rating":        rating_score,
            "newness":       newness_score,
        }

        # Get goal weights from profile config
        goal_name = user_goal.value.upper()
        if goal_name not in self.profile.goal_weights:
            goal_name = user_goal.value
        weights = self.profile.goal_weights[goal_name].to_dict()
        regret_risk = 0.0
        reasons: List[SmartRemovalReason] = []

        for signal_name, raw_value in raw_signals.items():
            w = weights.get(signal_name, 0.0)
            contribution = raw_value * w
            regret_risk += contribution

            direction = "supports_keeping" if raw_value > 0.5 else "supports_removal"
            reasons.append(SmartRemovalReason(
                signal_name=signal_name,
                direction=direction,
                weight=w,
                score_contribution=round(contribution, 4),
                explanation=self._explain_signal(signal_name, raw_value, target),
            ))

        regret_risk = round(max(0.0, min(1.0, regret_risk)), 3)

        # Sort reasons by absolute contribution (biggest first)
        reasons.sort(key=lambda r: abs(r.score_contribution), reverse=True)

        # ── Confidence ──
        confidence = self._compute_confidence(data_gaps, target)

        # ── Verdict ──
        verdict = self._derive_verdict(regret_risk, confidence)

        # ── Explainability ──
        what_you_lose = self._what_you_lose(target, impact_by_occasion, replacement)
        what_you_keep = self._what_you_keep(target, wardrobe, replacement)

        desc = self._desc(target)

        summary = self._build_summary(desc, verdict, regret_risk, confidence, reasons[:3])

        return SmartRemovalVerdict(
            garment_id=garment_id,
            garment_description=desc,
            garment_attributes=target.attributes,
            regret_risk=regret_risk,
            verdict=verdict,
            confidence=confidence,
            signals=raw_signals,
            reasons=reasons,
            outfits_affected_weighted=round(impact_weighted, 2),
            outfits_affected_by_occasion=impact_by_occasion,
            replacement=replacement,
            what_you_lose=what_you_lose,
            what_you_keep=what_you_keep,
            data_gaps=data_gaps,
            user_goal=user_goal,
            summary=summary,
        )

    def analyze_wardrobe(
        self,
        wardrobe: List[Garment],
        *,
        outfits: Optional[List] = None,
        context: Optional[UserContext] = None,
        user_goal: UserRemovalGoal = UserRemovalGoal.MAXIMIZE_OPTIONS,
    ) -> List[SmartRemovalVerdict]:
        """Analyze every garment in the wardrobe and return sorted verdicts.

        Returns list sorted by regret_risk ascending (safest to remove first).
        """
        results = []
        for garment in wardrobe:
            if garment.history.is_retired:
                continue
            v = self.analyze(
                garment.id, wardrobe,
                outfits=outfits, context=context, user_goal=user_goal,
            )
            results.append(v)

        results.sort(key=lambda v: v.regret_risk)
        return results

    # ------------------------------------------------------------------ #
    # Signal Computers                                                    #
    # ------------------------------------------------------------------ #

    def _recency_score(
        self, garment: Garment, now: datetime,
    ) -> Tuple[float, List[DataGap]]:
        """Exponential-decay recency: 1.0 = just worn, 0.0 = very old / never.

        Uses a half-life model so that a garment worn 60 days ago has
        score ≈ 0.5, worn 120 days ago ≈ 0.25, etc.
        """
        gaps: List[DataGap] = []
        records = garment.history.wear_records
        if not records:
            gaps.append(DataGap(field="wear_records", description="No wear history available"))
            return 0.0, gaps

        last = max(r.worn_at for r in records)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        days = max(0, (now - last).days)
        score = math.pow(0.5, days / self.recency_half_life)
        return round(score, 3), gaps

    def _frequency_score(
        self, garment: Garment, now: datetime,
    ) -> Tuple[float, List[DataGap]]:
        """Time-adjusted frequency: recent wears count more than old ones.

        Each wear gets weight ``0.5 ^ (days_ago / half_life)`` and the total
        is normalised into [0, 1] with a soft cap.
        """
        gaps: List[DataGap] = []
        records = garment.history.wear_records
        if not records:
            return 0.0, gaps

        weighted_total = 0.0
        for r in records:
            worn = r.worn_at
            if worn.tzinfo is None:
                worn = worn.replace(tzinfo=timezone.utc)
            days = max(0, (now - worn).days)
            weighted_total += math.pow(0.5, days / self.profile.frequency_soft_cap.half_life_days)

        # Soft-cap from profile
        max_equiv_wears = self.profile.frequency_soft_cap.max_recent_equivalent_wears
        score = min(1.0, weighted_total / max_equiv_wears)
        return round(score, 3), gaps

    def _versatility_score(self, garment: Garment, wardrobe: List[Garment]) -> float:
        """True versatility = diversity of contexts covered.

        Three sub-components:
          1. Season breadth (how many of 4 seasons)
          2. Occasion breadth (how many distinct occasion-levels reachable by formality)
          3. Category pairing breadth (how many other categories it matches)
        """
        attrs = garment.attributes

        # Season breadth (0-1)
        seasons = set(attrs.season_suitable or [])
        season_breadth = len(seasons) / 4.0

        # Occasion breadth via formality mapping
        formality = attrs.formality_level or FormalityLevel.CASUAL
        f_val = FORMALITY_ORDER.get(formality, 1)
        reachable_occasions = set()
        for occ_name, imp in self.occasion_importance.items():
            try:
                occ_enum = Occasion(occ_name)
            except ValueError:
                continue
            occ_formality_val = self._occasion_formality_level(occ_name)
            if abs(f_val - occ_formality_val) <= self.formality_flexibility:
                reachable_occasions.add(occ_name)
        occasion_breadth = min(1.0, len(reachable_occasions) / 8.0)

        # Category pairing breadth
        wardrobe_cats = {g.attributes.category for g in wardrobe if g.id != garment.id}
        compatible_cats = set()
        for cat in wardrobe_cats:
            if cat != attrs.category:
                compatible_cats.add(cat)
        category_breadth = min(1.0, len(compatible_cats) / 4.0)

        # Apply profile weights
        weights = self.profile.versatility
        score = (
            weights.season_weight * season_breadth
            + weights.occasion_weight * occasion_breadth
            + weights.category_weight * category_breadth
        )
        return round(min(1.0, score), 3)

    def _outfit_impact(
        self,
        target: Garment,
        wardrobe: List[Garment],
        outfits: Optional[List] = None,
    ) -> Tuple[float, Dict[str, int]]:
        """Quality-weighted outfit impact.

        If real ``Outfit`` objects are provided, uses their ``overall_score``
        and occasion data.  Otherwise falls back to a combinatorial estimate
        weighted by occasion importance.
        """
        target_cat = target.attributes.category
        impact_by_occasion: Dict[str, int] = {}

        if outfits:
            # Use real outfit data
            weighted_impact = 0.0
            for outfit in outfits:
                garment_ids = getattr(outfit, "garment_ids", [])
                if not garment_ids:
                    items = getattr(outfit, "items", [])
                    garment_ids = [
                        getattr(item, "garment_id", None) or
                        getattr(getattr(item, "garment", None), "id", None)
                        for item in items
                    ]
                if target.id in garment_ids:
                    score = getattr(outfit, "overall_score", 0.5)
                    weighted_impact += score

                    # Occasion breakdown
                    occ = None
                    history = getattr(outfit, "history", None)
                    if history:
                        creation = getattr(history, "creation", None)
                        if creation:
                            occ = getattr(creation, "occasion_intent", None)
                    occ_key = occ.value if occ else "unknown"
                    impact_by_occasion[occ_key] = impact_by_occasion.get(occ_key, 0) + 1

            return weighted_impact, impact_by_occasion

        # Fallback: combinatorial estimate
        other_cats: Dict[str, int] = {}
        for g in wardrobe:
            if g.id != target.id:
                other_cats.setdefault(g.attributes.category.value, 0)
                other_cats[g.attributes.category.value] += 1

        complement_counts = [
            c for cat, c in other_cats.items()
            if cat != target_cat.value
        ]
        complement_counts.sort(reverse=True)

        raw_combos = 1
        for c in complement_counts[:2]:
            raw_combos *= max(1, c)
        raw_combos = min(raw_combos, 50)

        # Weight by occasion importance of the garment's formality
        formality = target.attributes.formality_level or FormalityLevel.CASUAL
        f_val = FORMALITY_ORDER.get(formality, 1)
        best_occ_importance = 0.3
        for occ_name, imp in self.occasion_importance.items():
            occ_f_val = self._occasion_formality_level(occ_name)
            if abs(f_val - occ_f_val) <= 1:
                best_occ_importance = max(best_occ_importance, imp)
                impact_by_occasion[occ_name] = impact_by_occasion.get(occ_name, 0) + raw_combos

        weighted = raw_combos * best_occ_importance
        return round(weighted, 2), impact_by_occasion

    def _replacement_quality(
        self, target: Garment, wardrobe: List[Garment],
    ) -> ReplacementQuality:
        """Validate replacements by season, occasion, formality, color-role."""
        target_attrs = target.attributes
        target_seasons = set(target_attrs.season_suitable or [])
        target_formality = target_attrs.formality_level or FormalityLevel.CASUAL
        target_f_val = FORMALITY_ORDER.get(target_formality, 1)
        target_is_neutral = (
            target_attrs.color.primary.lower() in self.neutral_colors
            if target_attrs.color else False
        )

        candidates: List[ReplacementCandidate] = []

        for g in wardrobe:
            if g.id == target.id:
                continue
            if g.attributes.category != target_attrs.category:
                continue

            g_attrs = g.attributes
            g_seasons = set(g_attrs.season_suitable or [])
            g_formality = g_attrs.formality_level or FormalityLevel.CASUAL
            g_f_val = FORMALITY_ORDER.get(g_formality, 1)
            g_is_neutral = (
                g_attrs.color.primary.lower() in self.neutral_colors
                if g_attrs.color else False
            )

            # Season overlap
            matched_seasons = list(target_seasons & g_seasons)
            season_coverage = len(matched_seasons) / max(1, len(target_seasons))

            # Formality compatibility
            formality_ok = abs(target_f_val - g_f_val) <= self.formality_flexibility

            # Color role match
            color_role_ok = (target_is_neutral == g_is_neutral)

            # Occasion overlap
            matched_occasions: List[str] = []
            for occ_name in self.occasion_importance:
                occ_f = self._occasion_formality_level(occ_name)
                target_reach = abs(target_f_val - occ_f) <= self.formality_flexibility
                g_reach = abs(g_f_val - occ_f) <= self.formality_flexibility
                if target_reach and g_reach:
                    matched_occasions.append(occ_name)

            # Coverage score
            rq_weights = self.profile.replacement_quality_weights
            coverage = (
                rq_weights.season_coverage * season_coverage
                + rq_weights.formality_compatibility * (1.0 if formality_ok else 0.3)
                + rq_weights.color_role_match * (1.0 if color_role_ok else 0.5)
                + rq_weights.occasion_coverage * min(1.0, len(matched_occasions) / 4.0)
            )

            reason = None
            if not formality_ok:
                reason = f"Formality mismatch ({g_formality.value} vs {target_formality.value})"
            elif season_coverage < 0.5:
                reason = f"Covers only {len(matched_seasons)}/{len(target_seasons)} seasons"
            elif not color_role_ok:
                reason = "Different color role (neutral vs accent)"

            candidates.append(ReplacementCandidate(
                garment_id=g.id,
                garment_description=self._desc(g),
                coverage_match=round(coverage, 2),
                matched_seasons=matched_seasons,
                matched_occasions=matched_occasions,
                formality_compatible=formality_ok,
                color_role_match=color_role_ok,
                reason_not_perfect=reason,
            ))

        # Sort by coverage descending
        candidates.sort(key=lambda c: c.coverage_match, reverse=True)

        # Valid = coverage >= 0.6
        valid = [c for c in candidates if c.coverage_match >= 0.6]

        # Coverage gaps
        coverage_gaps: List[str] = []
        if valid:
            best = valid[0]
            if not best.formality_compatible:
                coverage_gaps.append(f"No formality-compatible replacement for {target_formality.value}")
            uncovered_seasons = target_seasons - set(best.matched_seasons)
            for s in uncovered_seasons:
                coverage_gaps.append(f"Season '{s.value}' not covered by best replacement")
        else:
            coverage_gaps.append("No replacement covers ≥60% of this garment's role")

        quality_level = (
            "high" if len(valid) >= 3
            else "medium" if len(valid) >= 1
            else "low" if candidates
            else "none"
        )

        return ReplacementQuality(
            valid_count=len(valid),
            quality_level=quality_level,
            best_candidate=valid[0] if valid else (candidates[0] if candidates else None),
            all_candidates=candidates[:5],
            coverage_gaps=coverage_gaps,
        )

    def _rating_score(self, garment: Garment) -> Tuple[float, List[DataGap]]:
        """Average user rating → [0, 1].  No rating → 0.5 (neutral)."""
        gaps: List[DataGap] = []
        avg = garment.history.average_rating
        if avg is None:
            gaps.append(DataGap(field="user_rating", description="No user ratings recorded"))
            return 0.5, gaps
        return round((avg - 1.0) / 4.0, 3), gaps   # map 1-5 → 0-1

    def _newness_score(
        self, garment: Garment, now: datetime,
    ) -> Tuple[float, List[DataGap]]:
        """Protect recently acquired items from premature removal.

        Returns a high score (→ keep) for items acquired within 30 days,
        decaying to 0 after ~120 days.
        """
        gaps: List[DataGap] = []
        acquired = garment.history.acquired_at
        if acquired is None:
            # Fall back to garment.created_at
            acquired = garment.created_at
            if acquired is None:
                gaps.append(DataGap(field="acquired_at", description="No acquisition date available"))
                return 0.0, gaps

        if acquired.tzinfo is None:
            acquired = acquired.replace(tzinfo=timezone.utc)
        days_owned = max(0, (now - acquired).days)
        # 0-30 days → score ~1, 60 days → ~0.5, 120 days → ~0.25
        score = math.pow(0.5, days_owned / 60.0)
        return round(score, 3), gaps

    # ------------------------------------------------------------------ #
    # Verdict & Confidence                                                #
    # ------------------------------------------------------------------ #

    def _derive_verdict(self, regret_risk: float, confidence: float) -> str:
        """Map regret_risk to verdict, with low-confidence safety."""
        # If confidence is very low, be cautious: shift towards CONSIDER
        effective = regret_risk
        if confidence < self.profile.confidence_strictness.minimum_confidence_for_action:
            effective = max(regret_risk, self.profile.removal_aggressiveness.consider_threshold - 0.1)

        thresholds = self.profile.removal_aggressiveness
        if effective < thresholds.donate_threshold:
            return "DONATE"
        elif effective < thresholds.safe_to_remove_threshold:
            return "SAFE_TO_REMOVE"
        elif effective < thresholds.consider_threshold:
            return "CONSIDER"
        else:
            return "KEEP"

    def _compute_confidence(self, data_gaps: List[DataGap], garment: Garment) -> float:
        """Confidence based on data completeness."""
        base = 1.0
        # Each data gap reduces confidence
        base -= 0.15 * len(data_gaps)
        # Bonus for having many wear records (more data = more confident)
        n_wears = garment.history.total_wears
        if n_wears >= 10:
            base += 0.05
        elif n_wears == 0:
            base -= 0.1
        return round(max(0.1, min(1.0, base)), 2)

    # ------------------------------------------------------------------ #
    # Explainability                                                      #
    # ------------------------------------------------------------------ #

    def _what_you_lose(
        self,
        target: Garment,
        impact_by_occasion: Dict[str, int],
        replacement: ReplacementQuality,
    ) -> List[str]:
        lines: List[str] = []
        total_outfits = sum(impact_by_occasion.values())
        if total_outfits:
            occ_parts = [f"{k}: {v}" for k, v in sorted(impact_by_occasion.items(), key=lambda x: -x[1])[:3]]
            lines.append(f"~{total_outfits} outfit(s) affected ({', '.join(occ_parts)})")

        if replacement.coverage_gaps:
            for gap in replacement.coverage_gaps[:2]:
                lines.append(gap)

        attrs = target.attributes
        if attrs.color and attrs.color.primary.lower() in self.neutral_colors:
            lines.append(f"Losing a neutral ({attrs.color.primary}) — harder to replace in outfit building")

        seasons = attrs.season_suitable or []
        if len(seasons) >= 3:
            lines.append(f"Multi-season piece covering {', '.join(s.value for s in seasons)}")

        return lines

    def _what_you_keep(
        self,
        target: Garment,
        wardrobe: List[Garment],
        replacement: ReplacementQuality,
    ) -> List[str]:
        lines: List[str] = []
        same_cat = [
            g for g in wardrobe
            if g.attributes.category == target.attributes.category and g.id != target.id
        ]
        lines.append(f"{len(same_cat)} item(s) still in '{target.attributes.category.value}' category")

        if replacement.valid_count:
            best = replacement.best_candidate
            if best:
                lines.append(
                    f"Best replacement: {best.garment_description} "
                    f"({best.coverage_match:.0%} coverage)"
                )

        if replacement.valid_count >= 2:
            lines.append(f"{replacement.valid_count} validated replacements available")

        return lines

    def _explain_signal(self, name: str, value: float, garment: Garment) -> str:
        """One-sentence explanation for a signal."""
        if name == "recency":
            if value > 0.7:
                return "Worn recently — removing it may feel like a loss."
            elif value > 0.3:
                return "Not worn for a while, but not forgotten."
            else:
                return "Not worn for a long time — low recency attachment."
        elif name == "frequency":
            if value > 0.5:
                return f"Worn regularly ({garment.history.total_wears} times with recent emphasis)."
            else:
                return f"Rarely worn ({garment.history.total_wears} total wears)."
        elif name == "versatility":
            if value > 0.6:
                return "High versatility — works across many seasons, occasions, and pairings."
            else:
                return "Limited versatility — suited for few contexts."
        elif name == "outfit_impact":
            return f"Removing this garment would impact outfits (weighted score: {value:.2f})."
        elif name == "replacement":
            if value > 0.7:
                return "Few or no valid replacements — this role would be left unfilled."
            else:
                return "Good replacements exist — the gap would be filled."
        elif name == "rating":
            avg = garment.history.average_rating
            if avg is not None:
                return f"Average user rating: {avg}/5."
            return "No user rating available — neutral impact."
        elif name == "newness":
            if value > 0.5:
                return "Recently acquired — give it more time before deciding."
            return "Owned for a while — fair to evaluate for removal."
        return f"Signal '{name}' has value {value:.2f}."

    def _build_summary(
        self,
        desc: str,
        verdict: str,
        regret_risk: float,
        confidence: float,
        top_reasons: List[SmartRemovalReason],
    ) -> str:
        parts = [f"'{desc}' → {verdict} (regret risk: {regret_risk:.0%}, confidence: {confidence:.0%})."]
        if top_reasons:
            parts.append("Top reasons:")
            for r in top_reasons[:3]:
                arrow = "↑" if r.direction == "supports_keeping" else "↓"
                parts.append(f"  {arrow} {r.explanation}")
        return " ".join(parts)

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find(garment_id: str, wardrobe: List[Garment]) -> Optional[Garment]:
        for g in wardrobe:
            if g.id == garment_id:
                return g
        return None

    @staticmethod
    def _desc(garment: Garment) -> str:
        attrs = garment.attributes
        parts = []
        if attrs.color and attrs.color.primary:
            parts.append(attrs.color.primary)
        if attrs.subcategory:
            parts.append(attrs.subcategory)
        elif attrs.category:
            parts.append(attrs.category.value)
        if attrs.material and attrs.material.primary:
            parts.append(attrs.material.primary)
        return " ".join(parts) if parts else f"Garment {garment.id}"

    @staticmethod
    def _not_found(garment_id: str, user_goal: UserRemovalGoal) -> SmartRemovalVerdict:
        return SmartRemovalVerdict(
            garment_id=garment_id,
            garment_description="Unknown garment",
            regret_risk=0.0,
            verdict="SAFE_TO_REMOVE",
            confidence=0.1,
            data_gaps=[DataGap(field="garment", description="Garment not found in wardrobe")],
            user_goal=user_goal,
            summary=f"Garment '{garment_id}' not found in wardrobe.",
        )

    @staticmethod
    def _occasion_formality_level(occ_name: str) -> int:
        """Map occasion name → approximate formality int."""
        mapping = {
            "formal": 5, "wedding": 5, "black_tie": 6,
            "interview": 4, "business": 4, "work": 3,
            "cocktail": 4, "evening": 4, "event": 3,
            "date": 2, "travel": 1, "daily_wear": 1,
            "casual": 1, "weekend": 1, "outdoor": 1,
            "beach": 0, "gym": 0, "sport": 0,
        }
        return mapping.get(occ_name, 1)
