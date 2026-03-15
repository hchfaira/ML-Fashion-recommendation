"""SeasonalAuditEngine — classifies each wardrobe piece as ready / adaptable / store.

Classification logic
--------------------
Each garment gets a *readiness score* in [0, 1]:

    score = material_score * 0.40
          + color_score    * 0.25
          + formality_score* 0.20
          + declared_season* 0.15

Thresholds:
    score >= 0.65  → READY
    score >= 0.35  → ADAPTABLE
    else           → STORE

The engine is config-driven (``season_transition_data.json``) so thresholds
and material/color lists can be tuned without touching code.

Optimisation notes
------------------
* Single-pass over garments — O(n) overall.
* All lookups are dict-based O(1) per garment.
* Config is loaded once in ``__init__`` and cached.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from config import get_config
from src.core.models import Garment, Season
from src.core.travel_models import (
    GarmentSeasonAudit,
    SeasonAuditResult,
    SeasonCoverage,
    SeasonReadiness,
)

logger = logging.getLogger(__name__)

# Score thresholds
_READY_THRESHOLD = 0.65
_ADAPTABLE_THRESHOLD = 0.35


class SeasonalAuditEngine:
    """Classify each garment against a target season.

    Parameters
    ----------
    config_path:
        Optional override path for ``season_transition_data.json``.

    Usage
    -----
    ::

        engine = SeasonalAuditEngine()
        result = engine.audit(garments, target_season="fall")
    """

    def __init__(self, config_path=None):
        cfg = get_config()
        raw = cfg.get_data("season_transition_data", default={})
        self._season_readiness: Dict = raw.get("season_readiness", {})
        self._adaptability: Dict = raw.get("adaptability_score", {})
        self._transition_pairs: Dict = raw.get("season_pairs", {})
        logger.debug("SeasonalAuditEngine loaded config for seasons: %s", list(self._season_readiness.keys()))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit(
        self,
        garments: List[Garment],
        target_season: str,
    ) -> SeasonAuditResult:
        """Audit a wardrobe against *target_season*.

        Parameters
        ----------
        garments:
            Full wardrobe as ``List[Garment]``.
        target_season:
            One of ``spring``, ``summer``, ``fall``, ``winter``.

        Returns
        -------
        SeasonAuditResult
        """
        target = target_season.lower()
        rules = self._season_readiness.get(target, {})

        audits: List[GarmentSeasonAudit] = []
        for garment in garments:
            audit = self._audit_garment(garment, target, rules)
            audits.append(audit)

        ready = sum(1 for a in audits if a.readiness == SeasonReadiness.READY)
        adaptable = sum(1 for a in audits if a.readiness == SeasonReadiness.ADAPTABLE)
        store = sum(1 for a in audits if a.readiness == SeasonReadiness.STORE)
        total = len(audits)

        overall_pct = ((ready + 0.5 * adaptable) / total * 100) if total > 0 else 0.0

        coverage = self._compute_coverage(audits, rules)
        gaps = self._identify_gaps(coverage)

        return SeasonAuditResult(
            target_season=target,
            garment_audits=audits,
            ready_count=ready,
            adaptable_count=adaptable,
            store_count=store,
            coverage_by_category=coverage,
            overall_readiness_pct=round(overall_pct, 1),
            top_gaps=gaps,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _audit_garment(
        self, garment: Garment, target_season: str, rules: Dict
    ) -> GarmentSeasonAudit:
        attrs = garment.attributes
        reasons: List[str] = []
        score = 0.0

        # 1. Material score (40%)
        mat_score, mat_reasons = self._score_material(attrs, rules)
        score += mat_score * 0.40
        reasons.extend(mat_reasons)

        # 2. Color score (25%)
        col_score, col_reasons = self._score_color(attrs, rules)
        score += col_score * 0.25
        reasons.extend(col_reasons)

        # 3. Formality score (20%)
        form_score, form_reasons = self._score_formality(attrs, rules)
        score += form_score * 0.20
        reasons.extend(form_reasons)

        # 4. Declared season suitability (15%)
        sea_score, sea_reasons = self._score_declared_season(attrs, target_season)
        score += sea_score * 0.15
        reasons.extend(sea_reasons)

        # Apply multi-season versatility bonus
        if self._is_multi_season(attrs):
            score = min(1.0, score + 0.10)
            reasons.append("Multi-season versatile piece (+0.1 bonus)")

        score = max(0.0, min(1.0, score))

        readiness = (
            SeasonReadiness.READY
            if score >= _READY_THRESHOLD
            else SeasonReadiness.ADAPTABLE
            if score >= _ADAPTABLE_THRESHOLD
            else SeasonReadiness.STORE
        )

        cat = attrs.category.value if hasattr(attrs, "category") else "unknown"
        primary_color = ""
        if hasattr(attrs, "color") and attrs.color:
            primary_color = getattr(attrs.color, "primary", "") or ""

        desc = f"{primary_color} {cat}".strip()

        return GarmentSeasonAudit(
            garment_id=garment.id,
            description=desc,
            category=cat,
            readiness=readiness,
            score=round(score, 3),
            reasons=reasons[:6],  # cap to keep output lean
        )

    # --- material ---

    def _score_material(self, attrs, rules: Dict):
        boost = [m.lower() for m in rules.get("boost_materials", [])]
        penalise = [m.lower() for m in rules.get("penalise_materials", [])]
        reasons = []

        mat = ""
        if hasattr(attrs, "material") and attrs.material:
            mat = (getattr(attrs.material, "primary", "") or "").lower()

        if not mat:
            return 0.6, ["Material unknown — neutral score"]

        for bm in boost:
            if bm in mat:
                reasons.append(f"Material '{mat}' is ideal for this season")
                return 1.0, reasons
        for pm in penalise:
            if pm in mat:
                reasons.append(f"Material '{mat}' is unsuitable for this season")
                return 0.1, reasons

        reasons.append(f"Material '{mat}' is neutral for this season")
        return 0.6, reasons

    # --- color ---

    def _score_color(self, attrs, rules: Dict):
        boost_colors = [c.lower() for c in rules.get("boost_colors", [])]
        penalise_colors = [c.lower() for c in rules.get("penalise_colors", [])]
        reasons = []

        primary = ""
        if hasattr(attrs, "color") and attrs.color:
            primary = (getattr(attrs.color, "primary", "") or "").lower()

        if not primary:
            return 0.6, ["Color unknown — neutral score"]

        # Check against neutral adaptable colors
        neutral_colors = [c.lower() for c in self._adaptability.get("neutral_colors", [])]
        if any(nc in primary for nc in neutral_colors):
            reasons.append(f"Neutral color '{primary}' works any season")
            return 0.9, reasons

        for bc in boost_colors:
            if bc in primary:
                reasons.append(f"Color '{primary}' suits {rules.get('_season', 'this')} season well")
                return 1.0, reasons
        for pc in penalise_colors:
            if pc in primary:
                reasons.append(f"Color '{primary}' may clash this season")
                return 0.2, reasons

        return 0.6, [f"Color '{primary}' is neutral for this season"]

    # --- formality ---

    def _score_formality(self, attrs, rules: Dict):
        boost_formality = [f.lower() for f in rules.get("boost_formality", [])]
        reasons = []

        fv = ""
        if hasattr(attrs, "formality_level") and attrs.formality_level:
            fv = attrs.formality_level.value.lower()

        if not fv or not boost_formality:
            return 0.7, []

        if any(bf in fv for bf in boost_formality):
            reasons.append(f"Formality '{fv}' matches this season's typical use")
            return 1.0, reasons

        reasons.append(f"Formality '{fv}' less common this season")
        return 0.5, reasons

    # --- declared season ---

    def _score_declared_season(self, attrs, target_season: str):
        reasons = []
        seasons: List = []
        if hasattr(attrs, "season_suitable"):
            seasons = attrs.season_suitable or []
        elif hasattr(attrs, "seasonality") and attrs.seasonality:
            seasons = getattr(attrs.seasonality, "suitable_seasons", []) or []

        if not seasons:
            return 0.5, ["No season metadata — neutral score"]

        season_values = [
            (s.value if hasattr(s, "value") else str(s)).lower() for s in seasons
        ]
        if target_season.lower() in season_values:
            reasons.append(f"Garment is tagged for {target_season}")
            return 1.0, reasons

        reasons.append(f"Garment not tagged for {target_season} (tagged: {', '.join(season_values)})")
        return 0.0, reasons

    # --- versatility ---

    def _is_multi_season(self, attrs) -> bool:
        multi_mats = [m.lower() for m in self._adaptability.get("multi_season_materials", [])]
        neutral_cols = [c.lower() for c in self._adaptability.get("neutral_colors", [])]
        versatile_forms = [f.lower() for f in self._adaptability.get("versatile_formality", [])]

        mat_ok = False
        if hasattr(attrs, "material") and attrs.material:
            mat_str = (getattr(attrs.material, "primary", "") or "").lower()
            mat_ok = any(m in mat_str for m in multi_mats)

        col_ok = False
        if hasattr(attrs, "color") and attrs.color:
            col_str = (getattr(attrs.color, "primary", "") or "").lower()
            col_ok = any(c in col_str for c in neutral_cols)

        form_ok = False
        if hasattr(attrs, "formality_level") and attrs.formality_level:
            fv = attrs.formality_level.value.lower()
            form_ok = any(f in fv for f in versatile_forms)

        return (mat_ok or col_ok) and form_ok

    # --- coverage ---

    def _compute_coverage(
        self, audits: List[GarmentSeasonAudit], rules: Dict
    ) -> List[SeasonCoverage]:
        required: Dict[str, int] = rules.get("required_coverage", {})
        available: Dict[str, int] = {}
        for a in audits:
            if a.readiness != SeasonReadiness.STORE:
                available[a.category] = available.get(a.category, 0) + 1

        result = []
        for cat, req in required.items():
            avail = available.get(cat, 0)
            result.append(
                SeasonCoverage(
                    category=cat,
                    required=req,
                    available=avail,
                    gap=max(0, req - avail),
                )
            )
        return result

    @staticmethod
    def _identify_gaps(coverage: List[SeasonCoverage]) -> List[str]:
        return [
            f"Need {c.gap} more {c.category}(s) (have {c.available}, need {c.required})"
            for c in coverage
            if c.gap > 0
        ]
