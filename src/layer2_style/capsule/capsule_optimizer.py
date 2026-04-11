"""
Capsule Optimizer — Greedy + 2-opt Local Search
================================================
Selects the best N garments from a wardrobe that maximize a composite
objective function: outfit combinations × occasion coverage × color
cohesion × practicality.

Algorithm
---------
1. **Hard filter** — drop garments incompatible with the target context
   (wrong season, wrong formality band, low confidence).
2. **Category seed** — ensure at least one piece per required category
   (top, bottom, shoes …) so the capsule is structurally valid.
3. **Greedy fill** — iteratively add the piece that maximises marginal
   gain in outfit combinations until we reach N.
4. **2-opt local search** — for each selected piece, try swapping it
   with every non-selected candidate; accept the swap when the total
   capsule score improves.  Repeat until convergence (no improving swap)
   or max iterations reached.

Complexity
----------
* Greedy phase:   O(N × C)  with C = |candidates|
* 2-opt phase:    O(I × N × C) with I = max_iterations (default 3)
* Typical wall-time: < 50 ms for 100-garment wardrobe

No external dependencies beyond NumPy (already a project dep).
All scoring delegates to the capsule_service helpers — no duplication.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "capsule" / "capsule_config.json"


def _load_config() -> Dict[str, Any]:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_CFG = _load_config()
_SCORE_WEIGHTS: Dict[str, float] = _CFG.get("score_weights", {
    "valid_combinations": 0.40,
    "occasion_coverage": 0.25,
    "color_cohesion": 0.20,
    "practicality": 0.15,
})
_CONFIDENCE_MIN: float = _CFG.get("confidence_min", 0.60)
_CLIMATE_SEASONS: Dict[str, List[str]] = _CFG.get("climate_seasons", {
    "warm": ["summer", "spring"],
    "cold": ["winter", "autumn"],
    "mixed": ["summer", "spring", "winter", "autumn"],
})
_FORMALITY_TIERS: Dict[str, int] = _CFG.get("formality_tiers", {
    "casual": 0, "smart_casual": 1, "business": 2, "formal": 3, "evening": 3,
})
_MATERIAL_PRACTICALITY: Dict[str, float] = _CFG.get("material_practicality", {"default": 0.65})
_CATEGORY_ROLES: Dict[str, str] = _CFG.get("category_roles", {
    "top": "anchor", "dress": "anchor", "bottom": "anchor",
    "outerwear": "layer", "shoes": "shoes", "accessory": "accent",
})
_CAT_VERSATILITY: Dict[str, float] = _CFG.get("category_versatility_base", {
    "top": 1.0, "dress": 0.95, "bottom": 0.90, "outerwear": 0.70,
    "shoes": 0.60, "accessory": 0.40, "default": 0.50,
})
_OUTFIT_TARGETS = _CFG.get("outfit_targets", {
    "small": {"max_pieces_le": 5, "target_outfits": 6},
    "medium": {"max_pieces_le": 7, "target_outfits": 10},
    "large": {"max_pieces_le": 99, "target_outfits": 15},
})
_SEEDING_ORDER: List[str] = _CFG.get("seeding_order", [
    "top", "bottom", "shoes", "dress", "outerwear", "accessory",
])
_OCCASION_FORMALITIES: Dict[str, List[str]] = _CFG.get("occasion_formalities", {})
_OCCASION_TEMPLATES: Dict[str, Dict[str, int]] = _CFG.get("occasion_templates", {})
_MISSING_LABELS: Dict[str, str] = _CFG.get("missing_piece_labels", {})
_NEUTRAL_FAMILIES = [
    (e["name"], tuple(e["rgb"]))
    for e in _CFG.get("neutral_families", [
        {"name": "Black", "rgb": [0, 0, 0]},
        {"name": "White", "rgb": [255, 255, 255]},
    ])
]
_MAX_MISSING: int = _CFG.get("max_missing_pieces", 5)
_MAX_2OPT_ITERS: int = 3


# ─────────────────────────────────────────────────────────────
# Garment helpers (mirror capsule_service logic, no duplication)
# ─────────────────────────────────────────────────────────────

def _garment_cat(g: Dict[str, Any]) -> str:
    attrs = g.get("attributes") or {}
    cat = attrs.get("category", "")
    if isinstance(cat, dict):
        return cat.get("value", "")
    return str(cat)


def _form_tier(formality: Optional[str]) -> int:
    if not formality:
        return 0
    return _FORMALITY_TIERS.get(formality.lower().replace(" ", "_"), 1)


def _mat_score(material: Optional[str]) -> float:
    if not material:
        return _MATERIAL_PRACTICALITY.get("default", 0.65)
    m = material.lower().strip()
    for key, score in _MATERIAL_PRACTICALITY.items():
        if key == "default":
            continue
        if key in m:
            return score
    return _MATERIAL_PRACTICALITY.get("default", 0.65)


def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.lstrip("#")
    if len(h) != 6:
        return (200, 190, 185)
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except Exception:
        return (200, 190, 185)


def _nearest_neutral(hex_str: str) -> str:
    r, g, b = _hex_to_rgb(hex_str)
    best_name, best_dist = "Neutral", float("inf")
    for name, (cr, cg, cb) in _NEUTRAL_FAMILIES:
        dist = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def _outfit_target(max_pieces: int) -> int:
    for tier in ("small", "medium", "large"):
        t = _OUTFIT_TARGETS.get(tier, {})
        if max_pieces <= t.get("max_pieces_le", 99):
            return t.get("target_outfits", 15)
    return 15


def _versatility_score(g: Dict[str, Any]) -> float:
    attrs = g.get("attributes") or {}
    cat = _garment_cat(g)
    base = _CAT_VERSATILITY.get(cat, _CAT_VERSATILITY.get("default", 0.50))
    seasons_bonus = len(attrs.get("seasons") or []) / 4 * 0.1
    return base + seasons_bonus + _mat_score(attrs.get("material")) * 0.1


# ─────────────────────────────────────────────────────────────
# Outfit counting
# ─────────────────────────────────────────────────────────────

def _count_valid_outfits(garments: List[Dict], occasion_types: List[str]) -> int:
    tops = [g for g in garments if _garment_cat(g) == "top"]
    bottoms = [g for g in garments if _garment_cat(g) == "bottom"]
    dresses = [g for g in garments if _garment_cat(g) == "dress"]
    outerw = [g for g in garments if _garment_cat(g) == "outerwear"]
    shoes_lst = [g for g in garments if _garment_cat(g) == "shoes"]

    target_tiers = {_form_tier(f) for f in occasion_types}
    accepted_tiers: Set[int] = set()
    for t in target_tiers:
        accepted_tiers |= {max(0, t - 1), t, min(3, t + 1)}

    shoe_mult = max(1, len(shoes_lst))
    count = 0

    for top in tops:
        for bottom in bottoms:
            top_tier = _form_tier((top.get("attributes") or {}).get("formality"))
            bottom_tier = _form_tier((bottom.get("attributes") or {}).get("formality"))
            if abs(top_tier - bottom_tier) > 1:
                continue
            avg_tier = (top_tier + bottom_tier) // 2
            if avg_tier not in accepted_tiers:
                continue
            count += (1 + len(outerw)) * shoe_mult

    for dress in dresses:
        d_tier = _form_tier((dress.get("attributes") or {}).get("formality"))
        if d_tier not in accepted_tiers:
            continue
        count += (1 + len(outerw)) * shoe_mult

    return count


# ─────────────────────────────────────────────────────────────
# Capsule scoring
# ─────────────────────────────────────────────────────────────

def score_capsule(
    garments: List[Dict],
    occasion_types: List[str],
    max_pieces: int,
) -> Dict[str, Any]:
    """Score a candidate capsule.  Returns a dict with total_score and sub-metrics."""
    n = len(garments)
    if n == 0:
        return {
            "total_score": 0.0,
            "valid_combinations": 0,
            "color_palette": [],
            "color_names": [],
            "occasion_coverage": {},
            "practicality_score": 0.0,
        }

    outfit_count = _count_valid_outfits(garments, occasion_types)
    n_target = _outfit_target(max_pieces)
    comb_norm = min(1.0, outfit_count / n_target)

    # Occasion coverage
    occ_coverage: Dict[str, float] = {}
    for occ in occasion_types:
        occ_tier = _form_tier(occ)
        occ_items = [
            g for g in garments
            if abs(_form_tier((g.get("attributes") or {}).get("formality")) - occ_tier) <= 1
        ]
        occ_coverage[occ] = min(1.0, len(occ_items) / max(2, max_pieces * 0.3))
    occ_score = sum(occ_coverage.values()) / max(len(occasion_types), 1)

    # Colour cohesion
    hex_values = [
        (g.get("attributes") or {}).get("color_hex")
        for g in garments
        if (g.get("attributes") or {}).get("color_hex")
    ]
    family_counts: Dict[str, int] = {}
    for hx in hex_values:
        fam = _nearest_neutral(hx)
        family_counts[fam] = family_counts.get(fam, 0) + 1
    total_colored = len(hex_values) or 1
    top3 = sorted(family_counts.items(), key=lambda x: -x[1])[:3]
    top3_total = sum(c for _, c in top3)
    palette_variety = 1.0 if len(family_counts) <= 3 else 0.5
    color_cohesion = min(1.0, (top3_total / total_colored) * 0.6 + palette_variety * 0.4)

    palette_names = [fam for fam, _ in top3]
    palette_hexes: List[str] = []
    for fam_name, _ in top3:
        candidates_h = [
            (g.get("attributes") or {}).get("color_hex")
            for g in garments
            if (g.get("attributes") or {}).get("color_hex")
            and _nearest_neutral((g.get("attributes") or {}).get("color_hex")) == fam_name
        ]
        palette_hexes.append(candidates_h[0] if candidates_h else "#EDE8E2")

    practicality = sum(
        _mat_score((g.get("attributes") or {}).get("material")) for g in garments
    ) / n

    w = _SCORE_WEIGHTS
    total_score = round(
        w.get("valid_combinations", 0.40) * comb_norm * 100
        + w.get("occasion_coverage", 0.25) * occ_score * 100
        + w.get("color_cohesion", 0.20) * color_cohesion * 100
        + w.get("practicality", 0.15) * practicality * 100,
        1,
    )

    return {
        "total_score": total_score,
        "valid_combinations": outfit_count,
        "color_palette": palette_hexes,
        "color_names": palette_names,
        "occasion_coverage": {k: round(v, 2) for k, v in occ_coverage.items()},
        "practicality_score": round(practicality, 2),
    }


# ─────────────────────────────────────────────────────────────
# Core algorithm
# ─────────────────────────────────────────────────────────────

class CapsuleOptimizer:
    """
    Greedy + 2-opt optimizer for N-piece context-aware capsule wardrobe.

    Usage::

        optimizer = CapsuleOptimizer()
        result = optimizer.optimize(
            garments=wardrobe_dicts,
            n_pieces=12,
            occasion_types=["casual", "smart_casual"],
            climate="mixed",
        )
    """

    def __init__(self, max_2opt_iterations: int = _MAX_2OPT_ITERS) -> None:
        self._max_iters = max_2opt_iterations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        garments: List[Dict[str, Any]],
        n_pieces: int,
        occasion_types: List[str] | None = None,
        climate: str = "mixed",
        anchor_ids: List[str] | None = None,
        excluded_ids: List[str] | None = None,
    ) -> "CapsuleOptimizerResult":
        """
        Select the best *n_pieces* from *garments* for the given context.

        Parameters
        ----------
        garments
            Serialized wardrobe items (list of dicts with ``id``, ``attributes``).
        n_pieces
            Exact number of pieces to select (5–33).
        occasion_types
            Target occasions — defaults to ``["casual", "smart_casual"]``.
        climate
            ``warm | cold | mixed`` — filters by season suitability.
        anchor_ids
            Garment IDs the user insists on including (forced into capsule).
        excluded_ids
            Garment IDs to exclude from consideration.

        Returns
        -------
        CapsuleOptimizerResult
        """
        occasion_types = occasion_types or ["casual", "smart_casual"]
        anchor_ids = set(anchor_ids or [])
        excluded_ids = set(excluded_ids or [])

        # ── Step 0: validation ────────────────────────────────────────
        if n_pieces < 1:
            n_pieces = 1
        if n_pieces > len(garments):
            n_pieces = len(garments)

        # ── Step 1: hard filter ───────────────────────────────────────
        candidates = self._hard_filter(garments, climate, excluded_ids)
        if not candidates:
            candidates = [g for g in garments if g.get("id", "") not in excluded_ids]

        # ── Step 2: force anchor pieces ───────────────────────────────
        selected: List[Dict[str, Any]] = []
        used_ids: Set[str] = set()

        for g in candidates:
            if g.get("id", "") in anchor_ids:
                selected.append(g)
                used_ids.add(g["id"])

        # ── Step 3: category seed ─────────────────────────────────────
        candidates_sorted = sorted(candidates, key=_versatility_score, reverse=True)

        for cat_key in _SEEDING_ORDER:
            if len(selected) >= n_pieces:
                break
            best = next(
                (g for g in candidates_sorted
                 if _garment_cat(g) == cat_key and g.get("id", "") not in used_ids),
                None,
            )
            if best:
                selected.append(best)
                used_ids.add(best["id"])

        # ── Step 4: greedy fill ───────────────────────────────────────
        while len(selected) < n_pieces:
            best_gain = -1
            best_cand = None
            cur_count = _count_valid_outfits(selected, occasion_types)

            for g in candidates_sorted:
                if g.get("id", "") in used_ids:
                    continue
                new_count = _count_valid_outfits(selected + [g], occasion_types)
                gain = new_count - cur_count
                if gain > best_gain:
                    best_gain = gain
                    best_cand = g

            if best_cand is None:
                break
            selected.append(best_cand)
            used_ids.add(best_cand["id"])

        # ── Step 5: 2-opt local search ────────────────────────────────
        selected = self._two_opt(selected, candidates_sorted, occasion_types, n_pieces, anchor_ids)

        # ── Step 6: score final capsule ───────────────────────────────
        scored = score_capsule(selected, occasion_types, n_pieces)

        # ── Step 7: compute per-piece contributions ───────────────────
        piece_contributions = self._compute_contributions(selected, occasion_types)

        # ── Step 8: detect missing pieces ─────────────────────────────
        missing = self._detect_missing(selected, occasion_types, climate)

        # ── Step 9: build alternatives ────────────────────────────────
        alternatives = self._build_alternatives(
            selected, candidates_sorted, occasion_types, n_pieces, anchor_ids,
        )

        return CapsuleOptimizerResult(
            selected_garments=selected,
            n_pieces=len(selected),
            total_wardrobe=len(garments),
            total_score=scored["total_score"],
            valid_combinations=scored["valid_combinations"],
            color_palette=scored["color_palette"],
            color_names=scored["color_names"],
            occasion_coverage=scored["occasion_coverage"],
            practicality_score=scored["practicality_score"],
            piece_contributions=piece_contributions,
            missing_pieces=missing,
            alternatives=alternatives,
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _hard_filter(
        self,
        garments: List[Dict[str, Any]],
        climate: str,
        excluded_ids: Set[str],
    ) -> List[Dict[str, Any]]:
        """Remove garments incompatible with the target context."""
        accepted_seasons = _CLIMATE_SEASONS.get(climate, _CLIMATE_SEASONS.get("mixed", []))
        result: List[Dict[str, Any]] = []

        for g in garments:
            gid = g.get("id", "")
            if gid in excluded_ids:
                continue
            attrs = g.get("attributes") or {}
            conf = attrs.get("confidence", 1.0) or 1.0
            if conf < _CONFIDENCE_MIN:
                continue
            seasons = attrs.get("seasons") or []
            if seasons and not any(s.lower() in accepted_seasons for s in seasons):
                continue
            result.append(g)

        return result

    def _two_opt(
        self,
        selected: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        occasion_types: List[str],
        n_pieces: int,
        anchor_ids: Set[str],
    ) -> List[Dict[str, Any]]:
        """Try swapping each selected piece with each non-selected candidate."""
        used_ids = {g["id"] for g in selected}
        pool = [g for g in candidates if g["id"] not in used_ids]

        if not pool:
            return selected

        current_score = score_capsule(selected, occasion_types, n_pieces)["total_score"]
        improved = True
        iteration = 0

        while improved and iteration < self._max_iters:
            improved = False
            iteration += 1

            for i, piece in enumerate(selected):
                # Never swap anchor pieces
                if piece.get("id", "") in anchor_ids:
                    continue

                for replacement in pool:
                    trial = list(selected)
                    trial[i] = replacement
                    trial_score = score_capsule(trial, occasion_types, n_pieces)["total_score"]

                    if trial_score > current_score:
                        # Accept the swap
                        old_id = selected[i]["id"]
                        selected = trial
                        current_score = trial_score
                        used_ids.discard(old_id)
                        used_ids.add(replacement["id"])
                        pool = [g for g in candidates if g["id"] not in used_ids]
                        improved = True
                        logger.debug(
                            "2-opt: swapped %s → %s (score %.1f → %.1f)",
                            old_id, replacement["id"],
                            current_score - (trial_score - current_score),
                            current_score,
                        )
                        break  # restart inner loop with updated selection

                if improved:
                    break  # restart outer loop

        return selected

    def _compute_contributions(
        self,
        selected: List[Dict[str, Any]],
        occasion_types: List[str],
    ) -> List[Dict[str, Any]]:
        """For each selected piece, compute how many outfits it enables."""
        base_count = _count_valid_outfits(selected, occasion_types)
        contributions: List[Dict[str, Any]] = []

        for g in selected:
            without = [x for x in selected if x.get("id") != g.get("id")]
            without_count = _count_valid_outfits(without, occasion_types)
            contribution = max(0, base_count - without_count)
            cat = _garment_cat(g)
            contributions.append({
                "garment_id": g.get("id", ""),
                "category": cat,
                "role": _CATEGORY_ROLES.get(cat, "anchor"),
                "outfit_contribution": contribution,
                "versatility_score": round(_versatility_score(g), 3),
            })

        return sorted(contributions, key=lambda x: -x["outfit_contribution"])

    def _detect_missing(
        self,
        selected: List[Dict[str, Any]],
        occasion_types: List[str],
        climate: str,
    ) -> List[str]:
        """Identify gaps vs the ideal occasion template."""
        selected_cats = {_garment_cat(g) for g in selected}
        missing: List[str] = []

        # Check against templates for each occasion type
        for occ in occasion_types:
            template = _OCCASION_TEMPLATES.get(occ, {})
            for cat_need in template:
                if cat_need not in selected_cats:
                    label = _MISSING_LABELS.get(cat_need, cat_need.capitalize())
                    if label not in missing:
                        missing.append(label)

        if climate == "cold" and "outerwear" not in selected_cats:
            missing.append("Warm layer (coat or sweater)")
        if climate == "warm" and not any(
            "linen" in ((g.get("attributes") or {}).get("material") or "").lower()
            for g in selected
        ):
            missing.append("Breathable linen or cotton piece")

        return missing[:_MAX_MISSING]

    def _build_alternatives(
        self,
        best_selected: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        occasion_types: List[str],
        n_pieces: int,
        anchor_ids: Set[str],
    ) -> List[Dict[str, Any]]:
        """Build up to 2 alternative capsule options."""
        alternatives: List[Dict[str, Any]] = []
        best_ids = {g["id"] for g in best_selected}

        # Alt 1: swap lowest-contribution piece
        if len(best_selected) > 1:
            base_count = _count_valid_outfits(best_selected, occasion_types)
            min_piece = min(
                best_selected,
                key=lambda g: (
                    base_count - _count_valid_outfits(
                        [x for x in best_selected if x["id"] != g["id"]], occasion_types
                    )
                ) if g["id"] not in anchor_ids else float("inf"),
            )
            if min_piece["id"] not in anchor_ids:
                alt_base = [g for g in best_selected if g["id"] != min_piece["id"]]
                replacement = next(
                    (g for g in candidates if g["id"] not in best_ids and g["id"] not in anchor_ids),
                    None,
                )
                if replacement:
                    alt_sel = alt_base + [replacement]
                    alt_scored = score_capsule(alt_sel, occasion_types, n_pieces)
                    alternatives.append({
                        "garments": alt_sel,
                        "total_score": alt_scored["total_score"],
                        "valid_combinations": alt_scored["valid_combinations"],
                        "label": "swap_lowest_contribution",
                    })

        # Alt 2: practicality-first (reorder by material)
        pract_sorted = sorted(
            best_selected,
            key=lambda g: _mat_score((g.get("attributes") or {}).get("material")),
            reverse=True,
        )
        alt_b = pract_sorted[:n_pieces]
        if len(alt_b) >= 3:
            alt_scored = score_capsule(alt_b, occasion_types, n_pieces)
            alternatives.append({
                "garments": alt_b,
                "total_score": alt_scored["total_score"],
                "valid_combinations": alt_scored["valid_combinations"],
                "label": "practicality_first",
            })

        return alternatives


# ─────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────

class CapsuleOptimizerResult:
    """Holds the output of a capsule optimization run."""

    __slots__ = (
        "selected_garments", "n_pieces", "total_wardrobe", "total_score",
        "valid_combinations", "color_palette", "color_names",
        "occasion_coverage", "practicality_score", "piece_contributions",
        "missing_pieces", "alternatives",
    )

    def __init__(
        self,
        selected_garments: List[Dict[str, Any]],
        n_pieces: int,
        total_wardrobe: int,
        total_score: float,
        valid_combinations: int,
        color_palette: List[str],
        color_names: List[str],
        occasion_coverage: Dict[str, float],
        practicality_score: float,
        piece_contributions: List[Dict[str, Any]],
        missing_pieces: List[str],
        alternatives: List[Dict[str, Any]],
    ) -> None:
        self.selected_garments = selected_garments
        self.n_pieces = n_pieces
        self.total_wardrobe = total_wardrobe
        self.total_score = total_score
        self.valid_combinations = valid_combinations
        self.color_palette = color_palette
        self.color_names = color_names
        self.occasion_coverage = occasion_coverage
        self.practicality_score = practicality_score
        self.piece_contributions = piece_contributions
        self.missing_pieces = missing_pieces
        self.alternatives = alternatives

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_garments": self.selected_garments,
            "n_pieces": self.n_pieces,
            "total_wardrobe": self.total_wardrobe,
            "total_score": self.total_score,
            "valid_combinations": self.valid_combinations,
            "color_palette": self.color_palette,
            "color_names": self.color_names,
            "occasion_coverage": self.occasion_coverage,
            "practicality_score": self.practicality_score,
            "piece_contributions": self.piece_contributions,
            "missing_pieces": self.missing_pieces,
            "alternatives": self.alternatives,
        }
