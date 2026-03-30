"""
Capsule Builder Service
=======================
Pure rule-based capsule wardrobe algorithm.
All tuneable constants are loaded from config/capsule/capsule_config.json —
no LLM calls are made here.

Public API
----------
build_travel_capsule(request: TravelCapsuleRequest, garments: list[dict]) -> TravelCapsuleResponse
"""
from __future__ import annotations

import json
from itertools import product as iproduct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "capsule" / "capsule_config.json"


def _load_config() -> Dict[str, Any]:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_CFG: Dict[str, Any] = _load_config()

# Unpack top-level config sections into module-level constants
_SCORE_WEIGHTS: Dict[str, float] = _CFG["score_weights"]
_CONFIDENCE_MIN: float = _CFG["confidence_min"]
_OUTFIT_TARGETS: Dict[str, Any] = _CFG["outfit_targets"]
_MATERIAL_PRACTICALITY: Dict[str, float] = _CFG["material_practicality"]
_CLIMATE_SEASONS: Dict[str, List[str]] = _CFG["climate_seasons"]
_NEUTRAL_FAMILIES: List[Tuple[str, Tuple[int, int, int]]] = [
    (e["name"], tuple(e["rgb"])) for e in _CFG["neutral_families"]
]
_FORMALITY_TIERS: Dict[str, int] = _CFG["formality_tiers"]
_OCCASION_FORMALITIES: Dict[str, List[str]] = _CFG["occasion_formalities"]
_CATEGORY_ROLES: Dict[str, str] = _CFG["category_roles"]
_CAT_VERSATILITY: Dict[str, float] = _CFG["category_versatility_base"]
_OCCASION_TEMPLATES: Dict[str, Dict[str, int]] = _CFG["occasion_templates"]
_MISSING_LABELS: Dict[str, str] = _CFG["missing_piece_labels"]
_BEACH_MATERIALS: List[str] = _CFG["beach_allowed_materials"]
_SHORT_TRIP_DAYS: int = _CFG["short_trip_days_threshold"]
_SHORT_TRIP_PRACT_MIN: float = _CFG["short_trip_practicality_min"]
_SEEDING_ORDER: List[str] = _CFG["seeding_order"]
_MAX_MISSING: int = _CFG["max_missing_pieces"]


# ─────────────────────────────────────────────────────────────
# Pydantic request / response models
# ─────────────────────────────────────────────────────────────

class TravelCapsuleRequest(BaseModel):
    user_id: str
    occasion: str = "travel"
    destination_climate: str = "mixed"   # warm | cold | mixed
    duration_days: int = 7
    occasion_types: List[str] = ["casual", "smart_casual"]
    max_pieces: int = 10


class CapsuleGarmentEntry(BaseModel):
    garment: Dict[str, Any]
    role: str                  # anchor | layer | accent | shoes | accessory
    outfit_contribution: int


class CapsuleGroup(BaseModel):
    garments: List[CapsuleGarmentEntry]
    valid_combinations: int
    color_palette: List[str]
    color_names: List[str]
    occasion_coverage: Dict[str, float]
    practicality_score: float
    total_score: float
    summary: str


class TravelCapsuleResponse(BaseModel):
    best_group: CapsuleGroup
    alternative_groups: List[CapsuleGroup]
    missing_pieces: List[str]
    total_wardrobe_items: int
    pieces_selected: int
    source: str = "rule"


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

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


def _mat_score(material: Optional[str]) -> float:
    """Return practicality score for a material string."""
    if not material:
        return _MATERIAL_PRACTICALITY["default"]
    m = material.lower().strip()
    for key, score in _MATERIAL_PRACTICALITY.items():
        if key == "default":
            continue
        if key in m:
            return score
    return _MATERIAL_PRACTICALITY["default"]


def _form_tier(formality: Optional[str]) -> int:
    if not formality:
        return 0
    return _FORMALITY_TIERS.get(formality.lower().replace(" ", "_"), 1)


def _garment_cat(g: Dict[str, Any]) -> str:
    """Extract category string from a garment dict."""
    attrs = g.get("attributes") or {}
    cat = attrs.get("category", "")
    # Handle both plain str and enum-like {"value": "top"} shapes
    if isinstance(cat, dict):
        return cat.get("value", "")
    return str(cat)


def _outfit_target(max_pieces: int) -> int:
    """Return the expected outfit count for the given capsule size."""
    for tier in ("small", "medium", "large"):
        t = _OUTFIT_TARGETS[tier]
        if max_pieces <= t["max_pieces_le"]:
            return t["target_outfits"]
    return _OUTFIT_TARGETS["large"]["target_outfits"]


# ─────────────────────────────────────────────────────────────
# Outfit counting
# ─────────────────────────────────────────────────────────────

def _count_valid_outfits(garments: List[Dict], occasion_types: List[str]) -> int:
    """Count valid outfit combinations (top+bottom or dress) from a garment list."""
    tops      = [g for g in garments if _garment_cat(g) == "top"]
    bottoms   = [g for g in garments if _garment_cat(g) == "bottom"]
    dresses   = [g for g in garments if _garment_cat(g) == "dress"]
    outerw    = [g for g in garments if _garment_cat(g) == "outerwear"]
    shoes_lst = [g for g in garments if _garment_cat(g) == "shoes"]

    target_tiers = {_form_tier(f) for f in occasion_types}
    accepted_tiers: set = set()
    for t in target_tiers:
        accepted_tiers |= {max(0, t - 1), t, min(3, t + 1)}

    shoe_mult = max(1, len(shoes_lst))
    count = 0

    for top in tops:
        for bottom in bottoms:
            top_tier    = _form_tier((top.get("attributes") or {}).get("formality"))
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
# Scoring
# ─────────────────────────────────────────────────────────────

def _score_capsule(
    garments: List[Dict],
    occasion_types: List[str],
    max_pieces: int,
) -> Dict[str, Any]:
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
        candidates = [
            (g.get("attributes") or {}).get("color_hex")
            for g in garments
            if (g.get("attributes") or {}).get("color_hex")
            and _nearest_neutral((g.get("attributes") or {}).get("color_hex")) == fam_name
        ]
        palette_hexes.append(candidates[0] if candidates else "#EDE8E2")

    practicality = sum(
        _mat_score((g.get("attributes") or {}).get("material")) for g in garments
    ) / n

    w = _SCORE_WEIGHTS
    total_score = round(
        w["valid_combinations"] * comb_norm * 100
        + w["occasion_coverage"]  * occ_score  * 100
        + w["color_cohesion"]     * color_cohesion * 100
        + w["practicality"]       * practicality    * 100,
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
# Group builder
# ─────────────────────────────────────────────────────────────

def _build_group(
    garments: List[Dict],
    occasion_types: List[str],
    max_pieces: int,
) -> CapsuleGroup:
    scored = _score_capsule(garments, occasion_types, max_pieces)
    base_count = scored["valid_combinations"]

    entries: List[CapsuleGarmentEntry] = []
    for g in garments:
        without = [x for x in garments if x["id"] != g["id"]]
        without_count = _count_valid_outfits(without, occasion_types)
        contribution = max(0, base_count - without_count)
        role = _CATEGORY_ROLES.get(_garment_cat(g), "anchor")
        entries.append(CapsuleGarmentEntry(
            garment=g,
            role=role,
            outfit_contribution=contribution,
        ))

    occ_str = " & ".join(occasion_types) if occasion_types else "travel"
    summary = (
        f"{len(garments)} versatile pieces covering "
        f"{scored['valid_combinations']} outfits across {occ_str}."
    )

    return CapsuleGroup(
        garments=entries,
        valid_combinations=scored["valid_combinations"],
        color_palette=scored["color_palette"],
        color_names=scored["color_names"],
        occasion_coverage=scored["occasion_coverage"],
        practicality_score=scored["practicality_score"],
        total_score=scored["total_score"],
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def build_travel_capsule(
    request: TravelCapsuleRequest,
    garments: List[Dict[str, Any]],
) -> TravelCapsuleResponse:
    """
    Build a travel capsule from a pre-loaded list of garment dicts.

    Parameters
    ----------
    request : TravelCapsuleRequest
        Capsule parameters (occasion, climate, duration, max_pieces …)
    garments : list[dict]
        Serialised GarmentItem objects already loaded by the caller.

    Returns
    -------
    TravelCapsuleResponse
    """
    occasion       = request.occasion
    climate        = request.destination_climate
    duration       = request.duration_days
    max_pieces     = request.max_pieces
    occasion_types = request.occasion_types or _OCCASION_FORMALITIES.get(occasion, ["casual"])
    accepted_seasons = _CLIMATE_SEASONS.get(climate, _CLIMATE_SEASONS["mixed"])

    # ── Step 1: Filter candidates ─────────────────────────────
    candidates: List[Dict] = []
    for g in garments:
        attrs = g.get("attributes") or {}
        conf = attrs.get("confidence", 1.0) or 1.0
        if conf < _CONFIDENCE_MIN:
            continue
        seasons = attrs.get("seasons") or []
        if seasons and not any(s.lower() in accepted_seasons for s in seasons):
            continue
        if occasion == "beach":
            mat = (attrs.get("material") or "").lower()
            if mat and not any(m in mat for m in _BEACH_MATERIALS):
                continue
        candidates.append(g)

    if not candidates:
        candidates = list(garments)

    # ── Step 2: Sort by versatility ───────────────────────────
    def _vscore(g: Dict) -> float:
        attrs = g.get("attributes") or {}
        cat   = _garment_cat(g)
        base  = _CAT_VERSATILITY.get(cat, _CAT_VERSATILITY["default"])
        seasons_bonus = len(attrs.get("seasons") or []) / 4 * 0.1
        return base + seasons_bonus + _mat_score(attrs.get("material")) * 0.1

    candidates_sorted = sorted(candidates, key=_vscore, reverse=True)

    # ── Step 3: Category seed ─────────────────────────────────
    selected: List[Dict] = []
    used_ids: set = set()
    for cat_key in _SEEDING_ORDER:
        best = next(
            (g for g in candidates_sorted if _garment_cat(g) == cat_key and g["id"] not in used_ids),
            None,
        )
        if best and len(selected) < max_pieces:
            selected.append(best)
            used_ids.add(best["id"])

    # ── Step 4: Greedy fill ───────────────────────────────────
    for _ in range(max_pieces - len(selected)):
        best_gain  = -1
        best_cand  = None
        cur_count  = _count_valid_outfits(selected, occasion_types)
        for g in candidates_sorted:
            if g["id"] in used_ids:
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

    # Short trip: prefer light materials
    if occasion == "travel" and duration <= _SHORT_TRIP_DAYS:
        light = [g for g in selected if _mat_score((g.get("attributes") or {}).get("material")) >= _SHORT_TRIP_PRACT_MIN]
        if len(light) >= 4:
            selected = light[:max_pieces]

    # ── Step 5: Build groups ──────────────────────────────────
    best_group   = _build_group(selected, occasion_types, max_pieces)
    alternatives: List[CapsuleGroup] = []

    # Alt A — swap lowest-contribution item
    if len(selected) > 1 and best_group.garments:
        min_entry   = min(best_group.garments, key=lambda e: e.outfit_contribution)
        min_id      = min_entry.garment["id"]
        alt_a_base  = [g for g in selected if g["id"] != min_id]
        replacement = next(
            (g for g in candidates_sorted if g["id"] not in {x["id"] for x in alt_a_base + selected}),
            None,
        )
        if replacement:
            alternatives.append(_build_group(alt_a_base + [replacement], occasion_types, max_pieces))

    # Alt B — practicality-first
    pract_sorted = sorted(selected, key=lambda g: _mat_score((g.get("attributes") or {}).get("material")), reverse=True)
    alt_b = pract_sorted[:min(len(selected), max_pieces)]
    if len(alt_b) >= 3:
        alternatives.append(_build_group(alt_b, occasion_types, max_pieces))

    # Alt C — minimalist (6 pieces)
    if len(selected) > 6:
        alternatives.append(_build_group(selected[:6], occasion_types, max(6, max_pieces)))

    # ── Step 6: Missing pieces ────────────────────────────────
    selected_cats = {_garment_cat(g) for g in selected}
    template      = _OCCASION_TEMPLATES.get(occasion, {"top": 2, "bottom": 2, "shoes": 1})
    missing: List[str] = []
    for cat_need in template:
        if cat_need not in selected_cats:
            missing.append(_MISSING_LABELS.get(cat_need, cat_need.capitalize()))

    if climate == "cold" and "outerwear" not in selected_cats:
        missing.append("Warm layer (coat or sweater)")
    if climate == "warm" and not any(
        "linen" in ((g.get("attributes") or {}).get("material") or "").lower() for g in selected
    ):
        missing.append("Breathable linen or cotton piece")

    return TravelCapsuleResponse(
        best_group=best_group,
        alternative_groups=alternatives,
        missing_pieces=list(dict.fromkeys(missing))[: _MAX_MISSING],
        total_wardrobe_items=len(garments),
        pieces_selected=len(selected),
        source="rule",
    )
