"""
Inspiration Scorer — Layer 2 Style
====================================

Scores how well a candidate outfit matches a decoded Style DNA from an
inspiration image.

Pipeline:
  1. LLM decodes the inspiration image → Style DNA (palette, formality, 
     aesthetic, silhouette, key piece type)
  2. For each wardrobe garment × each slot, compute a fit score
  3. Greedy slot-filling assembles complete outfits
  4. Body-shape constraints veto bad combos
  5. Inspiration fidelity re-ranks the assembled outfits

Public API:
    scorer = InspirationScorer()
    dna    = await scorer.decode_inspiration(image_b64, body_shape)
    slots  = scorer.build_slot_blueprint(dna)
    combos = scorer.assemble_outfits(slots, garments, top_k, body_shape)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.core.models import Garment, GarmentCategory, UserContext
from src.core import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Style DNA
# ═══════════════════════════════════════════════════════════════

@dataclass
class StyleDNA:
    """Decoded style signal from an inspiration image."""
    palette: List[str] = field(default_factory=list)          # e.g. ["ivory","camel","cognac"]
    palette_hex: List[str] = field(default_factory=list)      # hex codes
    formality: str = "casual"                                  # casual, smart_casual, business, formal
    aesthetic: List[str] = field(default_factory=list)         # e.g. ["quiet luxury","minimalist"]
    silhouette_top: str = "regular"                            # slim, regular, oversized
    silhouette_bottom: str = "regular"
    key_piece_category: Optional[str] = None                   # e.g. "outerwear"
    key_piece_description: Optional[str] = None                # "oversized blazer"
    layering_depth: int = 1                                    # 1 = no layers, 2+ = layered
    season_vibe: str = "all_season"                            # spring, summer, fall, winter
    mood: Optional[str] = None                                 # "editorial", "street", "bohemian"
    color_relationship: str = "tonal"                          # tonal, complementary, monochrome, contrast


# ═══════════════════════════════════════════════════════════════
#  Slot Blueprint
# ═══════════════════════════════════════════════════════════════

@dataclass
class SlotRequirement:
    """Requirements for one outfit slot."""
    role: str               # "hero", "supporting", "shoes", "accessory"
    categories: List[str]   # acceptable garment categories
    importance: float       # 1.0 = must-have, 0.5 = nice-to-have
    # Style DNA alignment prefs
    preferred_formalities: List[str] = field(default_factory=list)
    preferred_silhouette: Optional[str] = None
    color_preference: str = "palette"  # "palette" | "neutral" | "dark" | "any"
    # Body shape modifiers
    body_promote: List[str] = field(default_factory=list)      # subcategory keywords to prefer
    body_avoid: List[str] = field(default_factory=list)        # subcategory keywords to penalize


@dataclass
class ScoredCandidate:
    """A garment scored for a specific slot."""
    garment: Garment
    slot_role: str
    score: float
    breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class AssembledOutfit:
    """A fully assembled outfit from slots."""
    garments: List[Garment]
    slot_assignments: Dict[str, str]   # slot_role → garment.id
    total_score: float
    inspiration_fidelity: float
    body_harmony: float
    style_score: float
    breakdown: Dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
#  Body Shape Constraint Tables
# ═══════════════════════════════════════════════════════════════

# Slot-level preferences per body shape
BODY_SHAPE_SLOT_RULES: Dict[str, Dict[str, Dict]] = {
    "pear": {
        "top":       {"promote": ["structured", "puff sleeve", "boat neck", "wide", "off-shoulder"],
                      "avoid": ["cropped", "tight", "clingy"],
                      "color_pref": "light"},    # light top draws eye up
        "bottom":    {"promote": ["A-line", "straight", "bootcut", "midi"],
                      "avoid": ["wide-leg", "mini", "printed", "pleated"],
                      "color_pref": "dark"},
        "shoes":     {"promote": ["pointed", "heel", "nude"],
                      "avoid": ["ankle strap", "chunky flat"]},
        "accessory": {"promote": ["necklace", "earring", "scarf"],
                      "avoid": ["wide belt", "hip bag"]},
    },
    "apple": {
        "top":       {"promote": ["A-line", "empire", "V-neck", "wrap", "relaxed"],
                      "avoid": ["tight", "bodycon", "horizontal stripe"],
                      "color_pref": "dark"},
        "bottom":    {"promote": ["straight", "bootcut", "slim"],
                      "avoid": ["wide-leg", "pleated front"],
                      "color_pref": "dark"},
        "shoes":     {"promote": ["heel", "pointed"],
                      "avoid": []},
        "accessory": {"promote": ["long necklace", "V pendant"],
                      "avoid": ["wide belt"]},
    },
    "hourglass": {
        "top":       {"promote": ["fitted", "wrap", "V-neck", "peplum"],
                      "avoid": ["boxy", "oversized"],
                      "color_pref": "any"},
        "bottom":    {"promote": ["high-waist", "fitted", "pencil", "straight"],
                      "avoid": ["low-rise", "baggy"],
                      "color_pref": "any"},
        "shoes":     {"promote": ["heel", "any"], "avoid": []},
        "accessory": {"promote": ["belt", "waist-defining"], "avoid": []},
    },
    "rectangle": {
        "top":       {"promote": ["peplum", "ruffle", "layered", "wrap", "structured"],
                      "avoid": ["boxy straight"],
                      "color_pref": "any"},
        "bottom":    {"promote": ["wide-leg", "A-line", "pleated", "flared"],
                      "avoid": ["straight tube"],
                      "color_pref": "any"},
        "shoes":     {"promote": ["any"], "avoid": []},
        "accessory": {"promote": ["belt", "statement"], "avoid": []},
    },
    "inverted_triangle": {
        "top":       {"promote": ["V-neck", "slim", "raglan", "dark"],
                      "avoid": ["puff sleeve", "wide neckline", "epaulette"],
                      "color_pref": "dark"},
        "bottom":    {"promote": ["wide-leg", "flared", "A-line", "printed"],
                      "avoid": ["skinny", "slim"],
                      "color_pref": "light"},
        "shoes":     {"promote": ["any"], "avoid": []},
        "accessory": {"promote": ["hip belt", "long pendant"], "avoid": ["shoulder pad"]},
    },
}

# Assembly-level hard vetoes: (body_shape → list of veto functions)
# Each function takes (top_garment, bottom_garment) → bool (True = veto)
def _veto_pear_tight_top_wide_bottom(top: Garment, bottom: Garment) -> bool:
    """Pear: tight top + wide-leg bottom emphasizes hip width."""
    t_fit = (top.attributes.fit or "").lower()
    b_sub = (bottom.attributes.subcategory or "").lower()
    return t_fit in ("tight", "fitted", "slim", "bodycon") and "wide" in b_sub

def _veto_apple_tight_top_no_layer(top: Garment, _: Garment) -> bool:
    """Apple: tight top without layering draws attention to midsection."""
    t_fit = (top.attributes.fit or "").lower()
    return t_fit in ("tight", "bodycon", "fitted")


ASSEMBLY_VETOES: Dict[str, List] = {
    "pear": [_veto_pear_tight_top_wide_bottom],
    "apple": [_veto_apple_tight_top_no_layer],
}


# ═══════════════════════════════════════════════════════════════
#  Color Matching Utilities
# ═══════════════════════════════════════════════════════════════

def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert #RRGGBB to (r, g, b)."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (128, 128, 128)
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _color_distance(hex1: str, hex2: str) -> float:
    """Euclidean RGB distance normalized to 0-1 (0 = identical, 1 = max)."""
    r1, g1, b1 = _hex_to_rgb(hex1)
    r2, g2, b2 = _hex_to_rgb(hex2)
    dist = math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
    return min(1.0, dist / 441.67)  # max euclidean = sqrt(3*255^2) ≈ 441.67


# Simple name-to-hex fallback for common colors
_COLOR_NAME_HEX: Dict[str, str] = {
    "black": "#1A1A1A", "white": "#FAFAFA", "ivory": "#FFFFF0",
    "cream": "#FFFDD0", "beige": "#F5F5DC", "camel": "#C19A6B",
    "tan": "#D2B48C", "brown": "#8B4513", "cognac": "#9A463D",
    "navy": "#1B2A4A", "blue": "#3366CC", "light blue": "#ADD8E6",
    "red": "#CC3333", "burgundy": "#800020", "pink": "#FFC0CB",
    "green": "#228B22", "olive": "#808000", "grey": "#888888",
    "gray": "#888888", "charcoal": "#333333", "lavender": "#E6E6FA",
    "purple": "#800080", "yellow": "#FFD700", "orange": "#FF8C00",
    "coral": "#FF7F50", "mint": "#98FF98", "nude": "#E8C4A0",
    "gold": "#FFD700", "silver": "#C0C0C0", "khaki": "#BDB76B",
}


def _name_to_hex(name: str) -> Optional[str]:
    return _COLOR_NAME_HEX.get(name.lower().strip())


def _color_affinity(garment_hex: Optional[str], garment_color_name: str,
                    dna_palette_hex: List[str], dna_palette_names: List[str]) -> float:
    """Score how well a garment color matches the Style DNA palette. 0-1."""
    if not dna_palette_hex and not dna_palette_names:
        return 0.5  # neutral when no palette

    best = 0.0

    # Try hex comparison first
    g_hex = garment_hex
    if not g_hex:
        g_hex = _name_to_hex(garment_color_name)

    if g_hex:
        for p_hex in dna_palette_hex:
            if p_hex:
                dist = _color_distance(g_hex, p_hex)
                sim = 1.0 - dist
                best = max(best, sim)

    # Name comparison fallback
    g_name = garment_color_name.lower().strip()
    for p_name in dna_palette_names:
        if g_name == p_name.lower().strip():
            best = max(best, 0.95)
        # Partial match (e.g. "light blue" matches "blue")
        elif g_name in p_name.lower() or p_name.lower() in g_name:
            best = max(best, 0.70)

    return best


# ═══════════════════════════════════════════════════════════════
#  Main Scorer
# ═══════════════════════════════════════════════════════════════

class InspirationScorer:
    """
    Scores garments against a Style DNA and assembles outfit combinations
    that best match the inspiration while respecting body shape constraints.
    """

    # --- Slot Blueprint Builder ---

    def build_slot_blueprint(
        self,
        dna: StyleDNA,
        body_shape: Optional[str] = None,
    ) -> List[SlotRequirement]:
        """
        Convert Style DNA into a list of slot requirements.
        Body shape rules are overlaid if provided.
        """
        formalities = [dna.formality]
        if dna.formality == "smart_casual":
            formalities.append("casual")
        elif dna.formality == "business":
            formalities.append("smart_casual")

        slots: List[SlotRequirement] = []

        # Hero slot — key piece or top
        hero_cats = ["top"]
        if dna.key_piece_category:
            kpc = dna.key_piece_category.lower()
            if kpc in ("outerwear", "dress"):
                hero_cats = [kpc]
            else:
                hero_cats = [kpc, "top"]

        slots.append(SlotRequirement(
            role="hero",
            categories=hero_cats,
            importance=1.0,
            preferred_formalities=formalities,
            preferred_silhouette=dna.silhouette_top,
            color_preference="palette",
        ))

        # Supporting bottom (skip if hero is a dress)
        if "dress" not in hero_cats:
            slots.append(SlotRequirement(
                role="bottom",
                categories=["bottom"],
                importance=1.0,
                preferred_formalities=formalities,
                preferred_silhouette=dna.silhouette_bottom,
                color_preference="palette",
            ))

        # Shoes
        slots.append(SlotRequirement(
            role="shoes",
            categories=["shoes"],
            importance=0.7,
            preferred_formalities=formalities,
            color_preference="neutral",
        ))

        # Optional accessory
        slots.append(SlotRequirement(
            role="accessory",
            categories=["accessory"],
            importance=0.4,
            color_preference="any",
        ))

        # Optional outerwear for layering
        if dna.layering_depth >= 2 and "outerwear" not in hero_cats:
            slots.append(SlotRequirement(
                role="layer",
                categories=["outerwear"],
                importance=0.5,
                preferred_formalities=formalities,
                color_preference="palette",
            ))

        # Overlay body shape constraints
        if body_shape:
            self._apply_body_shape_to_slots(slots, body_shape)

        return slots

    def _apply_body_shape_to_slots(self, slots: List[SlotRequirement], body_shape: str):
        """Overlay body shape promote/avoid rules onto slot requirements."""
        shape = body_shape.lower().strip()
        rules = BODY_SHAPE_SLOT_RULES.get(shape, {})
        if not rules:
            return

        for slot in slots:
            # Map slot role to body shape category
            slot_cat = slot.role
            if slot_cat == "hero":
                slot_cat = slot.categories[0] if slot.categories else "top"
            if slot_cat == "layer":
                slot_cat = "top"  # outerwear follows top rules
            if slot_cat == "bottom":
                slot_cat = "bottom"

            rule = rules.get(slot_cat, {})
            if rule:
                slot.body_promote = rule.get("promote", [])
                slot.body_avoid = rule.get("avoid", [])
                # Apply color preference from body shape
                color_pref = rule.get("color_pref")
                if color_pref and color_pref != "any":
                    slot.color_preference = color_pref

    # --- Garment-Slot Scoring ---

    def score_garment_for_slot(
        self,
        garment: Garment,
        slot: SlotRequirement,
        dna: StyleDNA,
    ) -> ScoredCandidate:
        """
        Score a single garment against a single slot requirement.

        Weights:
          color_affinity:    30%
          category_match:    20%
          formality_delta:   15%
          aesthetic_overlap:  10%
          material_affinity:  10%
          body_fit_score:     15%
        """
        bd: Dict[str, float] = {}

        # 1. Category match (hard gate)
        cat_val = garment.attributes.category.value.lower()
        cat_match = 1.0 if cat_val in [c.lower() for c in slot.categories] else 0.0
        bd["category"] = cat_match

        if cat_match == 0.0:
            return ScoredCandidate(garment=garment, slot_role=slot.role, score=0.0, breakdown=bd)

        # 2. Color affinity
        g_color = garment.attributes.color.primary if garment.attributes.color else "unknown"
        g_hex = (garment.attributes.color.hex_codes[0]
                 if garment.attributes.color and garment.attributes.color.hex_codes
                 else None)
        bd["color"] = _color_affinity(g_hex, g_color, dna.palette_hex, dna.palette)

        # 3. Formality delta
        f_garment = garment.attributes.formality_level.value if garment.attributes.formality_level else "casual"
        f_levels = {"casual": 0, "smart_casual": 1, "business_casual": 2, "business": 3, "formal": 4}
        g_level = f_levels.get(f_garment, 0)
        dna_level = f_levels.get(dna.formality, 0)
        delta = abs(g_level - dna_level)
        bd["formality"] = max(0.0, 1.0 - delta * 0.3)

        # 4. Aesthetic overlap (check subcategory / material keywords)
        aesthetic_score = 0.5  # baseline
        sub = (garment.attributes.subcategory or "").lower()
        mat = garment.attributes.material.primary if garment.attributes.material and garment.attributes.material.primary else ""
        combined_text = f"{sub} {mat}".lower()
        for tag in dna.aesthetic:
            if tag.lower() in combined_text:
                aesthetic_score = min(1.0, aesthetic_score + 0.2)
        bd["aesthetic"] = aesthetic_score

        # 5. Material affinity (simple heuristic)
        bd["material"] = 0.6  # neutral baseline

        # 6. Body fit score
        body_score = self._body_fit_score(garment, slot)
        bd["body_fit"] = body_score

        # Weighted sum
        total = (
            bd["color"]     * 0.30 +
            bd["formality"] * 0.15 +
            bd["aesthetic"]  * 0.10 +
            bd["material"]  * 0.10 +
            bd["body_fit"]  * 0.15 +
            1.0             * 0.20   # category bonus (already passed gate)
        )

        return ScoredCandidate(garment=garment, slot_role=slot.role, score=total, breakdown=bd)

    def _body_fit_score(self, garment: Garment, slot: SlotRequirement) -> float:
        """Score how well a garment fits the body-shape constraints of a slot."""
        if not slot.body_promote and not slot.body_avoid:
            return 0.7  # neutral

        sub = (garment.attributes.subcategory or "").lower()
        fit = (garment.attributes.fit or "").lower()
        combined = f"{sub} {fit}"

        score = 0.6  # baseline

        # Promote matches
        for keyword in slot.body_promote:
            if keyword.lower() in combined:
                score = min(1.0, score + 0.15)

        # Avoid matches
        for keyword in slot.body_avoid:
            if keyword.lower() in combined:
                score = max(0.1, score - 0.25)

        return score

    # --- Outfit Assembly ---

    def assemble_outfits(
        self,
        slots: List[SlotRequirement],
        garments: List[Garment],
        dna: StyleDNA,
        body_shape: Optional[str] = None,
        top_k: int = 3,
    ) -> List[AssembledOutfit]:
        """
        Greedy slot-filling assembly:
        1. Sort slots by importance (hero first)
        2. For each critical slot, pick top-N candidates
        3. For each hero pick, greedily fill remaining slots
        4. Apply assembly vetoes
        5. Score and rank complete outfits
        """
        t0 = __import__("time").perf_counter()

        # Sort slots: required first, then optional
        sorted_slots = sorted(slots, key=lambda s: -s.importance)

        # Pre-score all garments for all slots
        slot_candidates: Dict[str, List[ScoredCandidate]] = {}
        for slot in sorted_slots:
            candidates = []
            for g in garments:
                sc = self.score_garment_for_slot(g, slot, dna)
                if sc.score > 0.15:
                    candidates.append(sc)
            candidates.sort(key=lambda c: -c.score)
            slot_candidates[slot.role] = candidates

        # Find the hero slot
        hero_slot = sorted_slots[0] if sorted_slots else None
        if not hero_slot or not slot_candidates.get(hero_slot.role):
            return []

        # Try top-5 hero candidates, build an outfit for each
        hero_picks = slot_candidates[hero_slot.role][:5]
        assembled: List[AssembledOutfit] = []

        for hero_candidate in hero_picks:
            used_ids = {hero_candidate.garment.id}
            outfit_garments = [hero_candidate.garment]
            assignments = {hero_slot.role: hero_candidate.garment.id}
            slot_scores = [hero_candidate.score]

            for slot in sorted_slots[1:]:  # skip hero
                best = None
                for c in slot_candidates.get(slot.role, []):
                    if c.garment.id not in used_ids:
                        best = c
                        break
                if best:
                    used_ids.add(best.garment.id)
                    outfit_garments.append(best.garment)
                    assignments[slot.role] = best.garment.id
                    slot_scores.append(best.score)
                elif slot.importance >= 0.9:
                    # Required slot unfilled — skip this outfit
                    outfit_garments = []
                    break

            if len(outfit_garments) < 2:
                continue

            # Assembly vetoes
            if body_shape and not self._passes_assembly_vetoes(outfit_garments, body_shape):
                continue

            # Calculate composite scores
            avg_slot = sum(slot_scores) / len(slot_scores) if slot_scores else 0.5
            fidelity = self._inspiration_fidelity(outfit_garments, dna)
            body_h = self._body_harmony(outfit_garments, body_shape)

            total = avg_slot * 0.40 + fidelity * 0.30 + body_h * 0.30

            assembled.append(AssembledOutfit(
                garments=outfit_garments,
                slot_assignments=assignments,
                total_score=round(total, 3),
                inspiration_fidelity=round(fidelity, 3),
                body_harmony=round(body_h, 3),
                style_score=round(avg_slot, 3),
                breakdown={
                    "slot_avg": round(avg_slot, 3),
                    "inspiration_fidelity": round(fidelity, 3),
                    "body_harmony": round(body_h, 3),
                },
            ))

        # Sort by total_score descending
        assembled.sort(key=lambda a: -a.total_score)
        result = assembled[:top_k]

        ms = (__import__("time").perf_counter() - t0) * 1000
        logger.info(f"InspirationScorer assembled {len(result)} outfits in {ms:.1f}ms")

        return result

    def _passes_assembly_vetoes(self, garments: List[Garment], body_shape: str) -> bool:
        """Check assembly-level hard vetoes."""
        vetoes = ASSEMBLY_VETOES.get(body_shape.lower(), [])
        if not vetoes:
            return True

        tops = [g for g in garments if g.attributes.category in
                {GarmentCategory.TOP, GarmentCategory.OUTERWEAR}]
        bottoms = [g for g in garments if g.attributes.category == GarmentCategory.BOTTOM]

        for top in tops:
            for bot in bottoms:
                for veto_fn in vetoes:
                    if veto_fn(top, bot):
                        logger.debug(f"Assembly veto: {veto_fn.__name__} rejected {top.id}+{bot.id}")
                        return False
        return True

    def _inspiration_fidelity(self, garments: List[Garment], dna: StyleDNA) -> float:
        """How well does the outfit reconstruct the inspiration palette + aesthetic?"""
        if not dna.palette and not dna.palette_hex:
            return 0.6

        # Palette coverage: how many DNA colors are represented
        covered = 0
        for p_name, p_hex in zip(dna.palette, dna.palette_hex + [""] * 10):
            best_sim = 0.0
            for g in garments:
                g_color = g.attributes.color.primary if g.attributes.color else ""
                g_hex = (g.attributes.color.hex_codes[0]
                         if g.attributes.color and g.attributes.color.hex_codes
                         else None)
                sim = _color_affinity(g_hex, g_color, [p_hex] if p_hex else [], [p_name])
                best_sim = max(best_sim, sim)
            if best_sim > 0.5:
                covered += 1

        palette_coverage = covered / max(1, len(dna.palette))

        # Aesthetic density
        aesthetic_hits = 0
        all_text = " ".join(
            f"{g.attributes.subcategory or ''} {g.attributes.material.primary if g.attributes.material and g.attributes.material.primary else ''}"
            for g in garments
        ).lower()
        for tag in dna.aesthetic:
            if tag.lower() in all_text:
                aesthetic_hits += 1
        aesthetic_density = aesthetic_hits / max(1, len(dna.aesthetic)) if dna.aesthetic else 0.5

        return palette_coverage * 0.55 + aesthetic_density * 0.45

    def _body_harmony(self, garments: List[Garment], body_shape: Optional[str]) -> float:
        """Average body fit score across all garments."""
        if not body_shape:
            return 0.7

        shape = body_shape.lower()
        rules = BODY_SHAPE_SLOT_RULES.get(shape, {})
        if not rules:
            return 0.7

        scores = []
        for g in garments:
            cat = g.attributes.category.value.lower()
            rule = rules.get(cat, {})
            if not rule:
                scores.append(0.7)
                continue

            sub = (g.attributes.subcategory or "").lower()
            fit = (g.attributes.fit or "").lower()
            combined = f"{sub} {fit}"

            s = 0.65
            for kw in rule.get("promote", []):
                if kw.lower() in combined:
                    s = min(1.0, s + 0.12)
            for kw in rule.get("avoid", []):
                if kw.lower() in combined:
                    s = max(0.1, s - 0.20)
            scores.append(s)

        return sum(scores) / len(scores) if scores else 0.7
