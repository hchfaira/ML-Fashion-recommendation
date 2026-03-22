"""
Prompt-to-Filters Translator — Layer 3 Context
================================================

Converts a :class:`ParsedPrompt` into concrete wardrobe filter criteria
that can be applied to garment/outfit data without any LLM call.

Responsibilities:
  • Expand color names into colour families (navy ⊂ blue)
  • Derive formality bounds from occasion when not explicit
  • Map style names to known system styles
  • Build hard-exclusion and soft-preference lists
  • Produce a :class:`WardrobeFilters` dataclass ready for downstream use

Public API:
    translator = PromptToFilters()
    filters = translator.translate(parsed_prompt)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.core import get_logger
from src.layer4_llm.prompt_parser import ParsedPrompt

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Colour family expansion
# ---------------------------------------------------------------------------

_COLOR_FAMILIES: Dict[str, List[str]] = {
    "blue": ["blue", "navy", "cobalt", "sky", "denim", "indigo", "teal", "turquoise", "cerulean", "periwinkle"],
    "red": ["red", "burgundy", "maroon", "crimson", "scarlet", "wine", "cherry"],
    "green": ["green", "olive", "emerald", "sage", "forest", "mint", "lime", "khaki"],
    "yellow": ["yellow", "mustard", "gold", "lemon", "canary"],
    "orange": ["orange", "rust", "burnt orange", "amber", "tangerine", "coral", "peach"],
    "pink": ["pink", "blush", "rose", "hot pink", "bubblegum", "salmon", "mauve", "fuchsia"],
    "purple": ["purple", "lavender", "lilac", "violet", "plum", "mauve"],
    "black": ["black", "charcoal", "onyx", "jet"],
    "white": ["white", "ivory", "cream", "off-white", "chalk"],
    "brown": ["brown", "camel", "tan", "beige", "nude", "coffee", "chocolate", "taupe"],
    "grey": ["grey", "gray", "silver", "ash", "stone", "slate"],
}

# Reverse map: specific → family
_COLOR_TO_FAMILY: Dict[str, str] = {
    shade: family
    for family, shades in _COLOR_FAMILIES.items()
    for shade in shades
}

# Occasion → compatible colour vibes (soft hints, not hard filters)
_OCCASION_COLOR_VIBES: Dict[str, List[str]] = {
    "wedding": ["blush", "ivory", "champagne", "gold", "navy", "dusty rose"],
    "gala": ["black", "navy", "burgundy", "silver", "gold"],
    "work": ["navy", "grey", "black", "white", "beige", "camel"],
    "date": ["black", "burgundy", "navy", "blush", "red"],
    "beach": ["white", "turquoise", "coral", "yellow", "sky"],
    "casual": [],  # no restriction
    "gym": ["black", "grey", "navy"],
    "funeral": ["black", "navy", "grey"],
}

# Style → associated garment attributes (soft hints)
_STYLE_GARMENT_HINTS: Dict[str, Dict] = {
    "minimalist": {"preferred_patterns": ["solid", "plain"], "avoid_patterns": ["floral", "print", "graphic"]},
    "bohemian": {"preferred_patterns": ["floral", "paisley", "ethnic"], "preferred_materials": ["linen", "cotton", "silk"]},
    "elegant": {"preferred_materials": ["silk", "satin", "chiffon", "velvet"], "avoid_patterns": ["graphic", "logo"]},
    "sporty": {"preferred_materials": ["cotton", "polyester", "spandex"], "preferred_patterns": ["solid", "stripe"]},
    "classic": {"preferred_patterns": ["solid", "stripe", "plaid"], "avoid_patterns": ["graphic", "logo", "tie-dye"]},
    "edgy": {"preferred_colors": ["black", "grey", "red"], "preferred_patterns": ["leather", "chain", "stud"]},
    "streetwear": {"preferred_patterns": ["graphic", "logo"], "preferred_materials": ["cotton", "denim"]},
}

# Semantic expansion for vague descriptors
_SEMANTIC_EXPANSION: Dict[str, Dict] = {
    "comfortable": {"formality_penalty": 0.2, "preferred_materials": ["cotton", "linen", "jersey"]},
    "flowy": {"preferred_silhouettes": ["a-line", "flare", "wide-leg"], "preferred_materials": ["chiffon", "silk", "linen"]},
    "powerful": {"preferred_patterns": ["solid"], "preferred_colors": ["black", "navy", "burgundy"]},
    "playful": {"preferred_patterns": ["floral", "print", "colorblock"]},
    "romantic": {"preferred_patterns": ["floral", "lace"], "preferred_materials": ["chiffon", "silk", "lace"]},
    "bold": {"preferred_patterns": ["colorblock", "print", "graphic"], "preferred_colors": ["red", "yellow", "cobalt"]},
    "confident": {"preferred_colors": ["black", "red", "navy"]},
}


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class WardrobeFilters:
    """
    Concrete wardrobe search criteria derived from a ParsedPrompt.

    Hard filters (must satisfy):
        excluded_types, formality_min, formality_max

    Soft filters (scored, not eliminated):
        target_colors, color_families, target_styles, target_occasion,
        preferred_patterns, preferred_materials, season_hint
    """
    # Hard filters
    excluded_types: List[str] = field(default_factory=list)
    formality_min: float = 0.0
    formality_max: float = 1.0

    # Soft colour filters
    target_colors: List[str] = field(default_factory=list)
    color_families: List[str] = field(default_factory=list)
    occasion_color_vibes: List[str] = field(default_factory=list)

    # Soft style filters
    target_styles: List[str] = field(default_factory=list)
    target_occasion: Optional[str] = None
    preferred_patterns: List[str] = field(default_factory=list)
    avoid_patterns: List[str] = field(default_factory=list)
    preferred_materials: List[str] = field(default_factory=list)

    # Season
    season_hint: Optional[str] = None

    # Anchor pieces (free-text descriptions)
    anchor_pieces: List[str] = field(default_factory=list)

    # Scoring weights (α, β, γ)
    weight_color: float = 0.35
    weight_style: float = 0.35
    weight_occasion: float = 0.30

    def to_dict(self) -> dict:
        return {
            "excluded_types": self.excluded_types,
            "formality_min": self.formality_min,
            "formality_max": self.formality_max,
            "target_colors": self.target_colors,
            "color_families": self.color_families,
            "occasion_color_vibes": self.occasion_color_vibes,
            "target_styles": self.target_styles,
            "target_occasion": self.target_occasion,
            "preferred_patterns": self.preferred_patterns,
            "avoid_patterns": self.avoid_patterns,
            "preferred_materials": self.preferred_materials,
            "season_hint": self.season_hint,
            "anchor_pieces": self.anchor_pieces,
            "weight_color": self.weight_color,
            "weight_style": self.weight_style,
            "weight_occasion": self.weight_occasion,
        }


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class PromptToFilters:
    """
    Translates a :class:`ParsedPrompt` into a :class:`WardrobeFilters`.

    All logic is deterministic (no LLM calls).
    """

    def translate(self, parsed: ParsedPrompt) -> WardrobeFilters:
        """Return a :class:`WardrobeFilters` from *parsed*."""
        if parsed.is_off_topic:
            logger.warning("PromptToFilters: off-topic prompt — returning empty filters")
            return WardrobeFilters()

        f = WardrobeFilters()

        # 1. Formality bounds
        f.formality_min, f.formality_max = parsed.formality_range

        # 2. Excluded garment types
        f.excluded_types = list(parsed.excluded_types)

        # 3. Color expansion
        f.target_colors = list(parsed.colors)
        f.color_families = self._expand_color_families(parsed.colors)

        # 4. Occasion
        f.target_occasion = parsed.occasion
        if parsed.occasion and parsed.occasion in _OCCASION_COLOR_VIBES:
            f.occasion_color_vibes = _OCCASION_COLOR_VIBES[parsed.occasion]

        # 5. Style hints
        f.target_styles = list(parsed.styles)
        for style in parsed.styles:
            hints = _STYLE_GARMENT_HINTS.get(style, {})
            f.preferred_patterns += hints.get("preferred_patterns", [])
            f.avoid_patterns += hints.get("avoid_patterns", [])
            f.preferred_materials += hints.get("preferred_materials", [])

        # 6. Mood / semantic expansion
        if parsed.mood and parsed.mood in _SEMANTIC_EXPANSION:
            expansion = _SEMANTIC_EXPANSION[parsed.mood]
            f.preferred_patterns += expansion.get("preferred_patterns", [])
            f.preferred_materials += expansion.get("preferred_materials", [])
            if "preferred_colors" in expansion:
                f.occasion_color_vibes += expansion["preferred_colors"]
            # Comfort → push formality_max down slightly
            if "formality_penalty" in expansion:
                f.formality_max = max(0.0, f.formality_max - expansion["formality_penalty"])

        # 7. Season
        f.season_hint = parsed.season_hint

        # 8. Anchor pieces
        f.anchor_pieces = list(parsed.anchor_pieces)

        # 9. Deduplicate lists
        f.preferred_patterns = _dedup(f.preferred_patterns)
        f.avoid_patterns = _dedup(f.avoid_patterns)
        f.preferred_materials = _dedup(f.preferred_materials)
        f.color_families = _dedup(f.color_families)
        f.occasion_color_vibes = _dedup(f.occasion_color_vibes)

        # 10. Adjust scoring weights based on what we know
        f.weight_color, f.weight_style, f.weight_occasion = self._compute_weights(parsed)

        logger.debug(
            "PromptToFilters: occasion=%s colors=%s formality=[%.2f,%.2f]",
            f.target_occasion, f.target_colors, f.formality_min, f.formality_max,
        )
        return f

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expand_color_families(self, colors: List[str]) -> List[str]:
        """Return the family names for each requested color (e.g. navy → blue)."""
        families: List[str] = []
        for color in colors:
            family = _COLOR_TO_FAMILY.get(color)
            if family and family not in families:
                families.append(family)
            elif color not in families:
                families.append(color)
        return families

    def get_color_shades(self, color: str) -> List[str]:
        """Return all shades belonging to the same color family as *color*."""
        family = _COLOR_TO_FAMILY.get(color, color)
        return _COLOR_FAMILIES.get(family, [color])

    def garment_matches_color_filter(
        self,
        garment_colors: List[str],
        filters: WardrobeFilters,
        strict: bool = False,
    ) -> float:
        """
        Return a score in [0, 1] for how well *garment_colors* matches the filter.

        strict=True → at least one color must match exactly or by family.
        strict=False → any overlap gives a non-zero score.
        """
        if not filters.target_colors and not filters.color_families:
            return 0.5  # No color constraint → neutral

        garment_lower = [c.lower() for c in garment_colors]
        score = 0.0

        # Exact match
        exact = sum(1 for c in garment_lower if c in filters.target_colors)
        if exact:
            score = min(1.0, exact * 0.6)

        # Family match
        garment_families = {_COLOR_TO_FAMILY.get(c, c) for c in garment_lower}
        family_hits = sum(1 for f in garment_families if f in filters.color_families)
        if family_hits:
            score = max(score, min(1.0, family_hits * 0.4))

        # Occasion vibe bonus
        vibe_hits = sum(1 for c in garment_lower if c in filters.occasion_color_vibes)
        if vibe_hits:
            score = max(score, 0.3)

        if strict and score == 0.0:
            return 0.0

        return score

    def garment_matches_style_filter(
        self,
        garment_styles: Dict[str, float],
        filters: WardrobeFilters,
    ) -> float:
        """Return a score in [0, 1] for style overlap."""
        if not filters.target_styles:
            return 0.5  # Neutral

        total = 0.0
        for style in filters.target_styles:
            total += garment_styles.get(style, 0.0)
        return min(1.0, total / max(len(filters.target_styles), 1))

    def garment_matches_formality(
        self, garment_formality: Optional[float], filters: WardrobeFilters
    ) -> bool:
        """Return True when garment formality is within the filter's range."""
        if garment_formality is None:
            return True  # Unknown formality → soft pass
        return filters.formality_min <= garment_formality <= filters.formality_max

    def _compute_weights(self, parsed: ParsedPrompt) -> Tuple[float, float, float]:
        """Return (w_color, w_style, w_occasion) summing to 1.0."""
        w_color = 0.35
        w_style = 0.35
        w_occasion = 0.30

        if parsed.colors and not parsed.occasion:
            w_color, w_style, w_occasion = 0.50, 0.30, 0.20
        elif parsed.occasion and not parsed.colors:
            w_color, w_style, w_occasion = 0.20, 0.30, 0.50
        elif parsed.occasion and parsed.colors:
            w_color, w_style, w_occasion = 0.35, 0.25, 0.40

        return w_color, w_style, w_occasion


def _dedup(items: List[str]) -> List[str]:
    seen: set = set()
    return [x for x in items if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]
