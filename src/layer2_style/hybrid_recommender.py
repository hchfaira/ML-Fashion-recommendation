"""
Hybrid Outfit Recommender
==========================

Combines StyleIntelligenceModel (Layer 2) and ContextEngine (Layer 3)
into a single unified recommendation pipeline.

Architecture:
  ┌─────────────────────────┐
  │   StyleIntelligenceModel │  → Pure style score  (aesthetic rules)
  │   (Layer 2 — 40%)        │    7-Point Rule, Color Harmony,
  │                          │    Proportions, Volume Balance
  └──────────┬───────────────┘
             │ combined
  ┌──────────▼───────────────┐
  │   ContextEngine           │  → Context score  (situational fit)
  │   (Layer 3 — 60%)         │    Weather, Occasion, Morphology,
  │                           │    Freshness, Activity, …
  └───────────────────────────┘

The two scores are blended into a single ``combined_score`` using
configurable weights (default: style 40 %, context 60 %).

Five integration points with the user profile:

  1. **Persistent storage** — ``set_user_profile()`` stores the profile and
     subsequent calls to ``score_outfit()`` / ``recommend()`` reuse it
     automatically (no need to pass ``user_season`` / ``body_shape`` again).

  2. **Context enrichment** — the stored profile is injected into the
     ``UserContext`` before it reaches the ContextEngine so that
     MorphologyAdvisor, ColorHarmonyAdvisor, etc. receive a rich context
     without the caller having to build it by hand.

  3. **Adaptive weights** — when the profile is *complete* (both colour
     season **and** body shape provided) the context weight is bumped
     because the engine has more data to personalise.  When the profile is
     empty the split stays balanced.

  4. **Pre-scoring garment filtering** — before generating combinations
     the morphology advisor eliminates garments flagged as ``avoid`` for the
     stored body shape, reducing noise and speeding up scoring.

  5. **Personalised explanations** — ``strengths`` and ``improvements`` in
     ``HybridScore`` reference the user's colour season and body shape so
     the text is relevant to *them* specifically.

Public API
----------
    recommender = HybridOutfitRecommender()
    # optionally personalise
    recommender.set_user_profile(color_season=ColorSeason.WINTER,
                                 body_shape=BodyShape.HOURGLASS)
    # score a single outfit
    result = await recommender.score_outfit(garments, user_context)
    # rank a whole wardrobe
    ranked = await recommender.recommend(wardrobe, user_context, top_k=5)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.core.models import Garment, Outfit, OutfitItem, UserContext
from src.core import get_logger
from src.layer2_style.style_model import StyleIntelligenceModel
from src.layer2_style.season_color_harmony import ColorSeason
from src.layer2_style.volume_balance_scorer import BodyShape
from src.layer3_context import ContextEngine
from src.layer3_context.context_engine import ContextCriteria
from src.layer3_context.morphology_advisor import MorphologyAdvisor

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Adaptive weight presets  (Point 3)
# ---------------------------------------------------------------------------
#   profile_completeness → (style_weight, context_weight)
_ADAPTIVE_WEIGHTS: Dict[str, Tuple[float, float]] = {
    "full":    (0.35, 0.65),   # both season + body shape known
    "partial": (0.42, 0.58),   # only one of the two known
    "none":    (0.50, 0.50),   # no profile at all
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HybridScore:
    """Complete scoring result from the hybrid recommender."""

    # Raw sub-scores
    style_score: float          # StyleIntelligenceModel advanced score  [0-1]
    context_score: float        # ContextEngine weighted score           [0-1]
    combined_score: float       # Weighted blend of the above            [0-1]

    # Human-readable grade (A / B / C / D / F)
    grade: str = "C"

    # Breakdown dicts forwarded from each engine
    style_breakdown: Dict[str, Any] = field(default_factory=dict)
    context_breakdown: Dict[str, Any] = field(default_factory=dict)

    # Actionable text
    strengths: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)

    # Weights actually used
    style_weight: float = 0.40
    context_weight: float = 0.60

    # Profile metadata (Point 5 — so the caller knows what was used)
    profile_used: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "combined_score": round(self.combined_score, 3),
            "grade": self.grade,
            "style_score": round(self.style_score, 3),
            "context_score": round(self.context_score, 3),
            "weights": {
                "style": self.style_weight,
                "context": self.context_weight,
            },
            "style_breakdown": self.style_breakdown,
            "context_breakdown": self.context_breakdown,
            "strengths": self.strengths,
            "improvements": self.improvements,
            "profile_used": self.profile_used,
        }


@dataclass
class RankedOutfit:
    """An outfit candidate with its full hybrid score."""

    garments: List[Garment]
    score: HybridScore
    outfit_id: str = ""

    @property
    def combined_score(self) -> float:
        return self.score.combined_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outfit_id": self.outfit_id,
            "garments": [g.id for g in self.garments],
            **self.score.to_dict(),
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HybridOutfitRecommender:
    """
    Orchestrates StyleIntelligenceModel + ContextEngine to produce
    personalized, context-aware outfit recommendations.

    Parameters
    ----------
    style_weight : float
        Weight given to StyleIntelligenceModel score (default 0.40).
    context_weight : float
        Weight given to ContextEngine score (default 0.60).
        ``style_weight + context_weight`` must equal 1.0.
    context_criteria : list[ContextCriteria] | None
        Criteria passed to ContextEngine (uses engine defaults if None).
    context_weights : dict[ContextCriteria, float] | None
        Per-criterion weights for ContextEngine (uses engine defaults if None).
    adaptive_weights : bool
        If True (default), the effective style/context split is automatically
        adjusted based on profile completeness.  The constructor values are
        used as the *base* and may be shifted.
    """

    def __init__(
        self,
        style_weight: float = 0.40,
        context_weight: float = 0.60,
        context_criteria: Optional[List[ContextCriteria]] = None,
        context_weights: Optional[Dict[ContextCriteria, float]] = None,
        adaptive_weights: bool = True,
    ) -> None:
        if abs(style_weight + context_weight - 1.0) > 1e-6:
            raise ValueError(
                f"style_weight + context_weight must equal 1.0, "
                f"got {style_weight + context_weight}"
            )

        # Base weights (may be overridden by adaptive logic)
        self._base_style_weight = style_weight
        self._base_context_weight = context_weight
        self.style_weight = style_weight
        self.context_weight = context_weight
        self._adaptive_weights = adaptive_weights

        # ── Point 1 — persistent profile storage ──────────────────────
        self._user_color_season: Optional[ColorSeason] = None
        self._user_body_shape: Optional[BodyShape] = None

        # Layer 2 — pure style engine
        self.style_model = StyleIntelligenceModel()

        # Layer 3 — context engine
        kwargs: Dict[str, Any] = {}
        if context_criteria is not None:
            kwargs["criteria"] = context_criteria
        if context_weights is not None:
            kwargs["weights"] = context_weights
        self.context_engine = ContextEngine(**kwargs)

        # Layer 3 — morphology advisor (used for Point 4 filtering)
        self._morphology_advisor = MorphologyAdvisor()

        logger.info(
            "HybridOutfitRecommender ready "
            f"(style={style_weight:.0%}, context={context_weight:.0%}, "
            f"adaptive={adaptive_weights})"
        )

    # ------------------------------------------------------------------
    # Point 1 — Public API : persistent profile storage
    # ------------------------------------------------------------------

    def set_user_profile(
        self,
        color_season: Optional[ColorSeason] = None,
        body_shape: Optional[BodyShape] = None,
    ) -> None:
        """
        Personalise both engines with the user's physical attributes.

        The values are **stored** so that subsequent calls to
        ``score_outfit()`` and ``recommend()`` reuse them automatically.

        - ``color_season`` influences Season Color Harmony in Layer 2.
        - ``body_shape`` influences Volume Balance in Layer 2 **and**
          pre-scoring filtering + personalised explanations.
        """
        self._user_color_season = color_season
        self._user_body_shape = body_shape

        # Forward to StyleIntelligenceModel (Layer 2)
        self.style_model.set_user_profile(
            color_season=color_season,
            body_shape=body_shape,
        )

        # ── Point 3 — adapt weights based on profile completeness ──
        if self._adaptive_weights:
            self._update_adaptive_weights()

        logger.debug(
            f"User profile set — season={color_season}, shape={body_shape} "
            f"→ weights: style={self.style_weight:.0%}, context={self.context_weight:.0%}"
        )

    @property
    def profile_completeness(self) -> str:
        """Return 'full', 'partial' or 'none'."""
        has_season = self._user_color_season is not None
        has_shape = self._user_body_shape is not None
        if has_season and has_shape:
            return "full"
        if has_season or has_shape:
            return "partial"
        return "none"

    # ------------------------------------------------------------------
    # Point 3 — adaptive weight adjustment
    # ------------------------------------------------------------------

    def _update_adaptive_weights(self) -> None:
        """Recalculate effective weights from profile completeness."""
        preset = _ADAPTIVE_WEIGHTS.get(self.profile_completeness, (0.40, 0.60))
        self.style_weight, self.context_weight = preset
        logger.debug(
            f"Adaptive weights ({self.profile_completeness}): "
            f"style={self.style_weight:.0%}, context={self.context_weight:.0%}"
        )

    # ------------------------------------------------------------------
    # Point 2 — enrich UserContext with stored profile
    # ------------------------------------------------------------------

    def _enrich_context(self, user_context: UserContext) -> UserContext:
        """
        Return a copy of *user_context* enriched with the stored profile.

        Only fills fields that are currently ``None`` / empty so that
        explicit caller values are never overwritten.
        """
        # Work on a copy to avoid mutating the caller's object
        ctx = user_context.model_copy()

        if self._user_body_shape and not ctx.body_type:
            ctx.body_type = self._user_body_shape.value

        if self._user_color_season:
            season_str = self._user_color_season.value if hasattr(
                self._user_color_season, "value"
            ) else str(self._user_color_season)
            if not ctx.color_season:
                ctx.color_season = season_str
            # Also derive undertone when missing
            if not ctx.skin_undertone:
                _SEASON_UNDERTONE = {
                    "spring": "warm",
                    "summer": "cool",
                    "autumn": "warm",
                    "winter": "cool",
                }
                ctx.skin_undertone = _SEASON_UNDERTONE.get(
                    season_str.lower(), "neutral"
                )

        if self._user_body_shape:
            shape_str = self._user_body_shape.value if hasattr(
                self._user_body_shape, "value"
            ) else str(self._user_body_shape)
            if not ctx.body_shape:
                ctx.body_shape = shape_str

        return ctx

    # ------------------------------------------------------------------
    # Point 4 — pre-scoring garment filtering
    # ------------------------------------------------------------------

    def _filter_garments_for_profile(
        self, garments: List[Garment]
    ) -> List[Garment]:
        """
        Remove garments flagged as ``avoid`` for the stored body shape.

        Returns the original list unchanged if no body shape is stored.
        """
        if not self._user_body_shape:
            return garments

        body_type_str = (
            self._user_body_shape.value
            if hasattr(self._user_body_shape, "value")
            else str(self._user_body_shape)
        )

        try:
            filtered = self._morphology_advisor.filter_by_body_type(
                garments, body_type_str
            )
            removed = len(garments) - len(filtered)
            if removed:
                logger.info(
                    f"Pre-filter: removed {removed}/{len(garments)} garments "
                    f"unsuitable for body shape '{body_type_str}'"
                )
            return filtered if filtered else garments  # never return empty
        except Exception as exc:
            logger.debug(f"Pre-filter skipped: {exc}")
            return garments

    # ------------------------------------------------------------------
    # Point 5 — personalised explanations
    # ------------------------------------------------------------------

    def _personalise_explanations(
        self,
        strengths: List[str],
        improvements: List[str],
        style_score: float,
        context_score: float,
    ) -> Tuple[List[str], List[str]]:
        """
        Rewrite generic strengths/improvements referencing the user's
        colour season and body shape so the text is relevant to *them*.
        """
        season = self._user_color_season
        shape = self._user_body_shape

        personal_strengths: List[str] = list(strengths)
        personal_improvements: List[str] = list(improvements)

        # --- colour season tips ---
        if season:
            season_val = season.value if hasattr(season, "value") else str(season)
            if context_score >= 0.7:
                personal_strengths.append(
                    f"Les couleurs de cette tenue conviennent à ta saison "
                    f"'{season_val}' — elles rehaussent ton teint naturel."
                )
            elif context_score < 0.5:
                personal_improvements.append(
                    f"Essaie des couleurs typiques de ta saison '{season_val}' "
                    f"pour mieux harmoniser la tenue avec ton teint."
                )

        # --- body shape tips ---
        if shape:
            shape_val = shape.value if hasattr(shape, "value") else str(shape)

            _SHAPE_FLAT = {
                "hourglass": "ta taille marquée",
                "pear": "le haut de ton corps",
                "apple": "tes jambes",
                "rectangle": "ta silhouette structurée",
                "inverted_triangle": "le bas de ton corps",
                "athletic": "ta silhouette athlétique",
            }
            highlight = _SHAPE_FLAT.get(shape_val.lower(), "tes proportions")

            if style_score >= 0.7:
                personal_strengths.append(
                    f"Les proportions de cette tenue mettent en valeur "
                    f"{highlight} (morphologie {shape_val})."
                )
            elif style_score < 0.5:
                personal_improvements.append(
                    f"Pour ta morphologie '{shape_val}', essaie de mettre "
                    f"en valeur {highlight} avec une silhouette plus adaptée."
                )

        # --- combined insight ---
        if season and shape:
            if style_score >= 0.7 and context_score >= 0.7:
                personal_strengths.append(
                    "Les deux moteurs sont d'accord : cette tenue allie "
                    "esthétique et adaptation à ton profil. ✨"
                )
            elif style_score > context_score + 0.15:
                personal_improvements.append(
                    "Cette tenue est belle esthétiquement mais pourrait "
                    "être mieux adaptée à ta morphologie et tes couleurs."
                )
            elif context_score > style_score + 0.15:
                personal_improvements.append(
                    "Cette tenue est bien adaptée à ton profil mais pourrait "
                    "gagner en créativité stylistique."
                )

        return personal_strengths, personal_improvements

    # ------------------------------------------------------------------
    # Public API — scoring
    # ------------------------------------------------------------------

    async def score_outfit(
        self,
        garments: List[Garment],
        user_context: UserContext,
        user_season: Optional[ColorSeason] = None,
        body_shape: Optional[BodyShape] = None,
    ) -> HybridScore:
        """
        Score a single outfit (list of garments) using both engines.

        Parameters
        ----------
        garments : list[Garment]
            The items forming the outfit.
        user_context : UserContext
            Situational context (occasion, weather, body_type, …).
        user_season : ColorSeason | None
            Override the color season stored in the user profile.
        body_shape : BodyShape | None
            Override the body shape stored in the user profile.

        Returns
        -------
        HybridScore
        """
        if len(garments) < 2:
            return HybridScore(
                style_score=0.5,
                context_score=0.5,
                combined_score=0.5,
                grade="C",
                improvements=["Need at least 2 garments for scoring."],
            )

        # ── Point 1 — use stored profile as defaults ──────────────────
        effective_season = user_season or self._user_color_season
        effective_shape = body_shape or self._user_body_shape

        # ── Point 2 — enrich context ──────────────────────────────────
        enriched_ctx = self._enrich_context(user_context)

        # --- Layer 2: StyleIntelligenceModel ----------------------------
        try:
            style_result = await self.style_model.score_outfit_advanced(
                garments,
                user_season=effective_season,
                body_shape=effective_shape,
            )
            style_score: float = style_result.get("overall_score", 0.5)
            style_breakdown: Dict[str, Any] = style_result.get("breakdown", {})
            strengths: List[str] = style_result.get("strengths", [])
            improvements: List[str] = style_result.get("improvements", [])
        except Exception as exc:
            logger.warning(f"StyleIntelligenceModel failed: {exc}")
            style_score = 0.5
            style_breakdown = {}
            strengths = []
            improvements = [f"Style scoring unavailable: {exc}"]

        # --- Layer 3: ContextEngine -------------------------------------
        outfit = self._build_outfit(garments, style_score)

        try:
            scored = await self.context_engine.apply_context(
                [outfit], enriched_ctx  # ← enriched, not raw
            )
            if scored:
                context_score: float = scored[0].overall_score
            else:
                context_score = 0.5

            context_breakdown: Dict[str, Any] = self._extract_context_breakdown(
                outfit, enriched_ctx
            )
        except Exception as exc:
            logger.warning(f"ContextEngine failed: {exc}")
            context_score = 0.5
            context_breakdown = {}
            improvements.append(f"Context scoring unavailable: {exc}")

        # ── Point 5 — personalise explanations ────────────────────────
        strengths, improvements = self._personalise_explanations(
            strengths, improvements, style_score, context_score
        )

        # --- Blend (Point 3 — weights may have been adapted) -----------
        combined = (
            self.style_weight * style_score
            + self.context_weight * context_score
        )
        combined = min(1.0, max(0.0, round(combined, 3)))

        grade = self._grade(combined)

        # Profile metadata for the caller
        profile_meta: Dict[str, Any] = {
            "color_season": (
                effective_season.value
                if effective_season and hasattr(effective_season, "value")
                else str(effective_season) if effective_season else None
            ),
            "body_shape": (
                effective_shape.value
                if effective_shape and hasattr(effective_shape, "value")
                else str(effective_shape) if effective_shape else None
            ),
            "completeness": self.profile_completeness,
            "adaptive_weights": self._adaptive_weights,
        }

        return HybridScore(
            style_score=round(style_score, 3),
            context_score=round(context_score, 3),
            combined_score=combined,
            grade=grade,
            style_breakdown=style_breakdown,
            context_breakdown=context_breakdown,
            strengths=strengths,
            improvements=improvements,
            style_weight=self.style_weight,
            context_weight=self.context_weight,
            profile_used=profile_meta,
        )

    # ------------------------------------------------------------------
    # Public API — recommendation
    # ------------------------------------------------------------------

    async def recommend(
        self,
        wardrobe: List[Garment],
        user_context: UserContext,
        top_k: int = 5,
        outfit_combinations: Optional[List[List[Garment]]] = None,
        user_season: Optional[ColorSeason] = None,
        body_shape: Optional[BodyShape] = None,
    ) -> List[RankedOutfit]:
        """
        Rank outfit combinations from a wardrobe.

        Parameters
        ----------
        wardrobe : list[Garment]
            All available garments.
        user_context : UserContext
            Situational context.
        top_k : int
            Number of top outfits to return.
        outfit_combinations : list[list[Garment]] | None
            Pre-built outfits.  If None the method generates simple
            top+bottom(+shoes) combinations from the wardrobe,
            **after** filtering out garments unsuitable for the stored
            body shape (Point 4).
        user_season : ColorSeason | None
        body_shape : BodyShape | None

        Returns
        -------
        list[RankedOutfit]  sorted best-first
        """
        if outfit_combinations is None:
            # ── Point 4 — pre-filter garments before combination ──────
            filtered_wardrobe = self._filter_garments_for_profile(wardrobe)
            outfit_combinations = self._generate_combinations(filtered_wardrobe)

        if not outfit_combinations:
            logger.warning("No outfit combinations available to rank.")
            return []

        # Score each combination concurrently
        tasks = [
            self.score_outfit(
                combo,
                user_context,
                user_season=user_season,
                body_shape=body_shape,
            )
            for combo in outfit_combinations
        ]
        scores: List[HybridScore] = await asyncio.gather(*tasks)

        ranked = [
            RankedOutfit(
                garments=combo,
                score=score,
                outfit_id=f"outfit_{i}",
            )
            for i, (combo, score) in enumerate(zip(outfit_combinations, scores))
        ]
        ranked.sort(key=lambda r: r.combined_score, reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_outfit(garments: List[Garment], initial_score: float) -> Outfit:
        """Wrap garments in an Outfit for ContextEngine."""
        from uuid import uuid4
        from src.core.models import GarmentCategory

        _CATEGORY_ROLE: dict = {
            GarmentCategory.TOP: "main_top",
            GarmentCategory.BOTTOM: "bottom",
            GarmentCategory.DRESS: "main_top",
            GarmentCategory.SHOES: "shoes",
            GarmentCategory.OUTERWEAR: "outerwear",
            GarmentCategory.ACCESSORY: "accessory",
            GarmentCategory.BAG: "accessory",
        }

        def _role(g: Garment) -> str:
            cat = getattr(g.attributes, "category", None)
            return _CATEGORY_ROLE.get(cat, "base_layer")

        items = [OutfitItem(garment=g, role=_role(g)) for g in garments]
        outfit = Outfit(
            id=str(uuid4()),
            items=items,
            compatibility_score=initial_score,
            style_coherence_score=initial_score,
            occasion_match_score=initial_score,
            overall_score=initial_score,
        )
        return outfit

    def _extract_context_breakdown(
        self,
        outfit: Outfit,
        context: UserContext,
    ) -> Dict[str, Any]:
        """Build a lightweight context breakdown dict for display purposes."""
        breakdown: Dict[str, Any] = {}
        if context.occasion:
            breakdown["occasion"] = context.occasion.value if hasattr(context.occasion, "value") else str(context.occasion)
        if context.weather:
            breakdown["weather"] = {
                "temperature_c": getattr(context.weather, "temperature_c", None),
                "condition": getattr(context.weather, "condition", None),
            }
        if context.body_type:
            breakdown["body_type"] = context.body_type
        if context.body_shape:
            breakdown["body_shape"] = context.body_shape
        if context.color_season:
            breakdown["color_season"] = context.color_season
        if context.skin_undertone:
            breakdown["skin_undertone"] = context.skin_undertone
        return breakdown

    @staticmethod
    def _generate_combinations(wardrobe: List[Garment]) -> List[List[Garment]]:
        """
        Generate basic top + bottom (+ optional shoes) combinations.

        Falls back to any 2-item combination when category info is missing.
        """
        from src.core.models import GarmentCategory

        tops = [g for g in wardrobe if getattr(g.attributes, "category", None) in (
            GarmentCategory.TOP, GarmentCategory.DRESS,
        )]
        bottoms = [g for g in wardrobe if getattr(g.attributes, "category", None) == GarmentCategory.BOTTOM]
        shoes = [g for g in wardrobe if getattr(g.attributes, "category", None) == GarmentCategory.SHOES]
        outerwear = [g for g in wardrobe if getattr(g.attributes, "category", None) == GarmentCategory.OUTERWEAR]

        combos: List[List[Garment]] = []

        if tops and bottoms:
            for t in tops:
                for b in bottoms:
                    combo: List[Garment] = [t, b]
                    if shoes:
                        combo.append(shoes[0])
                    if outerwear:
                        combo.append(outerwear[0])
                    combos.append(combo)
        else:
            # Fallback: pair every garment with every other garment
            for i, g1 in enumerate(wardrobe):
                for g2 in wardrobe[i + 1 :]:
                    combos.append([g1, g2])

        return combos

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 0.85:
            return "A"
        elif score >= 0.70:
            return "B"
        elif score >= 0.55:
            return "C"
        elif score >= 0.40:
            return "D"
        return "F"
