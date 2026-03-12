"""
Outfit Improver
Diagnoses outfit weaknesses and suggests improvements through
additions, replacements, and targeted purchases.

This module provides:
- Outfit diagnosis with weak/strong dimension identification
- Addition suggestions from existing wardrobe
- Replacement suggestions with expected score changes
- Purchase suggestions for ideal missing pieces
- Removal impact analysis for wardrobe planning
"""
from typing import List, Dict, Optional, Any, Tuple
from copy import deepcopy

from src.core.models import (
    Garment,
    GarmentCategory,
    FormalityLevel,
    Season,
    Occasion,
    UserContext,
    OutfitDiagnosis,
    AdditionSuggestion,
    ReplacementSuggestion,
    PurchaseTargeted,
    OutfitImprovementResult,
    RemovalImpact,
)
from src.core import get_logger
from .outfit_builder import OutfitBuilder, OutfitCandidate
from .outfit_scorecard import OutfitScorecard

logger = get_logger(__name__)

# Score thresholds
WEAK_THRESHOLD = 0.5
STRONG_THRESHOLD = 0.7

# Score dimension display names
DIMENSION_NAMES = {
    "seven_point": "Seven-Point Rule",
    "color_harmony": "Color Harmony",
    "three_color": "Three-Color Rule",
    "proportion": "Proportions",
    "volume_balance": "Volume Balance",
    "pattern_mixing": "Pattern Mixing",
    "design_principles": "Design Principles",
    "creativity": "Creativity",
    "total_style": "Total Style",
}

# Which categories can be added to an outfit without conflict
ADDABLE_CATEGORIES = {
    GarmentCategory.ACCESSORY,
    GarmentCategory.BAG,
    GarmentCategory.OUTERWEAR,
}

# Formality ordering for comparison
FORMALITY_ORDER = {
    FormalityLevel.VERY_CASUAL: 1,
    FormalityLevel.CASUAL: 2,
    FormalityLevel.SMART_CASUAL: 3,
    FormalityLevel.BUSINESS_CASUAL: 4,
    FormalityLevel.BUSINESS: 5,
    FormalityLevel.FORMAL: 6,
    FormalityLevel.BLACK_TIE: 7,
}


class OutfitImprover:
    """
    Analyzes outfit weaknesses and suggests improvements.

    Provides actionable recommendations to improve outfit scores
    by leveraging the existing wardrobe or suggesting purchases.

    Usage:
        improver = OutfitImprover()
        result = improver.improve(garments, wardrobe, context)
    """

    def __init__(self):
        self._builder = OutfitBuilder()

    # ==================== Main Entry Point ====================

    def improve(
        self,
        garments: List[Garment],
        wardrobe: List[Garment],
        context: Optional[UserContext] = None,
        profile: Optional[str] = None,
    ) -> OutfitImprovementResult:
        """
        Perform a complete outfit improvement analysis.

        Args:
            garments: The outfit to improve (list of garments)
            wardrobe: Full wardrobe for finding alternatives
            context: Optional user context for personalized scoring
            profile: Scoring profile to use

        Returns:
            OutfitImprovementResult with diagnosis, additions, replacements, and purchases
        """
        if not garments:
            return OutfitImprovementResult(
                diagnosis=OutfitDiagnosis(
                    overall_score=0.0,
                    grade="F",
                    improvement_potential=1.0,
                ),
                summary="No garments provided for analysis.",
            )

        logger.info(f"Improving outfit with {len(garments)} garments from wardrobe of {len(wardrobe)}")

        # 1. Score the current outfit
        scorecard = self._score_outfit(garments, context, profile)

        # 2. Diagnose
        diagnosis = self.diagnose_outfit(scorecard)

        # 3. Suggest additions from wardrobe
        additions = self.suggest_additions(garments, wardrobe, context, profile)

        # 4. Suggest replacements from wardrobe
        replacements = self.suggest_replacements(garments, wardrobe, context, profile)

        # 5. Suggest targeted purchases
        purchases = self.suggest_purchases(garments, scorecard, context)

        # 6. Summary
        summary = self._generate_improvement_summary(diagnosis, additions, replacements, purchases)

        return OutfitImprovementResult(
            diagnosis=diagnosis,
            additions=additions,
            replacements=replacements,
            purchase_suggestions=purchases,
            summary=summary,
        )

    # ==================== Diagnosis ====================

    def diagnose_outfit(
        self,
        scorecard: OutfitScorecard,
    ) -> OutfitDiagnosis:
        """
        Diagnose an outfit's weak and strong dimensions.

        Args:
            scorecard: Pre-calculated OutfitScorecard

        Returns:
            OutfitDiagnosis with weak/strong dimensions and improvement potential
        """
        overall = scorecard.scores.get("overall", 0.0)
        grade = scorecard.get_grade(overall)

        weak_dims: List[Dict[str, Any]] = []
        strong_dims: List[Dict[str, Any]] = []

        for dim, score in scorecard.scores.items():
            if dim in ("overall", "total_style"):
                continue
            display_name = DIMENSION_NAMES.get(dim, dim.replace("_", " ").title())
            entry = {
                "dimension": dim,
                "display_name": display_name,
                "score": round(score, 3),
                "grade": scorecard.get_grade(score),
            }
            if score < WEAK_THRESHOLD:
                weak_dims.append(entry)
            elif score >= STRONG_THRESHOLD:
                strong_dims.append(entry)

        # Sort: weakest first
        weak_dims.sort(key=lambda x: x["score"])
        strong_dims.sort(key=lambda x: x["score"], reverse=True)

        # Improvement potential: how much room between current and perfect
        improvement_potential = max(0.0, min(1.0, 1.0 - overall))

        return OutfitDiagnosis(
            overall_score=round(overall, 3),
            grade=grade,
            weak_dimensions=weak_dims,
            strong_dimensions=strong_dims,
            improvement_potential=round(improvement_potential, 3),
        )

    # ==================== Addition Suggestions ====================

    def suggest_additions(
        self,
        garments: List[Garment],
        wardrobe: List[Garment],
        context: Optional[UserContext] = None,
        profile: Optional[str] = None,
        max_suggestions: int = 3,
    ) -> List[AdditionSuggestion]:
        """
        Suggest garments from the wardrobe to add to the outfit.

        Only considers categories that won't conflict (accessories, bags, outerwear).

        Args:
            garments: Current outfit garments
            wardrobe: Full wardrobe
            context: Optional user context
            profile: Scoring profile
            max_suggestions: Maximum suggestions to return

        Returns:
            List of AdditionSuggestion sorted by expected score change (descending)
        """
        outfit_ids = {g.id for g in garments}
        outfit_cats = {g.attributes.category for g in garments}

        # Candidates: wardrobe items not already in outfit, in addable categories
        candidates = [
            g for g in wardrobe
            if g.id not in outfit_ids and g.attributes.category in ADDABLE_CATEGORIES
        ]

        # Also allow categories not yet in outfit
        for g in wardrobe:
            if g.id not in outfit_ids and g.attributes.category not in outfit_cats:
                if g not in candidates:
                    candidates.append(g)

        if not candidates:
            return []

        # Score base outfit
        base_score = self._quick_score(garments, context, profile)

        # Try each candidate
        suggestions: List[AdditionSuggestion] = []
        for candidate in candidates:
            new_garments = garments + [candidate]
            new_score = self._quick_score(new_garments, context, profile)
            delta = new_score - base_score

            if delta > -0.05:  # Only suggest if not significantly worse
                desc = self._garment_description(candidate)
                reason = self._addition_reason(candidate, delta)
                suggestions.append(AdditionSuggestion(
                    garment_id=candidate.id,
                    garment_description=desc,
                    expected_score_change=round(delta, 3),
                    reason=reason,
                ))

        suggestions.sort(key=lambda x: x.expected_score_change, reverse=True)
        return suggestions[:max_suggestions]

    # ==================== Replacement Suggestions ====================

    def suggest_replacements(
        self,
        garments: List[Garment],
        wardrobe: List[Garment],
        context: Optional[UserContext] = None,
        profile: Optional[str] = None,
        max_suggestions: int = 3,
    ) -> List[ReplacementSuggestion]:
        """
        Suggest replacing outfit garments with better alternatives from the wardrobe.

        Args:
            garments: Current outfit garments
            wardrobe: Full wardrobe
            context: Optional user context
            profile: Scoring profile
            max_suggestions: Maximum suggestions to return

        Returns:
            List of ReplacementSuggestion sorted by expected score change (descending)
        """
        outfit_ids = {g.id for g in garments}
        base_score = self._quick_score(garments, context, profile)

        suggestions: List[ReplacementSuggestion] = []

        for i, original in enumerate(garments):
            # Find same-category alternatives in wardrobe
            alternatives = [
                g for g in wardrobe
                if g.id not in outfit_ids
                and g.attributes.category == original.attributes.category
                and g.id != original.id
            ]

            for alt in alternatives:
                # Build new outfit with replacement
                new_garments = garments[:i] + [alt] + garments[i + 1:]
                new_score = self._quick_score(new_garments, context, profile)
                delta = new_score - base_score

                if delta > 0.01:  # Only suggest if improvement > 1%
                    suggestions.append(ReplacementSuggestion(
                        original_garment_id=original.id,
                        original_description=self._garment_description(original),
                        replacement_garment_id=alt.id,
                        replacement_description=self._garment_description(alt),
                        expected_score_change=round(delta, 3),
                        reason=self._replacement_reason(original, alt, delta),
                    ))

        suggestions.sort(key=lambda x: x.expected_score_change, reverse=True)
        return suggestions[:max_suggestions]

    # ==================== Purchase Suggestions ====================

    def suggest_purchases(
        self,
        garments: List[Garment],
        scorecard: OutfitScorecard,
        context: Optional[UserContext] = None,
        max_suggestions: int = 3,
    ) -> List[PurchaseTargeted]:
        """
        Suggest ideal garments to purchase to improve this outfit.

        Based on weak dimensions in the scorecard, recommend attributes
        for garments that would address the weaknesses.

        Args:
            garments: Current outfit garments
            scorecard: Pre-calculated scorecard
            context: Optional user context
            max_suggestions: Maximum suggestions

        Returns:
            List of PurchaseTargeted sorted by expected impact
        """
        suggestions: List[PurchaseTargeted] = []
        outfit_cats = {g.attributes.category for g in garments}

        for dim, score in scorecard.scores.items():
            if dim in ("overall", "total_style"):
                continue
            if score >= WEAK_THRESHOLD:
                continue

            purchase = self._purchase_for_dimension(dim, score, garments, outfit_cats, context)
            if purchase:
                suggestions.append(purchase)

        # Sort by expected impact (descending)
        suggestions.sort(key=lambda x: x.expected_score_change, reverse=True)
        return suggestions[:max_suggestions]

    # ==================== Removal Impact ====================

    def analyze_removal_impact(
        self,
        garment_id: str,
        wardrobe: List[Garment],
        context: Optional[UserContext] = None,
        profile: Optional[str] = None,
    ) -> RemovalImpact:
        """
        Analyze the impact of removing a garment from the wardrobe.

        Args:
            garment_id: ID of the garment to simulate removing
            wardrobe: Full wardrobe
            context: Optional user context
            profile: Scoring profile

        Returns:
            RemovalImpact with affected outfits and replacement info
        """
        target = None
        for g in wardrobe:
            if g.id == garment_id:
                target = g
                break

        if target is None:
            return RemovalImpact(
                garment_id=garment_id,
                garment_description="Unknown garment",
                garment_attributes=None,
                outfits_affected=0,
                versatility_score=0.0,
                replacement_available=False,
                is_critical=False,
                has_replacement=False,
                impact_level="low",
                summary=f"Garment {garment_id} not found in wardrobe.",
            )

        desc = self._garment_description(target)
        target_cat = target.attributes.category

        # Count same-category alternatives
        same_cat = [
            g for g in wardrobe
            if g.attributes.category == target_cat and g.id != garment_id
        ]
        replacement_available = len(same_cat) > 0

        # Estimate affected outfits — count compatible garments in other categories
        other_cats = {}
        for g in wardrobe:
            if g.id != garment_id:
                cat = g.attributes.category.value
                other_cats.setdefault(cat, []).append(g)

        # Rough estimate of outfits this garment participates in
        complementary_counts = []
        for cat, items in other_cats.items():
            if cat != target_cat.value:
                complementary_counts.append(len(items))

        outfits_affected = 1
        for c in complementary_counts[:2]:  # top 2 complementary categories
            outfits_affected *= max(1, c)
        outfits_affected = min(outfits_affected, 50)

        # Versatility
        wardrobe_by_cat: Dict[str, List[Garment]] = {}
        for g in wardrobe:
            cat = g.attributes.category.value
            wardrobe_by_cat.setdefault(cat, []).append(g)

        from .wardrobe_analyzer import WardrobeAnalyzer
        analyzer = WardrobeAnalyzer()
        versatility_score, _, _, _ = analyzer._score_garment_versatility(
            target, wardrobe, wardrobe_by_cat
        )

        # Replacement suggestions
        replacement_descs = [self._garment_description(g) for g in same_cat[:3]]

        # Impact level
        if not replacement_available and versatility_score > 0.5:
            impact_level = "critical"
        elif not replacement_available:
            impact_level = "high"
        elif versatility_score > 0.6:
            impact_level = "medium"
        else:
            impact_level = "low"

        summary_parts = [f"Removing '{desc}' would affect ~{outfits_affected} outfit(s)."]
        if replacement_available:
            summary_parts.append(f"{len(same_cat)} replacement(s) available in the same category.")
        else:
            summary_parts.append("No direct replacement available — consider purchasing a substitute.")
        summary_parts.append(f"Impact level: {impact_level}.")

        return RemovalImpact(
            garment_id=garment_id,
            garment_description=desc,
            garment_attributes=target.attributes,
            outfits_affected=outfits_affected,
            versatility_score=round(versatility_score, 2),
            replacement_available=replacement_available,
            is_critical=impact_level == "critical",
            has_replacement=replacement_available,
            replacement_suggestions=replacement_descs,
            impact_level=impact_level,
            summary=" ".join(summary_parts),
        )

    # ==================== Private Helpers ====================

    def _score_outfit(
        self,
        garments: List[Garment],
        context: Optional[UserContext] = None,
        profile: Optional[str] = None,
    ) -> OutfitScorecard:
        """Score an outfit using OutfitScorecard."""
        scorecard = OutfitScorecard(garments=garments, profile=profile)
        scorecard.calculate_all_scores(context)
        return scorecard

    def _quick_score(
        self,
        garments: List[Garment],
        context: Optional[UserContext] = None,
        profile: Optional[str] = None,
    ) -> float:
        """Get just the overall score for an outfit."""
        scorecard = self._score_outfit(garments, context, profile)
        return scorecard.scores.get("overall", 0.0)

    def _garment_description(self, garment: Garment) -> str:
        """Generate a human-readable garment description."""
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

    def _addition_reason(self, garment: Garment, delta: float) -> str:
        """Generate a reason for an addition suggestion."""
        cat = garment.attributes.category.value
        if delta > 0.05:
            return f"Adding this {cat} significantly improves the outfit score (+{delta:.1%})"
        elif delta > 0:
            return f"This {cat} slightly improves outfit harmony (+{delta:.1%})"
        else:
            return f"This {cat} can be added without reducing the outfit quality"

    def _replacement_reason(
        self, original: Garment, replacement: Garment, delta: float
    ) -> str:
        """Generate a reason for a replacement suggestion."""
        orig_color = original.attributes.color.primary if original.attributes.color else "?"
        repl_color = replacement.attributes.color.primary if replacement.attributes.color else "?"

        parts = [f"Replacing with this piece improves score by +{delta:.1%}."]

        if orig_color != repl_color:
            parts.append(f"Color change: {orig_color} → {repl_color} provides better harmony.")

        orig_form = original.attributes.formality_level
        repl_form = replacement.attributes.formality_level
        if orig_form != repl_form:
            parts.append(f"Formality adjustment: {orig_form.value} → {repl_form.value}.")

        return " ".join(parts)

    def _purchase_for_dimension(
        self,
        dimension: str,
        current_score: float,
        garments: List[Garment],
        outfit_cats: set,
        context: Optional[UserContext] = None,
    ) -> Optional[PurchaseTargeted]:
        """Generate a purchase suggestion for a weak dimension."""
        delta = max(0.05, WEAK_THRESHOLD - current_score)

        if dimension == "color_harmony":
            # Suggest a garment in a complementary color
            existing_colors = [
                g.attributes.color.primary.lower()
                for g in garments
                if g.attributes.color and g.attributes.color.primary
            ]
            return PurchaseTargeted(
                category="accessory",
                description="An accessory in a complementary color to improve color harmony",
                reason=f"Color harmony is low ({current_score:.0%}). A well-chosen accent piece can tie the palette together.",
                expected_score_change=round(delta, 3),
                suggested_attributes={
                    "avoid_colors": existing_colors[:3],
                    "prefer": "complementary or analogous color",
                },
            )

        elif dimension == "proportion":
            return PurchaseTargeted(
                category="bottom" if GarmentCategory.BOTTOM not in outfit_cats else "top",
                description="A garment with better proportional balance",
                reason=f"Proportions score is low ({current_score:.0%}). Aim for a 1/3-2/3 or golden ratio silhouette.",
                expected_score_change=round(delta, 3),
                suggested_attributes={
                    "fit": "tailored",
                    "silhouette": "structured",
                },
            )

        elif dimension == "volume_balance":
            return PurchaseTargeted(
                category="top" if GarmentCategory.TOP in outfit_cats else "bottom",
                description="A garment that balances volume with existing pieces",
                reason=f"Volume balance is low ({current_score:.0%}). Pair voluminous pieces with fitted ones.",
                expected_score_change=round(delta, 3),
                suggested_attributes={
                    "volume": "contrasting to current pieces",
                },
            )

        elif dimension == "seven_point":
            return PurchaseTargeted(
                category="accessory",
                description="An accessory to adjust the outfit's seven-point count",
                reason=f"Seven-point score is low ({current_score:.0%}). Accessories can fine-tune the complexity level.",
                expected_score_change=round(delta, 3),
                suggested_attributes={
                    "type": "belt, watch, or simple jewelry",
                },
            )

        elif dimension == "three_color":
            return PurchaseTargeted(
                category="accessory",
                description="A neutral-toned accessory to simplify the color scheme",
                reason=f"Three-color rule score is low ({current_score:.0%}). Reduce the number of distinct colors.",
                expected_score_change=round(delta, 3),
                suggested_attributes={
                    "color": "neutral (black, white, grey, navy)",
                },
            )

        elif dimension == "pattern_mixing":
            return PurchaseTargeted(
                category="top" if GarmentCategory.TOP in outfit_cats else "bottom",
                description="A solid or subtly patterned piece to balance pattern mixing",
                reason=f"Pattern mixing score is low ({current_score:.0%}). Replace a clashing pattern with a solid.",
                expected_score_change=round(delta, 3),
                suggested_attributes={
                    "pattern": "solid or micro pattern",
                },
            )

        elif dimension == "design_principles":
            return PurchaseTargeted(
                category="top",
                description="A well-structured piece that follows design fundamentals",
                reason=f"Design principles score is low ({current_score:.0%}). Consider pieces with clean lines and balanced structure.",
                expected_score_change=round(delta, 3),
                suggested_attributes={
                    "structure": "structured or semi-structured",
                    "fit": "tailored",
                },
            )

        return None

    def _generate_improvement_summary(
        self,
        diagnosis: OutfitDiagnosis,
        additions: List[AdditionSuggestion],
        replacements: List[ReplacementSuggestion],
        purchases: List[PurchaseTargeted],
    ) -> str:
        """Generate a human-readable improvement summary."""
        parts = [f"Outfit score: {diagnosis.overall_score:.0%} (Grade: {diagnosis.grade})."]

        if diagnosis.weak_dimensions:
            weak_names = [d["display_name"] for d in diagnosis.weak_dimensions[:3]]
            parts.append(f"Weak areas: {', '.join(weak_names)}.")

        if diagnosis.strong_dimensions:
            strong_names = [d["display_name"] for d in diagnosis.strong_dimensions[:2]]
            parts.append(f"Strengths: {', '.join(strong_names)}.")

        if additions:
            best = additions[0]
            parts.append(
                f"Best addition: {best.garment_description} (+{best.expected_score_change:.1%})."
            )

        if replacements:
            best = replacements[0]
            parts.append(
                f"Best replacement: swap '{best.original_description}' "
                f"with '{best.replacement_description}' (+{best.expected_score_change:.1%})."
            )

        if purchases:
            parts.append(f"{len(purchases)} purchase suggestion(s) to address weak dimensions.")

        return " ".join(parts)
