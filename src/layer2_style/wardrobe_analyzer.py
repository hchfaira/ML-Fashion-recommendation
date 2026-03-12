"""
Wardrobe Analyzer
Comprehensive analysis of a user's wardrobe for gap detection,
versatility scoring, and purchase recommendations.

This module provides:
- Distribution analysis across categories, colors, formality, seasons
- Gap detection and imbalance identification
- Occasion coverage analysis
- Garment versatility scoring
- What-if simulation for potential additions
- Prioritized purchase suggestions
"""
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter

from src.core.models import (
    Garment,
    GarmentCategory,
    GarmentAttributes,
    FormalityLevel,
    Season,
    Occasion,
    UserContext,
    ColorProfile,
    CategoryDistribution,
    WardrobeDistribution,
    WardrobeGap,
    OccasionCoverage,
    GarmentVersatility,
    PurchaseSuggestion,
    WardrobeAnalysisResult,
)
from src.core import get_logger
from .outfit_builder import OutfitBuilder, OutfitCandidate
from .outfit_scorecard import OutfitScorecard
from .color_harmony import ColorHarmonyAnalyzer

logger = get_logger(__name__)

# Ideal category ratios for a balanced wardrobe
IDEAL_CATEGORY_RATIOS = {
    GarmentCategory.TOP: 0.30,
    GarmentCategory.BOTTOM: 0.20,
    GarmentCategory.DRESS: 0.10,
    GarmentCategory.OUTERWEAR: 0.10,
    GarmentCategory.SHOES: 0.15,
    GarmentCategory.ACCESSORY: 0.10,
    GarmentCategory.BAG: 0.05,
}

# Minimum items per category for a functional wardrobe
MIN_CATEGORY_ITEMS = {
    GarmentCategory.TOP: 3,
    GarmentCategory.BOTTOM: 2,
    GarmentCategory.SHOES: 2,
    GarmentCategory.OUTERWEAR: 1,
}

# Occasion-to-formality mapping
OCCASION_FORMALITY = {
    Occasion.DAILY_WEAR: FormalityLevel.CASUAL,
    Occasion.WORK: FormalityLevel.BUSINESS_CASUAL,
    Occasion.BUSINESS: FormalityLevel.BUSINESS,
    Occasion.INTERVIEW: FormalityLevel.BUSINESS,
    Occasion.DATE: FormalityLevel.SMART_CASUAL,
    Occasion.COCKTAIL: FormalityLevel.SMART_CASUAL,
    Occasion.WEDDING: FormalityLevel.FORMAL,
    Occasion.FORMAL: FormalityLevel.FORMAL,
    Occasion.SPORT: FormalityLevel.VERY_CASUAL,
    Occasion.OUTDOOR: FormalityLevel.CASUAL,
    Occasion.TRAVEL: FormalityLevel.CASUAL,
    Occasion.BEACH: FormalityLevel.VERY_CASUAL,
    Occasion.CASUAL: FormalityLevel.CASUAL,
    Occasion.EVENING: FormalityLevel.SMART_CASUAL,
    Occasion.WEEKEND: FormalityLevel.CASUAL,
    Occasion.EVENT: FormalityLevel.FORMAL,
    Occasion.GYM: FormalityLevel.VERY_CASUAL,
}

# Required categories per occasion
OCCASION_REQUIRED_CATEGORIES = {
    Occasion.DAILY_WEAR: ["TOP", "BOTTOM", "SHOES"],
    Occasion.WORK: ["TOP", "BOTTOM", "SHOES"],
    Occasion.BUSINESS: ["TOP", "BOTTOM", "SHOES", "OUTERWEAR"],
    Occasion.DATE: ["TOP", "BOTTOM", "SHOES"],
    Occasion.COCKTAIL: ["TOP", "BOTTOM", "SHOES", "ACCESSORY"],
    Occasion.WEDDING: ["DRESS", "SHOES", "ACCESSORY"],
    Occasion.FORMAL: ["DRESS", "SHOES", "ACCESSORY"],
    Occasion.SPORT: ["TOP", "BOTTOM", "SHOES"],
    Occasion.OUTDOOR: ["TOP", "BOTTOM", "SHOES", "OUTERWEAR"],
    Occasion.TRAVEL: ["TOP", "BOTTOM", "SHOES", "OUTERWEAR"],
}

# Formality level numeric values for comparison
FORMALITY_ORDER = {
    FormalityLevel.VERY_CASUAL: 1,
    FormalityLevel.CASUAL: 2,
    FormalityLevel.SMART_CASUAL: 3,
    FormalityLevel.BUSINESS_CASUAL: 4,
    FormalityLevel.BUSINESS: 5,
    FormalityLevel.FORMAL: 6,
    FormalityLevel.BLACK_TIE: 7,
}


class WardrobeAnalyzer:
    """
    Comprehensive wardrobe analysis engine.

    Provides insights into wardrobe composition, identifies gaps,
    scores garment versatility, and generates purchase recommendations.

    Usage:
        analyzer = WardrobeAnalyzer()
        result = analyzer.analyze(wardrobe, user_context)
    """

    def __init__(self):
        self._color_analyzer = ColorHarmonyAnalyzer()

    # ==================== Main Entry Point ====================

    def analyze(
        self,
        wardrobe: List[Garment],
        context: Optional[UserContext] = None,
        occasions: Optional[List[Occasion]] = None,
        top_k_versatile: int = 5,
    ) -> WardrobeAnalysisResult:
        """
        Perform a complete wardrobe analysis.

        Args:
            wardrobe: All garments in the user's wardrobe
            context: Optional user context for personalized analysis
            occasions: Occasions to check coverage for (defaults to common ones)
            top_k_versatile: Number of top versatile items to return

        Returns:
            WardrobeAnalysisResult with distribution, gaps, coverage, and suggestions
        """
        if not wardrobe:
            return WardrobeAnalysisResult(
                distribution=WardrobeDistribution(),
                overall_score=0.0,
                summary="Empty wardrobe. Start building your collection!",
            )

        logger.info(f"Analyzing wardrobe with {len(wardrobe)} items")

        # 1. Distribution analysis
        distribution = self.analyze_distribution(wardrobe)

        # 2. Gap detection
        gaps = self.detect_gaps(wardrobe, context)

        # 3. Occasion coverage
        if occasions is None:
            occasions = [Occasion.DAILY_WEAR, Occasion.WORK, Occasion.DATE, Occasion.COCKTAIL, Occasion.FORMAL]
        occasion_coverage = [
            self.analyze_occasion_coverage(wardrobe, occ)
            for occ in occasions
        ]

        # 4. Versatility scoring
        top_versatile = self.calculate_versatility(wardrobe, top_k=top_k_versatile)

        # 5. Purchase suggestions
        purchase_suggestions = self.generate_purchase_suggestions(wardrobe, context, gaps, occasion_coverage)

        # 6. Overall score
        overall_score = self._calculate_overall_health(distribution, gaps, occasion_coverage)

        # 7. Summary
        summary = self._generate_summary(distribution, gaps, occasion_coverage, overall_score)

        return WardrobeAnalysisResult(
            distribution=distribution,
            gaps=gaps,
            occasion_coverage=occasion_coverage,
            top_versatile_items=top_versatile,
            purchase_suggestions=purchase_suggestions,
            overall_score=overall_score,
            summary=summary,
        )

    # ==================== Distribution Analysis ====================

    def analyze_distribution(self, wardrobe: List[Garment]) -> WardrobeDistribution:
        """
        Analyze the distribution of wardrobe items across multiple dimensions.

        Args:
            wardrobe: All garments

        Returns:
            WardrobeDistribution with counts and percentages
        """
        total = len(wardrobe)
        if total == 0:
            return WardrobeDistribution()

        # Category distribution
        cat_counter = Counter(g.attributes.category.value for g in wardrobe)
        by_category = self._counter_to_distribution(cat_counter, total)

        # Color distribution (primary color)
        color_counter = Counter(
            g.attributes.color.primary.lower() if g.attributes.color else "unknown"
            for g in wardrobe
        )
        by_color = self._counter_to_distribution(color_counter, total)

        # Formality distribution
        formality_counter = Counter(
            g.attributes.formality_level.value if g.attributes.formality_level else "casual"
            for g in wardrobe
        )
        by_formality = self._counter_to_distribution(formality_counter, total)

        # Season distribution
        season_counter: Counter = Counter()
        for g in wardrobe:
            seasons = g.attributes.season_suitable or []
            if g.attributes.seasonality and g.attributes.seasonality.seasons:
                seasons = g.attributes.seasonality.seasons
            if seasons:
                for s in seasons:
                    season_counter[s.value if hasattr(s, "value") else str(s)] += 1
            else:
                season_counter["all_season"] += 1
        by_season = self._counter_to_distribution(season_counter, total)

        # Material distribution
        mat_counter = Counter(
            g.attributes.material.primary.lower() if g.attributes.material else "unknown"
            for g in wardrobe
        )
        by_material = self._counter_to_distribution(mat_counter, total)

        # Pattern distribution
        pattern_counter = Counter(
            g.attributes.pattern.type.lower() if g.attributes.pattern else "solid"
            for g in wardrobe
        )
        by_pattern = self._counter_to_distribution(pattern_counter, total)

        return WardrobeDistribution(
            by_category=by_category,
            by_color=by_color,
            by_formality=by_formality,
            by_season=by_season,
            by_material=by_material,
            by_pattern=by_pattern,
            total_items=total,
        )

    # ==================== Gap Detection ====================

    def detect_gaps(
        self,
        wardrobe: List[Garment],
        context: Optional[UserContext] = None,
    ) -> List[WardrobeGap]:
        """
        Detect gaps and imbalances in the wardrobe.

        Args:
            wardrobe: All garments
            context: Optional user context for personalized gap detection

        Returns:
            List of identified gaps sorted by severity
        """
        gaps: List[WardrobeGap] = []
        total = len(wardrobe)
        if total == 0:
            return gaps

        cat_counter = Counter(g.attributes.category for g in wardrobe)

        # 1. Missing essential categories
        for cat, min_count in MIN_CATEGORY_ITEMS.items():
            count = cat_counter.get(cat, 0)
            if count == 0:
                gaps.append(WardrobeGap(
                    gap_type="category_missing",
                    severity="high",
                    description=f"No {cat.value} items in wardrobe",
                    recommendation=f"Add at least {min_count} {cat.value} items for a functional wardrobe",
                ))
            elif count < min_count:
                gaps.append(WardrobeGap(
                    gap_type="category_missing",
                    severity="medium",
                    description=f"Only {count} {cat.value} item(s) — minimum recommended is {min_count}",
                    recommendation=f"Add {min_count - count} more {cat.value} item(s)",
                ))

        # 2. Category imbalance — one category dominates
        for cat, count in cat_counter.items():
            ratio = count / total
            ideal = IDEAL_CATEGORY_RATIOS.get(cat, 0.10)
            if ratio > ideal * 2.5 and count > 3:
                gaps.append(WardrobeGap(
                    gap_type="category_imbalance",
                    severity="low",
                    description=f"{cat.value} is over-represented ({count}/{total}, {ratio:.0%} vs ideal {ideal:.0%})",
                    recommendation=f"Consider diversifying — you have many {cat.value} items relative to other categories",
                ))

        # 3. Color imbalance — very few colors represented
        color_counter = Counter(
            g.attributes.color.primary.lower() if g.attributes.color else "unknown"
            for g in wardrobe
        )
        unique_colors = len(color_counter)
        if total >= 5 and unique_colors <= 2:
            gaps.append(WardrobeGap(
                gap_type="color_imbalance",
                severity="medium",
                description=f"Very limited color palette: only {unique_colors} color(s) across {total} items",
                recommendation="Add items in complementary colors to increase outfit variety",
            ))

        # Dominant color check
        if color_counter:
            most_common_color, most_common_count = color_counter.most_common(1)[0]
            if total >= 5 and most_common_count / total > 0.6:
                gaps.append(WardrobeGap(
                    gap_type="color_imbalance",
                    severity="low",
                    description=f"Color '{most_common_color}' dominates ({most_common_count}/{total} items)",
                    recommendation="Balance with neutrals or accent colors",
                ))

        # 4. Formality gaps
        formality_counter = Counter(
            g.attributes.formality_level for g in wardrobe
        )
        has_casual = sum(
            formality_counter.get(f, 0)
            for f in [FormalityLevel.VERY_CASUAL, FormalityLevel.CASUAL]
        )
        has_smart = sum(
            formality_counter.get(f, 0)
            for f in [FormalityLevel.SMART_CASUAL, FormalityLevel.BUSINESS_CASUAL]
        )
        has_formal = sum(
            formality_counter.get(f, 0)
            for f in [FormalityLevel.BUSINESS, FormalityLevel.FORMAL, FormalityLevel.BLACK_TIE]
        )

        if total >= 5:
            if has_casual == 0:
                gaps.append(WardrobeGap(
                    gap_type="formality_gap",
                    severity="medium",
                    description="No casual items in wardrobe",
                    recommendation="Add casual pieces for everyday wear",
                ))
            if has_smart == 0:
                gaps.append(WardrobeGap(
                    gap_type="formality_gap",
                    severity="medium",
                    description="No smart-casual or business-casual items",
                    recommendation="Add versatile smart-casual pieces for date nights and casual office settings",
                ))
            if has_formal == 0 and context and context.occasion in [
                Occasion.BUSINESS, Occasion.FORMAL, Occasion.WEDDING, Occasion.INTERVIEW
            ]:
                gaps.append(WardrobeGap(
                    gap_type="formality_gap",
                    severity="high",
                    description="No formal items — needed for your upcoming occasion",
                    recommendation="Invest in at least one formal outfit set",
                ))

        # 5. Season gaps
        season_counter: Counter = Counter()
        for g in wardrobe:
            seasons = g.attributes.season_suitable or []
            if g.attributes.seasonality and g.attributes.seasonality.seasons:
                seasons = g.attributes.seasonality.seasons
            for s in seasons:
                season_counter[s if isinstance(s, Season) else Season(s)] += 1

        for season in Season:
            if season_counter.get(season, 0) == 0 and total >= 5:
                gaps.append(WardrobeGap(
                    gap_type="season_gap",
                    severity="medium",
                    description=f"No items suitable for {season.value}",
                    recommendation=f"Add {season.value}-appropriate items to ensure year-round coverage",
                ))

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        gaps.sort(key=lambda g: severity_order.get(g.severity, 3))

        return gaps

    # ==================== Occasion Coverage ====================

    def analyze_occasion_coverage(
        self,
        wardrobe: List[Garment],
        occasion: Occasion,
    ) -> OccasionCoverage:
        """
        Analyze how well the wardrobe covers a specific occasion.

        Args:
            wardrobe: All garments
            occasion: The occasion to check

        Returns:
            OccasionCoverage with score and missing categories
        """
        target_formality = OCCASION_FORMALITY.get(occasion, FormalityLevel.CASUAL)
        required_cats = OCCASION_REQUIRED_CATEGORIES.get(
            occasion, ["TOP", "BOTTOM", "SHOES"]
        )

        # Filter garments that are suitable for this occasion's formality
        suitable = []
        for g in wardrobe:
            garment_formality = g.attributes.formality_level or FormalityLevel.CASUAL
            garment_val = FORMALITY_ORDER.get(garment_formality, 2)
            target_val = FORMALITY_ORDER.get(target_formality, 2)
            # Allow ±1 formality level
            if abs(garment_val - target_val) <= 1:
                suitable.append(g)

        # Check which required categories are present
        available_cats = set(g.attributes.category.value.upper() for g in suitable)
        missing_cats = [c for c in required_cats if c not in available_cats]

        # Coverage score
        if not required_cats:
            coverage = 1.0 if suitable else 0.0
        else:
            covered = len(required_cats) - len(missing_cats)
            coverage = covered / len(required_cats)

        # Boost score if we have multiple options per category
        if coverage > 0:
            cat_counts = Counter(g.attributes.category.value.upper() for g in suitable)
            variety_bonus = min(0.2, sum(
                min(0.05, (count - 1) * 0.02)
                for count in cat_counts.values()
                if count > 1
            ))
            coverage = min(1.0, coverage + variety_bonus)

        suggestion = None
        if missing_cats:
            suggestion = f"Add {', '.join(missing_cats).lower()} items suitable for {occasion.value} occasions"
        elif coverage < 0.8:
            suggestion = f"Expand options for {occasion.value} — you have limited choices"

        return OccasionCoverage(
            occasion=occasion.value,
            coverage_score=round(coverage, 2),
            suitable_items_count=len(suitable),
            missing_categories=missing_cats,
            suggestion=suggestion,
        )

    # ==================== Versatility Scoring ====================

    def calculate_versatility(
        self,
        wardrobe: List[Garment],
        top_k: int = 5,
    ) -> List[GarmentVersatility]:
        """
        Score each garment's versatility — how many outfits it can be part of.

        Args:
            wardrobe: All garments
            top_k: Number of top versatile items to return

        Returns:
            List of GarmentVersatility sorted by score (descending)
        """
        if not wardrobe:
            return []

        results: List[GarmentVersatility] = []
        wardrobe_by_cat: Dict[str, List[Garment]] = {}
        for g in wardrobe:
            cat = g.attributes.category.value
            wardrobe_by_cat.setdefault(cat, []).append(g)

        for garment in wardrobe:
            score, compat_count, compat_cats, compat_occasions = self._score_garment_versatility(
                garment, wardrobe, wardrobe_by_cat
            )
            desc = self._garment_description(garment)
            results.append(GarmentVersatility(
                garment_id=garment.id,
                garment_description=desc,
                versatility_score=round(score, 2),
                compatible_outfit_count=compat_count,
                compatible_categories=compat_cats,
                compatible_occasions=compat_occasions,
            ))

        results.sort(key=lambda x: x.versatility_score, reverse=True)
        return results[:top_k]

    # ==================== What-If Simulation ====================

    def simulate_addition(
        self,
        wardrobe: List[Garment],
        virtual_garment: Garment,
        context: Optional[UserContext] = None,
    ) -> Dict[str, Any]:
        """
        Simulate adding a garment to the wardrobe and measure the impact.

        Args:
            wardrobe: Current wardrobe
            virtual_garment: The garment to simulate adding
            context: Optional user context

        Returns:
            Dict with before/after metrics and impact summary
        """
        # Before metrics
        before_distribution = self.analyze_distribution(wardrobe)
        before_gaps = self.detect_gaps(wardrobe, context)

        # After metrics
        new_wardrobe = wardrobe + [virtual_garment]
        after_distribution = self.analyze_distribution(new_wardrobe)
        after_gaps = self.detect_gaps(new_wardrobe, context)

        # Calculate versatility of the new garment
        wardrobe_by_cat: Dict[str, List[Garment]] = {}
        for g in new_wardrobe:
            cat = g.attributes.category.value
            wardrobe_by_cat.setdefault(cat, []).append(g)

        score, compat_count, _, _ = self._score_garment_versatility(
            virtual_garment, new_wardrobe, wardrobe_by_cat
        )

        gaps_resolved = len(before_gaps) - len(after_gaps)

        return {
            "garment_description": self._garment_description(virtual_garment),
            "versatility_score": round(score, 2),
            "new_outfit_combinations": compat_count,
            "gaps_before": len(before_gaps),
            "gaps_after": len(after_gaps),
            "gaps_resolved": max(0, gaps_resolved),
            "recommendation": "Highly recommended" if gaps_resolved > 0 and score > 0.5 else (
                "Good addition" if score > 0.3 else "Low impact addition"
            ),
        }

    # ==================== Purchase Suggestions ====================

    def generate_purchase_suggestions(
        self,
        wardrobe: List[Garment],
        context: Optional[UserContext] = None,
        gaps: Optional[List[WardrobeGap]] = None,
        occasion_coverage: Optional[List[OccasionCoverage]] = None,
        max_suggestions: int = 5,
    ) -> List[PurchaseSuggestion]:
        """
        Generate prioritized purchase suggestions based on wardrobe gaps.

        Args:
            wardrobe: Current wardrobe
            context: Optional user context
            gaps: Pre-computed gaps (computed if not provided)
            occasion_coverage: Pre-computed coverage (computed if not provided)
            max_suggestions: Maximum number of suggestions

        Returns:
            List of PurchaseSuggestion sorted by priority
        """
        if gaps is None:
            gaps = self.detect_gaps(wardrobe, context)
        if occasion_coverage is None:
            occasion_coverage = [
                self.analyze_occasion_coverage(wardrobe, occ)
                for occ in [Occasion.DAILY_WEAR, Occasion.WORK, Occasion.DATE, Occasion.COCKTAIL]
            ]

        suggestions: List[PurchaseSuggestion] = []
        priority = 1

        # From gaps (highest priority)
        for gap in gaps:
            if gap.severity == "high":
                cat = self._extract_category_from_gap(gap)
                suggestions.append(PurchaseSuggestion(
                    priority=priority,
                    category=cat,
                    description=gap.recommendation,
                    reason=gap.description,
                    estimated_outfit_increase=self._estimate_outfit_increase(cat, wardrobe),
                    suggested_colors=self._suggest_colors_for_wardrobe(wardrobe),
                    suggested_styles=[],
                    target_occasions=[],
                ))
                priority += 1

        # From occasion coverage gaps
        for cov in occasion_coverage:
            if cov.coverage_score < 0.6 and cov.missing_categories:
                for missing_cat in cov.missing_categories:
                    if not any(s.category.upper() == missing_cat.upper() for s in suggestions):
                        suggestions.append(PurchaseSuggestion(
                            priority=priority,
                            category=missing_cat.lower(),
                            description=f"Add {missing_cat.lower()} for {cov.occasion} occasions",
                            reason=f"Coverage for {cov.occasion} is only {cov.coverage_score:.0%}",
                            estimated_outfit_increase=self._estimate_outfit_increase(
                                missing_cat.lower(), wardrobe
                            ),
                            suggested_colors=self._suggest_colors_for_wardrobe(wardrobe),
                            suggested_styles=[],
                            target_occasions=[cov.occasion],
                        ))
                        priority += 1

        # Medium severity gaps
        for gap in gaps:
            if gap.severity == "medium" and priority <= max_suggestions:
                cat = self._extract_category_from_gap(gap)
                if not any(s.category == cat for s in suggestions):
                    suggestions.append(PurchaseSuggestion(
                        priority=priority,
                        category=cat,
                        description=gap.recommendation,
                        reason=gap.description,
                        estimated_outfit_increase=self._estimate_outfit_increase(cat, wardrobe),
                        suggested_colors=self._suggest_colors_for_wardrobe(wardrobe),
                        suggested_styles=[],
                        target_occasions=[],
                    ))
                    priority += 1

        return suggestions[:max_suggestions]

    # ==================== Private Helpers ====================

    def _counter_to_distribution(
        self, counter: Counter, total: int
    ) -> List[CategoryDistribution]:
        """Convert a Counter to a list of CategoryDistribution."""
        return sorted(
            [
                CategoryDistribution(
                    category=str(k),
                    count=v,
                    percentage=round((v / total) * 100, 1) if total > 0 else 0,
                )
                for k, v in counter.items()
            ],
            key=lambda x: x.count,
            reverse=True,
        )

    def _score_garment_versatility(
        self,
        garment: Garment,
        wardrobe: List[Garment],
        wardrobe_by_cat: Dict[str, List[Garment]],
    ) -> Tuple[float, int, List[str], List[str]]:
        """
        Score how versatile a single garment is.

        Returns:
            (score, compatible_count, compatible_categories, compatible_occasions)
        """
        cat = garment.attributes.category
        color = garment.attributes.color
        formality = garment.attributes.formality_level or FormalityLevel.CASUAL
        formality_val = FORMALITY_ORDER.get(formality, 2)

        compatible_count = 0
        compatible_cats: set = set()

        # Check compatibility with garments in other categories
        for other_cat, others in wardrobe_by_cat.items():
            if other_cat == cat.value:
                continue
            for other in others:
                other_formality = other.attributes.formality_level or FormalityLevel.CASUAL
                other_val = FORMALITY_ORDER.get(other_formality, 2)

                # Compatible if formality within 2 levels
                if abs(formality_val - other_val) <= 2:
                    compatible_count += 1
                    compatible_cats.add(other_cat)

        # Occasion compatibility
        compatible_occasions: List[str] = []
        for occ, occ_formality in OCCASION_FORMALITY.items():
            occ_val = FORMALITY_ORDER.get(occ_formality, 2)
            if abs(formality_val - occ_val) <= 1:
                compatible_occasions.append(occ.value)

        # Normalize score
        max_possible = max(1, len(wardrobe) - 1)
        base_score = compatible_count / max_possible

        # Bonus for neutral colors (more versatile)
        neutral_colors = {"black", "white", "grey", "gray", "navy", "beige", "cream", "khaki", "brown", "tan"}
        if color and color.primary.lower() in neutral_colors:
            base_score = min(1.0, base_score + 0.15)

        # Bonus for solid patterns
        if garment.attributes.pattern and garment.attributes.pattern.type.lower() == "solid":
            base_score = min(1.0, base_score + 0.05)

        # Bonus for covering many occasions
        occasion_bonus = min(0.1, len(compatible_occasions) * 0.02)
        base_score = min(1.0, base_score + occasion_bonus)

        return (
            round(base_score, 2),
            compatible_count,
            sorted(compatible_cats),
            compatible_occasions,
        )

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

    def _extract_category_from_gap(self, gap: WardrobeGap) -> str:
        """Extract the category name from a gap description."""
        for cat in GarmentCategory:
            if cat.value.lower() in gap.description.lower():
                return cat.value
        return "general"

    def _estimate_outfit_increase(self, category: str, wardrobe: List[Garment]) -> int:
        """Estimate how many new outfits a category addition would enable."""
        # Count items in complementary categories
        cat_map = {
            "top": ["bottom", "shoes"],
            "bottom": ["top", "shoes"],
            "shoes": ["top", "bottom"],
            "outerwear": ["top", "bottom"],
            "dress": ["shoes", "accessory"],
            "accessory": ["top", "bottom", "dress"],
        }
        complements = cat_map.get(category.lower(), ["top", "bottom"])
        complement_counts = []
        for comp in complements:
            count = sum(
                1 for g in wardrobe
                if g.attributes.category.value.lower() == comp
            )
            complement_counts.append(max(1, count))

        # Rough estimate: product of complement counts
        result = 1
        for c in complement_counts:
            result *= c
        return min(result, 50)  # Cap at 50

    def _suggest_colors_for_wardrobe(self, wardrobe: List[Garment]) -> List[str]:
        """Suggest colors that would complement the existing wardrobe."""
        existing_colors = set(
            g.attributes.color.primary.lower()
            for g in wardrobe
            if g.attributes.color and g.attributes.color.primary
        )

        # Always-safe neutrals
        suggestions = []
        neutrals = ["black", "white", "navy", "grey", "beige"]
        for n in neutrals:
            if n not in existing_colors:
                suggestions.append(n)
                if len(suggestions) >= 3:
                    break

        # Complementary colors
        for color in list(existing_colors)[:3]:
            complements = self._color_analyzer.get_complementary_colors(color)
            for c in complements:
                if c.lower() not in existing_colors and c not in suggestions:
                    suggestions.append(c)
                    if len(suggestions) >= 5:
                        return suggestions

        return suggestions[:5]

    def _calculate_overall_health(
        self,
        distribution: WardrobeDistribution,
        gaps: List[WardrobeGap],
        occasion_coverage: List[OccasionCoverage],
    ) -> float:
        """Calculate an overall wardrobe health score 0-1."""
        score = 1.0

        # Penalty for gaps
        for gap in gaps:
            if gap.severity == "high":
                score -= 0.15
            elif gap.severity == "medium":
                score -= 0.08
            elif gap.severity == "low":
                score -= 0.03

        # Average occasion coverage
        if occasion_coverage:
            avg_coverage = sum(c.coverage_score for c in occasion_coverage) / len(occasion_coverage)
            score = score * 0.6 + avg_coverage * 0.4

        # Bonus for good item count
        if distribution.total_items >= 15:
            score = min(1.0, score + 0.05)
        elif distribution.total_items < 5:
            score -= 0.1

        return round(max(0.0, min(1.0, score)), 2)

    def _generate_summary(
        self,
        distribution: WardrobeDistribution,
        gaps: List[WardrobeGap],
        occasion_coverage: List[OccasionCoverage],
        overall_score: float,
    ) -> str:
        """Generate a human-readable summary of the analysis."""
        parts = []
        parts.append(f"Wardrobe contains {distribution.total_items} items.")

        # Category breakdown
        if distribution.by_category:
            top_cats = distribution.by_category[:3]
            cats_str = ", ".join(f"{c.category} ({c.count})" for c in top_cats)
            parts.append(f"Top categories: {cats_str}.")

        # Gaps summary
        high_gaps = [g for g in gaps if g.severity == "high"]
        if high_gaps:
            parts.append(f"{len(high_gaps)} critical gap(s) detected: {high_gaps[0].description}.")
        elif gaps:
            parts.append(f"{len(gaps)} minor gap(s) detected.")
        else:
            parts.append("No significant gaps detected.")

        # Occasion coverage
        if occasion_coverage:
            low_coverage = [c for c in occasion_coverage if c.coverage_score < 0.6]
            if low_coverage:
                occ_names = ", ".join(c.occasion for c in low_coverage)
                parts.append(f"Low coverage for: {occ_names}.")
            else:
                parts.append("Good coverage across all checked occasions.")

        # Grade
        if overall_score >= 0.8:
            parts.append("Overall: Excellent wardrobe!")
        elif overall_score >= 0.6:
            parts.append("Overall: Good wardrobe with room for improvement.")
        elif overall_score >= 0.4:
            parts.append("Overall: Needs attention — several gaps to address.")
        else:
            parts.append("Overall: Significant gaps — consider building essentials first.")

        return " ".join(parts)
