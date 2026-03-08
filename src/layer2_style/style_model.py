"""
Style Intelligence Model - Main Orchestrator for Layer 2
This is the core differentiator of the system.

Learns style rules from runway looks WITHOUT copying specific designs.
Focuses on:
- Compatibility scoring between garments
- Color harmony analysis
- Silhouette coherence
- Proportion ratios
- Formality level matching
- Seven-Point Rule scoring
- Season Color Harmony
- Volume Balance
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.core.models import Garment, Outfit, OutfitItem, GarmentCategory
from src.core import get_logger
from .compatibility_scorer import CompatibilityScorer
from .color_harmony import ColorHarmonyAnalyzer
from .silhouette_analyzer import SilhouetteAnalyzer
from .formality_matcher import FormalityMatcher
from .seven_point_rule import SevenPointRuleScorer
from .season_color_harmony import SeasonColorHarmonyScorer, ColorSeason
from .proportion_scorer import ProportionScorer
from .volume_balance_scorer import VolumeBalanceScorer, BodyShape

logger = get_logger(__name__)


class StyleIntelligenceModel:
    """
    Core style intelligence engine.
    
    This model learns abstract style rules from fashion data:
    - What colors go well together (not specific color combinations)
    - What silhouettes create balanced outfits
    - What formality levels should match
    - What proportions look harmonious
    
    It does NOT learn to copy specific designer looks.
    
    Style Scoring Weights:
    | Rule                | Weight |
    |---------------------|--------|
    | 7-Point Rule        | 30%    |
    | Color Harmony       | 30%    |
    | Proportions         | 25%    |
    | Volume Balance      | 15%    |
    """
    
    def __init__(self):
        # Basic scorers
        self.compatibility_scorer = CompatibilityScorer()
        self.color_harmony = ColorHarmonyAnalyzer()
        self.silhouette_analyzer = SilhouetteAnalyzer()
        self.formality_matcher = FormalityMatcher()
        
        # Advanced style scorers
        self.seven_point_scorer = SevenPointRuleScorer()
        self.season_color_scorer = SeasonColorHarmonyScorer()
        self.proportion_scorer = ProportionScorer()
        self.volume_balance_scorer = VolumeBalanceScorer()
        
        # User preferences (can be personalized)
        self._user_season: Optional[ColorSeason] = None
        self._body_shape: Optional[BodyShape] = None
        
        # Weights for basic scoring (legacy compatibility)
        self.basic_weights = {
            "compatibility": 0.25,
            "color_harmony": 0.20,
            "silhouette": 0.15,
            "formality": 0.15,
            "proportion": 0.10,
            "seven_point_rule": 0.15
        }
        
        # Weights for advanced style scoring
        self.style_weights = {
            "seven_point": 0.30,
            "color_harmony": 0.30,
            "proportions": 0.25,
            "volume_balance": 0.15
        }
    
    def set_user_profile(
        self, 
        color_season: Optional[ColorSeason] = None,
        body_shape: Optional[BodyShape] = None
    ) -> None:
        """
        Set user profile for personalized scoring.
        
        Args:
            color_season: User's skin tone color season (SPRING, SUMMER, AUTUMN, WINTER)
            body_shape: User's body shape for volume recommendations
        """
        if color_season:
            self._user_season = color_season
        if body_shape:
            self._body_shape = body_shape
    
    async def score_outfit(self, items: List[Garment]) -> Dict[str, Any]:
        """
        Score a complete outfit across all style dimensions.
        
        Args:
            items: List of garments that make up the outfit
            
        Returns:
            Dictionary with individual scores and overall score
        """
        if len(items) < 2:
            return {
                "overall_score": 0.5,
                "message": "Need at least 2 items to score an outfit"
            }
        
        # Calculate individual scores
        compatibility_score = await self.compatibility_scorer.score_items(items)
        color_score = self.color_harmony.analyze_outfit_colors(items)
        silhouette_score = self.silhouette_analyzer.analyze_silhouette_balance(items)
        formality_score = self.formality_matcher.check_formality_consistency(items)
        proportion_score = self._calculate_proportion_score(items)
        
        # Calculate 7-point rule score
        seven_point_result = self.seven_point_scorer.analyze_outfit(items)
        seven_point_score = seven_point_result.score
        
        # Weighted combination
        overall_score = (
            self.basic_weights["compatibility"] * compatibility_score +
            self.basic_weights["color_harmony"] * color_score +
            self.basic_weights["silhouette"] * silhouette_score +
            self.basic_weights["formality"] * formality_score +
            self.basic_weights["proportion"] * proportion_score +
            self.basic_weights["seven_point_rule"] * seven_point_score
        )
        
        return {
            "overall_score": round(overall_score, 3),
            "breakdown": {
                "compatibility": round(compatibility_score, 3),
                "color_harmony": round(color_score, 3),
                "silhouette_balance": round(silhouette_score, 3),
                "formality_consistency": round(formality_score, 3),
                "proportion_balance": round(proportion_score, 3),
                "seven_point_rule": round(seven_point_score, 3)
            },
            "seven_point_analysis": {
                "total_points": seven_point_result.total_points,
                "is_harmonious": seven_point_result.is_harmonious,
                "recommendation": seven_point_result.recommendation,
                "item_breakdown": seven_point_result.breakdown
            },
            "suggestions": self._generate_suggestions(
                compatibility_score, color_score, silhouette_score, 
                formality_score, proportion_score, seven_point_score,
                seven_point_result
            )
        }
    
    async def score_outfit_advanced(
        self, 
        items: List[Garment],
        user_season: Optional[ColorSeason] = None,
        body_shape: Optional[BodyShape] = None
    ) -> Dict[str, Any]:
        """
        Advanced outfit scoring with all style rules.
        
        Uses the complete style scoring system:
        - 7-Point Rule (30%)
        - Season Color Harmony (30%)
        - Proportions / Rule of Thirds (25%)
        - Volume Balance (15%)
        
        Args:
            items: List of garments that make up the outfit
            user_season: Optional user's color season for personalized scoring
            body_shape: Optional body shape for volume recommendations
            
        Returns:
            Comprehensive style report dictionary
        """
        if len(items) < 2:
            return {
                "overall_score": 0.5,
                "grade": "C",
                "message": "Need at least 2 items for complete style analysis"
            }
        
        # Use stored profile if not provided
        season = user_season or self._user_season
        shape = body_shape or self._body_shape
        
        # Run all advanced scorers
        seven_point_result = self.seven_point_scorer.analyze_outfit(items)
        proportion_result = self.proportion_scorer.analyze_outfit(items)
        volume_result = self.volume_balance_scorer.analyze_outfit(items, shape)
        
        # Color harmony - use season if available
        if season:
            color_result = self.season_color_scorer.analyze_outfit(items, season)
        else:
            # Fallback to basic color harmony
            basic_color_score = self.color_harmony.analyze_outfit_colors(items)
            color_result = type('ColorResult', (), {
                'score': basic_color_score,
                'matches': [],
                'clashes': [],
                'recommendation': "Provide color season for personalized color analysis"
            })()
        
        # Calculate weighted overall score
        overall_score = (
            self.style_weights["seven_point"] * seven_point_result.score +
            self.style_weights["color_harmony"] * color_result.score +
            self.style_weights["proportions"] * proportion_result.score +
            self.style_weights["volume_balance"] * volume_result.score
        )
        
        # Calculate grade
        grade = self._calculate_grade(overall_score)
        
        # Generate comprehensive analysis
        return {
            "overall_score": round(overall_score, 3),
            "grade": grade,
            "breakdown": {
                "seven_point_rule": {
                    "score": round(seven_point_result.score, 3),
                    "weight": "30%",
                    "total_points": seven_point_result.total_points,
                    "is_harmonious": seven_point_result.is_harmonious,
                    "recommendation": seven_point_result.recommendation
                },
                "color_harmony": {
                    "score": round(color_result.score, 3),
                    "weight": "30%",
                    "matches": getattr(color_result, 'matches', []),
                    "clashes": getattr(color_result, 'clashes', []),
                    "recommendation": getattr(color_result, 'recommendation', '')
                },
                "proportions": {
                    "score": round(proportion_result.score, 3),
                    "weight": "25%",
                    "ratio": proportion_result.ratio.value if hasattr(proportion_result.ratio, 'value') else str(proportion_result.ratio),
                    "recommendation": proportion_result.recommendation
                },
                "volume_balance": {
                    "score": round(volume_result.score, 3),
                    "weight": "15%",
                    "balance_type": volume_result.balance_type,
                    "recommendation": volume_result.recommendation
                }
            },
            "strengths": self._identify_strengths(
                seven_point_result, color_result, proportion_result, volume_result
            ),
            "improvements": self._identify_improvements(
                seven_point_result, color_result, proportion_result, volume_result
            ),
            "personalization": {
                "color_season_set": season is not None,
                "body_shape_set": shape is not None
            }
        }
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score."""
        if score >= 0.85:
            return "A"
        elif score >= 0.70:
            return "B"
        elif score >= 0.55:
            return "C"
        elif score >= 0.40:
            return "D"
        return "F"
    
    def _identify_strengths(self, seven_point, color, proportion, volume) -> List[str]:
        """Identify outfit strengths."""
        strengths = []
        
        if seven_point.is_harmonious:
            strengths.append(f"Perfect 7-point balance ({seven_point.total_points} points)")
        
        if color.score >= 0.8:
            strengths.append("Beautiful color harmony")
        
        if proportion.score >= 0.7:
            ratio_str = proportion.ratio.value if hasattr(proportion.ratio, 'value') else str(proportion.ratio)
            strengths.append(f"Great proportions ({ratio_str})")
        
        if volume.score >= 0.7:
            strengths.append(f"Well-balanced volumes ({volume.balance_type.replace('_', ' ')})")
        
        return strengths
    
    def _identify_improvements(self, seven_point, color, proportion, volume) -> List[str]:
        """Identify areas for improvement."""
        improvements = []
        
        if not seven_point.is_harmonious:
            improvements.append(seven_point.recommendation)
        
        if color.score < 0.6 and hasattr(color, 'clashes') and color.clashes:
            improvements.append(f"Consider swapping colors: {', '.join(color.clashes[:2])}")
        
        if proportion.score < 0.6:
            improvements.append(proportion.recommendation)
        
        if volume.score < 0.6:
            improvements.append(volume.recommendation)
        
        return improvements
    
    async def find_best_match(
        self,
        base_item: Garment,
        candidates: List[Garment],
        top_k: int = 5
    ) -> List[Tuple[Garment, float]]:
        """
        Find the best matching items for a given garment.
        
        Args:
            base_item: The item to find matches for
            candidates: Pool of potential matching items
            top_k: Number of top matches to return
            
        Returns:
            List of (garment, score) tuples sorted by score
        """
        scores = []
        
        for candidate in candidates:
            # Skip same category items for top/bottom matching
            if self._should_skip_category(base_item, candidate):
                continue
            
            score = await self._calculate_pair_score(base_item, candidate)
            scores.append((candidate, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    async def generate_outfit(
        self,
        wardrobe: List[Garment],
        anchor_item: Optional[Garment] = None,
        target_categories: Optional[List[GarmentCategory]] = None
    ) -> Outfit:
        """
        Generate a complete outfit from a wardrobe.
        
        Args:
            wardrobe: Available garments
            anchor_item: Optional item to build outfit around
            target_categories: Categories to include in outfit
            
        Returns:
            Generated outfit with scores
        """
        if target_categories is None:
            target_categories = [
                GarmentCategory.TOP,
                GarmentCategory.BOTTOM,
                GarmentCategory.SHOES
            ]
        
        selected_items = []
        
        if anchor_item:
            selected_items.append(OutfitItem(
                garment=anchor_item,
                role="anchor"
            ))
            target_categories = [c for c in target_categories 
                               if c != anchor_item.attributes.category]
        
        # Greedily select best matching items for each category
        for category in target_categories:
            category_items = [g for g in wardrobe 
                           if g.attributes.category == category]
            
            if not category_items:
                continue
            
            if selected_items:
                # Find best match considering already selected items
                best_item = await self._find_best_for_outfit(
                    category_items,
                    [oi.garment for oi in selected_items]
                )
            else:
                # Just pick first available
                best_item = category_items[0]
            
            if best_item:
                selected_items.append(OutfitItem(
                    garment=best_item,
                    role=category.value
                ))
        
        # Score the complete outfit
        garments = [oi.garment for oi in selected_items]
        scores = await self.score_outfit(garments)
        
        return Outfit(
            id=self._generate_outfit_id(),
            items=selected_items,
            compatibility_score=scores["breakdown"]["compatibility"],
            style_coherence_score=scores["breakdown"]["silhouette_balance"],
            occasion_match_score=0.0,  # Will be set by context engine
            overall_score=scores["overall_score"]
        )
    
    async def _calculate_pair_score(
        self,
        item1: Garment,
        item2: Garment
    ) -> float:
        """Calculate compatibility score between two items."""
        compatibility = await self.compatibility_scorer.score_pair(item1, item2)
        color = self.color_harmony.score_color_pair(
            item1.attributes.color,
            item2.attributes.color
        )
        formality = self.formality_matcher.score_pair(
            item1.attributes.formality_level,
            item2.attributes.formality_level
        )
        
        return (0.4 * compatibility + 0.35 * color + 0.25 * formality)
    
    async def _find_best_for_outfit(
        self,
        candidates: List[Garment],
        current_outfit: List[Garment]
    ) -> Optional[Garment]:
        """Find best candidate that matches current outfit items."""
        best_score = -1
        best_item = None
        
        for candidate in candidates:
            # Calculate average score with all current items
            scores = []
            for outfit_item in current_outfit:
                score = await self._calculate_pair_score(candidate, outfit_item)
                scores.append(score)
            
            avg_score = np.mean(scores) if scores else 0
            
            if avg_score > best_score:
                best_score = avg_score
                best_item = candidate
        
        return best_item
    
    def _should_skip_category(self, item1: Garment, item2: Garment) -> bool:
        """Determine if category combination should be skipped."""
        cat1 = item1.attributes.category
        cat2 = item2.attributes.category
        
        # Same category usually not in same outfit (except accessories)
        if cat1 == cat2 and cat1 != GarmentCategory.ACCESSORY:
            return True
        
        return False
    
    def _calculate_proportion_score(self, items: List[Garment]) -> float:
        """
        Calculate proportion balance score.
        
        Good proportions consider:
        - Volume balance (top vs bottom)
        - Length ratios
        - Visual weight distribution
        """
        # Get tops and bottoms
        tops = [i for i in items if i.attributes.category in 
               [GarmentCategory.TOP, GarmentCategory.OUTERWEAR]]
        bottoms = [i for i in items if i.attributes.category == GarmentCategory.BOTTOM]
        
        if not tops or not bottoms:
            return 0.7  # Neutral score
        
        # Check fit combinations
        top_fits = [t.attributes.fit for t in tops if t.attributes.fit]
        bottom_fits = [b.attributes.fit for b in bottoms if b.attributes.fit]
        
        if not top_fits or not bottom_fits:
            return 0.7
        
        # Proportion rules (learned from fashion principles)
        # Oversized top + slim bottom = good
        # Fitted top + wide bottom = good
        # Oversized top + oversized bottom = can work but tricky
        
        proportion_rules = {
            ("oversized", "slim"): 0.9,
            ("oversized", "regular"): 0.8,
            ("relaxed", "slim"): 0.85,
            ("fitted", "wide"): 0.85,
            ("fitted", "relaxed"): 0.8,
            ("regular", "regular"): 0.75,
            ("oversized", "oversized"): 0.5,
        }
        
        top_fit = top_fits[0]
        bottom_fit = bottom_fits[0]
        
        return proportion_rules.get((top_fit, bottom_fit), 0.7)
    
    def _generate_suggestions(
        self,
        compatibility: float,
        color: float,
        silhouette: float,
        formality: float,
        proportion: float,
        seven_point_score: float = None,
        seven_point_result = None
    ) -> List[str]:
        """Generate improvement suggestions based on scores."""
        suggestions = []
        
        if color < 0.6:
            suggestions.append("Consider items with more complementary colors")
        
        if formality < 0.6:
            suggestions.append("Mix of formal and casual items - consider more consistency")
        
        if proportion < 0.6:
            suggestions.append("Try balancing proportions with different fits")
        
        if silhouette < 0.6:
            suggestions.append("Silhouettes may clash - try more cohesive shapes")
        
        # Seven-point rule suggestions
        if seven_point_result and not seven_point_result.is_harmonious:
            if seven_point_result.total_points < 7:
                suggestions.append(
                    f"Outfit scores {seven_point_result.total_points} points (target: 7-10). "
                    "Add a statement piece with texture, print, or interesting details."
                )
            elif seven_point_result.total_points > 10:
                suggestions.append(
                    f"Outfit scores {seven_point_result.total_points} points (target: 7-10). "
                    "Too many statement pieces - swap one for a basic staple."
                )
        
        return suggestions
    
    def _generate_outfit_id(self) -> str:
        """Generate unique outfit ID."""
        import uuid
        return f"outfit_{uuid.uuid4().hex[:12]}"
