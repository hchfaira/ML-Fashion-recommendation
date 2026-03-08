"""
Occasion Compatibility Scorer
Ensures outfit formality matches the event/occasion context.

A perfect outfit score is meaningless if worn to the wrong occasion.
This scorer creates a compatibility matrix between:
- GarmentAttributes.formality_level
- UserContext.occasion (event type)

| Mismatch Level      | Score Impact        |
|---------------------|---------------------|
| Perfect match       | +5 pts (100%)       |
| One level off       | +3 pts (60%)        |
| Two levels off      | +1 pt (20%)         |
| Three+ levels off   | 0 pts (style fail)  |
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.core.models import Garment, Occasion, FormalityLevel, UserContext
from src.core import get_logger

logger = get_logger(__name__)


@dataclass
class OccasionResult:
    """Result from occasion compatibility analysis."""
    score: float  # 0-1 normalized
    points: int   # Raw points (max 5)
    target_occasion: Optional[Occasion]
    outfit_formality: FormalityLevel
    expected_formality: FormalityLevel
    is_appropriate: bool
    mismatch_level: int  # 0 = perfect, 1 = slightly off, etc.
    recommendation: str


class OccasionScorer:
    """
    Scores outfits based on occasion appropriateness.
    
    Even the most stylish outfit fails if worn to the wrong event:
    - Smoking at the beach? Fashion fail.
    - Swimsuit at a wedding? Definitely no.
    - Jeans at black-tie? Faux pas.
    
    This scorer bridges the gap between "stylish" and "appropriate".
    """
    
    # Mapping of occasions to expected formality levels (ordered by preference)
    OCCASION_FORMALITY_MAP: Dict[Occasion, List[FormalityLevel]] = {
        Occasion.CASUAL: [
            FormalityLevel.CASUAL,
            FormalityLevel.VERY_CASUAL,
            FormalityLevel.SMART_CASUAL
        ],
        Occasion.BUSINESS: [
            FormalityLevel.BUSINESS,
            FormalityLevel.BUSINESS_CASUAL,
            FormalityLevel.SMART_CASUAL
        ],
        Occasion.FORMAL: [
            FormalityLevel.FORMAL,
            FormalityLevel.BUSINESS,
            FormalityLevel.BLACK_TIE
        ],
        Occasion.SPORT: [
            FormalityLevel.VERY_CASUAL,
            FormalityLevel.CASUAL
        ],
        Occasion.EVENING: [
            FormalityLevel.FORMAL,
            FormalityLevel.SMART_CASUAL,
            FormalityLevel.BLACK_TIE
        ],
        Occasion.BEACH: [
            FormalityLevel.VERY_CASUAL,
            FormalityLevel.CASUAL
        ],
        Occasion.DATE: [
            FormalityLevel.SMART_CASUAL,
            FormalityLevel.CASUAL,
            FormalityLevel.BUSINESS_CASUAL
        ],
    }
    
    # Primary expected formality for each occasion
    OCCASION_PRIMARY_FORMALITY: Dict[Occasion, FormalityLevel] = {
        Occasion.CASUAL: FormalityLevel.CASUAL,
        Occasion.BUSINESS: FormalityLevel.BUSINESS,
        Occasion.FORMAL: FormalityLevel.FORMAL,
        Occasion.SPORT: FormalityLevel.VERY_CASUAL,
        Occasion.EVENING: FormalityLevel.FORMAL,
        Occasion.BEACH: FormalityLevel.VERY_CASUAL,
        Occasion.DATE: FormalityLevel.SMART_CASUAL,
    }
    
    # Formality levels ordered from most casual to most formal
    FORMALITY_ORDER = [
        FormalityLevel.VERY_CASUAL,
        FormalityLevel.CASUAL,
        FormalityLevel.SMART_CASUAL,
        FormalityLevel.BUSINESS_CASUAL,
        FormalityLevel.BUSINESS,
        FormalityLevel.FORMAL,
        FormalityLevel.BLACK_TIE,
    ]
    
    def __init__(self):
        self.max_points = 5
    
    def analyze_outfit(
        self,
        items: List[Garment],
        occasion: Optional[Occasion] = None,
        context: Optional[UserContext] = None
    ) -> OccasionResult:
        """
        Analyze outfit appropriateness for an occasion.
        
        Args:
            items: Outfit garments
            occasion: Target occasion (overrides context if provided)
            context: User context with occasion info
            
        Returns:
            OccasionResult with score and analysis
        """
        # Determine target occasion
        target = occasion or (context.occasion if context else None)
        
        if not target:
            return OccasionResult(
                score=0.7,  # Neutral score without occasion
                points=3,
                target_occasion=None,
                outfit_formality=FormalityLevel.CASUAL,
                expected_formality=FormalityLevel.CASUAL,
                is_appropriate=True,
                mismatch_level=0,
                recommendation="Provide an occasion for accurate context scoring"
            )
        
        if not items:
            return OccasionResult(
                score=0.0,
                points=0,
                target_occasion=target,
                outfit_formality=FormalityLevel.CASUAL,
                expected_formality=self.OCCASION_PRIMARY_FORMALITY.get(target, FormalityLevel.CASUAL),
                is_appropriate=False,
                mismatch_level=5,
                recommendation="Add items to analyze"
            )
        
        # Calculate outfit's overall formality
        outfit_formality = self._calculate_outfit_formality(items)
        expected_formality = self.OCCASION_PRIMARY_FORMALITY.get(target, FormalityLevel.CASUAL)
        acceptable_levels = self.OCCASION_FORMALITY_MAP.get(target, [FormalityLevel.CASUAL])
        
        # Calculate mismatch
        mismatch_level = self._calculate_formality_distance(outfit_formality, expected_formality)
        is_appropriate = outfit_formality in acceptable_levels
        
        # Calculate points
        points = self._calculate_points(outfit_formality, expected_formality, acceptable_levels)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            target, outfit_formality, expected_formality, is_appropriate, mismatch_level
        )
        
        return OccasionResult(
            score=points / self.max_points,
            points=points,
            target_occasion=target,
            outfit_formality=outfit_formality,
            expected_formality=expected_formality,
            is_appropriate=is_appropriate,
            mismatch_level=mismatch_level,
            recommendation=recommendation
        )
    
    def _calculate_outfit_formality(self, items: List[Garment]) -> FormalityLevel:
        """
        Calculate the overall formality of an outfit.
        
        Uses the most formal item as the baseline, then adjusts based on:
        - Majority of items
        - Presence of casual pieces that bring down formality
        """
        formality_scores = []
        
        for item in items:
            level = item.attributes.formality_level
            if level in self.FORMALITY_ORDER:
                formality_scores.append(self.FORMALITY_ORDER.index(level))
            else:
                formality_scores.append(1)  # Default to casual
        
        if not formality_scores:
            return FormalityLevel.CASUAL
        
        # Use weighted approach: most formal piece counts more, but average matters
        max_formal = max(formality_scores)
        avg_formal = sum(formality_scores) / len(formality_scores)
        
        # Weight towards most formal (60/40 split)
        weighted_score = int(0.6 * max_formal + 0.4 * avg_formal)
        weighted_score = min(weighted_score, len(self.FORMALITY_ORDER) - 1)
        
        return self.FORMALITY_ORDER[weighted_score]
    
    def _calculate_formality_distance(
        self,
        outfit_level: FormalityLevel,
        expected_level: FormalityLevel
    ) -> int:
        """Calculate how many levels apart two formality levels are."""
        try:
            outfit_idx = self.FORMALITY_ORDER.index(outfit_level)
            expected_idx = self.FORMALITY_ORDER.index(expected_level)
            return abs(outfit_idx - expected_idx)
        except ValueError:
            return 3  # Default to significant mismatch
    
    def _calculate_points(
        self,
        outfit_level: FormalityLevel,
        expected_level: FormalityLevel,
        acceptable_levels: List[FormalityLevel]
    ) -> int:
        """Calculate points based on formality match."""
        # Perfect match
        if outfit_level == expected_level:
            return 5
        
        # Within acceptable range
        if outfit_level in acceptable_levels:
            # Determine how far from ideal
            acceptable_idx = acceptable_levels.index(outfit_level) if outfit_level in acceptable_levels else 99
            if acceptable_idx == 0:
                return 5  # Primary match
            elif acceptable_idx == 1:
                return 4  # Secondary match
            else:
                return 3  # Tertiary match
        
        # Calculate distance penalty
        distance = self._calculate_formality_distance(outfit_level, expected_level)
        
        if distance == 1:
            return 3  # One level off
        elif distance == 2:
            return 1  # Two levels off
        else:
            return 0  # Major mismatch
    
    def _generate_recommendation(
        self,
        occasion: Occasion,
        outfit_formality: FormalityLevel,
        expected_formality: FormalityLevel,
        is_appropriate: bool,
        mismatch_level: int
    ) -> str:
        """Generate recommendation based on analysis."""
        occasion_name = occasion.value.replace("_", " ").title()
        outfit_desc = outfit_formality.value.replace("_", " ")
        expected_desc = expected_formality.value.replace("_", " ")
        
        if is_appropriate and mismatch_level == 0:
            return f"Perfect for {occasion_name}! Your {outfit_desc} outfit is spot on."
        
        if is_appropriate:
            return f"Acceptable for {occasion_name}. You're in the right range."
        
        # Mismatch scenarios
        if mismatch_level >= 3:
            # Severe mismatch
            if self.FORMALITY_ORDER.index(outfit_formality) < self.FORMALITY_ORDER.index(expected_formality):
                return f"Too casual for {occasion_name}. This event calls for {expected_desc} attire."
            else:
                return f"Too formal for {occasion_name}. A more {expected_desc} approach would work better."
        
        elif mismatch_level == 2:
            if self.FORMALITY_ORDER.index(outfit_formality) < self.FORMALITY_ORDER.index(expected_formality):
                return f"Consider dressing up slightly for {occasion_name}. Aim for {expected_desc}."
            else:
                return f"You could dress down a bit for {occasion_name}. {expected_desc.title()} is the target."
        
        else:
            return f"Close! Just slightly adjust toward {expected_desc} for {occasion_name}."
    
    def get_formality_suggestions(
        self,
        occasion: Occasion
    ) -> Dict[str, List[str]]:
        """
        Get garment suggestions for an occasion.
        
        Returns suggestions for each category that would be appropriate.
        """
        expected = self.OCCASION_PRIMARY_FORMALITY.get(occasion, FormalityLevel.CASUAL)
        
        suggestions = {
            Occasion.CASUAL: {
                "tops": ["t-shirt", "casual button-down", "hoodie", "sweater"],
                "bottoms": ["jeans", "chinos", "shorts"],
                "shoes": ["sneakers", "loafers", "sandals"],
            },
            Occasion.BUSINESS: {
                "tops": ["dress shirt", "blouse", "blazer"],
                "bottoms": ["dress pants", "pencil skirt", "tailored trousers"],
                "shoes": ["oxfords", "heels", "loafers"],
            },
            Occasion.FORMAL: {
                "tops": ["tuxedo jacket", "formal dress", "evening gown"],
                "bottoms": ["dress pants", "formal skirt"],
                "shoes": ["dress shoes", "heels", "formal flats"],
            },
            Occasion.SPORT: {
                "tops": ["athletic top", "tank top", "sports bra"],
                "bottoms": ["leggings", "athletic shorts", "track pants"],
                "shoes": ["running shoes", "training shoes", "cleats"],
            },
            Occasion.EVENING: {
                "tops": ["evening top", "cocktail dress", "silk blouse"],
                "bottoms": ["dress pants", "statement skirt"],
                "shoes": ["heels", "dress shoes", "elegant flats"],
            },
            Occasion.BEACH: {
                "tops": ["tank top", "linen shirt", "swimwear cover-up"],
                "bottoms": ["shorts", "linen pants", "swimwear"],
                "shoes": ["sandals", "flip-flops", "espadrilles"],
            },
            Occasion.DATE: {
                "tops": ["nice blouse", "fitted sweater", "stylish top"],
                "bottoms": ["nice jeans", "skirt", "tailored pants"],
                "shoes": ["heels", "clean sneakers", "boots"],
            },
        }
        
        return suggestions.get(occasion, suggestions[Occasion.CASUAL])


# Convenience functions
def check_occasion_fit(
    items: List[Garment],
    occasion: Occasion
) -> bool:
    """Quick check if outfit is appropriate for occasion."""
    scorer = OccasionScorer()
    result = scorer.analyze_outfit(items, occasion)
    return result.is_appropriate


def get_occasion_score(
    items: List[Garment],
    occasion: Occasion
) -> float:
    """Get occasion appropriateness score (0-1)."""
    scorer = OccasionScorer()
    result = scorer.analyze_outfit(items, occasion)
    return result.score


def get_formality_gap(
    items: List[Garment],
    occasion: Occasion
) -> int:
    """Get formality gap (0 = perfect, higher = worse)."""
    scorer = OccasionScorer()
    result = scorer.analyze_outfit(items, occasion)
    return result.mismatch_level
