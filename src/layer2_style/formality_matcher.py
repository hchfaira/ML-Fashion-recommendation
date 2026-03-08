"""
Formality Matcher
Ensures formality level consistency across outfit items.
"""
from typing import List, Dict, Tuple
from enum import IntEnum

from config import get_config
from src.core.models import Garment, FormalityLevel
from src.core import get_logger

logger = get_logger(__name__)


class FormalityScore(IntEnum):
    """Numeric formality scores for comparison."""
    VERY_CASUAL = 1
    CASUAL = 2
    SMART_CASUAL = 3
    BUSINESS_CASUAL = 4
    BUSINESS = 5
    FORMAL = 6
    BLACK_TIE = 7


class FormalityMatcher:
    """
    Ensures formality consistency in outfits.
    
    An outfit looks best when all items are at similar
    formality levels. A suit jacket with sweatpants 
    creates visual discord.
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Load formality mapping from config
        occasion_data = self.config.get_data("occasion_data", default={})
        formality_mapping = occasion_data.get("formality_numeric_mapping", {})
        
        # Formality to numeric mapping
        self.formality_scores = {
            FormalityLevel.VERY_CASUAL: formality_mapping.get("very_casual", 1),
            FormalityLevel.CASUAL: formality_mapping.get("casual", 2),
            FormalityLevel.SMART_CASUAL: formality_mapping.get("smart_casual", 3),
            FormalityLevel.BUSINESS_CASUAL: formality_mapping.get("business_casual", 4),
            FormalityLevel.BUSINESS: formality_mapping.get("business", 5),
            FormalityLevel.FORMAL: formality_mapping.get("formal", 6),
            FormalityLevel.BLACK_TIE: formality_mapping.get("black_tie", 7)
        }
        
        # Load scoring parameters
        formality_params = self.config.get_parameters("model_parameters", "scoring.formality", default={})
        self.max_gap = formality_params.get("max_acceptable_gap", 2)
        self._score_perfect = formality_params.get("perfect_match", 1.0)
        self._score_one_gap = formality_params.get("one_gap", 0.9)
        self._score_two_gap = formality_params.get("two_gap", 0.7)
        self._score_three_gap = formality_params.get("three_gap", 0.4)
        self._score_severe = formality_params.get("severe_gap", 0.2)
    
    def check_formality_consistency(self, items: List[Garment]) -> float:
        """
        Check formality consistency across all items.
        
        Args:
            items: List of garments
            
        Returns:
            Consistency score between 0 and 1
        """
        if len(items) < 2:
            return 1.0
        
        # Get formality scores
        scores = [self.formality_scores[i.attributes.formality_level] 
                 for i in items]
        
        # Calculate range
        min_score = min(scores)
        max_score = max(scores)
        gap = max_score - min_score
        
        # Score based on gap
        if gap == 0:
            return 1.0  # Perfect match
        elif gap == 1:
            return 0.9  # Slight variation, very acceptable
        elif gap == 2:
            return 0.75  # Noticeable but workable
        elif gap == 3:
            return 0.5  # Significant mismatch
        else:
            return 0.3  # Severe mismatch
    
    def score_pair(
        self,
        formality1: FormalityLevel,
        formality2: FormalityLevel
    ) -> float:
        """
        Score formality compatibility between two items.
        
        Args:
            formality1: Formality level of first item
            formality2: Formality level of second item
            
        Returns:
            Compatibility score
        """
        score1 = self.formality_scores[formality1]
        score2 = self.formality_scores[formality2]
        
        gap = abs(score1 - score2)
        
        if gap == 0:
            return 1.0
        elif gap == 1:
            return 0.9
        elif gap == 2:
            return 0.7
        elif gap == 3:
            return 0.4
        else:
            return 0.2
    
    def get_target_formality(
        self,
        items: List[Garment]
    ) -> FormalityLevel:
        """
        Determine target formality level for outfit.
        
        Returns the most common/dominant formality level.
        
        Args:
            items: Current items in outfit
            
        Returns:
            Target formality level
        """
        if not items:
            return FormalityLevel.CASUAL
        
        # Get all formality levels
        levels = [i.attributes.formality_level for i in items]
        
        # Return most common
        from collections import Counter
        most_common = Counter(levels).most_common(1)[0][0]
        
        return most_common
    
    def get_compatible_formalities(
        self,
        current_formality: FormalityLevel
    ) -> List[FormalityLevel]:
        """
        Get formality levels compatible with current level.
        
        Args:
            current_formality: The formality to match
            
        Returns:
            List of compatible formality levels
        """
        current_score = self.formality_scores[current_formality]
        
        compatible = []
        for level, score in self.formality_scores.items():
            if abs(score - current_score) <= self.max_gap:
                compatible.append(level)
        
        return compatible
    
    def suggest_formality_adjustments(
        self,
        items: List[Garment]
    ) -> List[Dict]:
        """
        Suggest adjustments to improve formality consistency.
        
        Args:
            items: Current outfit items
            
        Returns:
            List of suggestions
        """
        if len(items) < 2:
            return []
        
        suggestions = []
        target = self.get_target_formality(items)
        target_score = self.formality_scores[target]
        
        for item in items:
            item_score = self.formality_scores[item.attributes.formality_level]
            gap = abs(item_score - target_score)
            
            if gap > self.max_gap:
                direction = "more formal" if item_score < target_score else "more casual"
                suggestions.append({
                    "item_id": item.id,
                    "current_formality": item.attributes.formality_level.value,
                    "target_formality": target.value,
                    "suggestion": f"Consider replacing with something {direction}"
                })
        
        return suggestions
    
    def is_appropriate_for_occasion(
        self,
        items: List[Garment],
        occasion_formality: FormalityLevel
    ) -> Tuple[bool, float]:
        """
        Check if outfit is appropriate for an occasion's formality.
        
        Args:
            items: Outfit items
            occasion_formality: Required formality for occasion
            
        Returns:
            Tuple of (is_appropriate, score)
        """
        if not items:
            return False, 0.0
        
        # Get average formality of outfit
        scores = [self.formality_scores[i.attributes.formality_level] 
                 for i in items]
        avg_score = sum(scores) / len(scores)
        
        occasion_score = self.formality_scores[occasion_formality]
        
        # Can dress slightly up, but not down
        diff = occasion_score - avg_score
        
        if diff <= 0.5:  # Outfit is formal enough
            if diff >= -1:  # Not too overdressed
                return True, 0.9
            else:  # Overdressed
                return True, 0.7
        elif diff <= 1.5:
            return True, 0.6  # Slightly underdressed
        else:
            return False, 0.3  # Too casual
