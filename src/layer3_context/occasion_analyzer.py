"""
Occasion Analyzer
Analyzes outfit appropriateness for different occasions.
"""
from typing import List, Dict
from enum import Enum

from src.core.models import Garment, Occasion, FormalityLevel, GarmentCategory
from src.core import get_logger

logger = get_logger(__name__)


class OccasionAnalyzer:
    """
    Analyzes outfit suitability for occasions.
    
    Maps occasions to expected clothing characteristics
    and scores outfits accordingly.
    """
    
    def __init__(self):
        # Occasion to formality mapping
        self.occasion_formality = {
            Occasion.CASUAL: FormalityLevel.CASUAL,
            Occasion.BUSINESS: FormalityLevel.BUSINESS,
            Occasion.FORMAL: FormalityLevel.FORMAL,
            Occasion.SPORT: FormalityLevel.VERY_CASUAL,
            Occasion.EVENING: FormalityLevel.SMART_CASUAL,
            Occasion.DATE: FormalityLevel.SMART_CASUAL,
            Occasion.BEACH: FormalityLevel.VERY_CASUAL
        }
        
        # Occasion-specific style preferences
        self.occasion_styles = {
            Occasion.CASUAL: ["relaxed", "comfortable", "everyday", "casual"],
            Occasion.BUSINESS: ["professional", "polished", "clean", "tailored"],
            Occasion.FORMAL: ["elegant", "sophisticated", "luxurious", "refined"],
            Occasion.SPORT: ["athletic", "sporty", "functional", "activewear"],
            Occasion.EVENING: ["glamorous", "chic", "trendy", "statement"],
            Occasion.DATE: ["romantic", "stylish", "flattering", "confident"],
            Occasion.BEACH: ["relaxed", "casual", "comfortable", "light"]
        }
        
        # Categories to avoid per occasion
        self.occasion_avoid = {
            Occasion.BUSINESS: ["athletic", "beachwear", "very_casual"],
            Occasion.FORMAL: ["athletic", "casual", "streetwear"],
            Occasion.SPORT: ["formal", "delicate", "restrictive"],
            Occasion.BEACH: ["formal", "heavy", "restrictive"]
        }
    
    def score_for_occasion(
        self,
        garments: List[Garment],
        occasion: Occasion
    ) -> float:
        """
        Score outfit appropriateness for occasion.
        
        Args:
            garments: List of garments in outfit
            occasion: Target occasion
            
        Returns:
            Score between 0 and 1
        """
        if not garments:
            return 0.5
        
        scores = []
        
        for garment in garments:
            # Formality alignment
            formality_score = self._score_formality_match(
                garment.attributes.formality_level,
                self.occasion_formality.get(occasion, FormalityLevel.CASUAL)
            )
            
            # Style tag alignment
            style_score = self._score_style_match(
                garment.attributes.style_tags,
                self.occasion_styles.get(occasion, [])
            )
            
            # Check for avoided styles
            avoid_penalty = self._check_avoided_styles(
                garment.attributes.style_tags,
                occasion
            )
            
            # Combine scores
            item_score = (0.5 * formality_score + 0.5 * style_score) * avoid_penalty
            scores.append(item_score)
        
        return sum(scores) / len(scores)
    
    def get_occasion_requirements(
        self,
        occasion: Occasion
    ) -> Dict:
        """
        Get clothing requirements for an occasion.
        
        Args:
            occasion: Target occasion
            
        Returns:
            Dictionary with requirements
        """
        return {
            "target_formality": self.occasion_formality.get(
                occasion, FormalityLevel.CASUAL
            ).value,
            "preferred_styles": self.occasion_styles.get(occasion, []),
            "avoid": self.occasion_avoid.get(occasion, []),
            "suggested_categories": self._get_suggested_categories(occasion)
        }
    
    def suggest_improvements(
        self,
        garments: List[Garment],
        occasion: Occasion
    ) -> List[str]:
        """
        Suggest improvements for occasion appropriateness.
        
        Args:
            garments: Current outfit
            occasion: Target occasion
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        target_formality = self.occasion_formality.get(occasion, FormalityLevel.CASUAL)
        
        for garment in garments:
            # Check formality
            if self._formality_gap(garment.attributes.formality_level, target_formality) > 1:
                if self._formality_value(garment.attributes.formality_level) < self._formality_value(target_formality):
                    suggestions.append(
                        f"Consider a more formal {garment.attributes.category.value}"
                    )
                else:
                    suggestions.append(
                        f"The {garment.attributes.category.value} might be too formal"
                    )
            
            # Check style conflicts
            avoid_list = self.occasion_avoid.get(occasion, [])
            conflicts = set(garment.attributes.style_tags) & set(avoid_list)
            if conflicts:
                suggestions.append(
                    f"The {garment.attributes.category.value} has styles "
                    f"({', '.join(conflicts)}) not ideal for {occasion.value}"
                )
        
        return suggestions
    
    def _score_formality_match(
        self,
        item_formality: FormalityLevel,
        target_formality: FormalityLevel
    ) -> float:
        """Score formality alignment."""
        gap = self._formality_gap(item_formality, target_formality)
        
        if gap == 0:
            return 1.0
        elif gap == 1:
            return 0.85
        elif gap == 2:
            return 0.6
        else:
            return 0.3
    
    def _score_style_match(
        self,
        item_styles: List[str],
        preferred_styles: List[str]
    ) -> float:
        """Score style tag alignment."""
        if not item_styles or not preferred_styles:
            return 0.6  # Neutral
        
        item_set = set(s.lower() for s in item_styles)
        preferred_set = set(s.lower() for s in preferred_styles)
        
        overlap = len(item_set & preferred_set)
        
        if overlap >= 2:
            return 1.0
        elif overlap == 1:
            return 0.8
        else:
            return 0.5
    
    def _check_avoided_styles(
        self,
        item_styles: List[str],
        occasion: Occasion
    ) -> float:
        """Return penalty multiplier for avoided styles."""
        avoid_list = self.occasion_avoid.get(occasion, [])
        if not avoid_list:
            return 1.0
        
        item_set = set(s.lower() for s in item_styles)
        avoid_set = set(s.lower() for s in avoid_list)
        
        conflicts = len(item_set & avoid_set)
        
        if conflicts == 0:
            return 1.0
        elif conflicts == 1:
            return 0.7
        else:
            return 0.4
    
    def _formality_gap(
        self,
        f1: FormalityLevel,
        f2: FormalityLevel
    ) -> int:
        """Calculate formality level gap."""
        order = [
            FormalityLevel.VERY_CASUAL,
            FormalityLevel.CASUAL,
            FormalityLevel.SMART_CASUAL,
            FormalityLevel.BUSINESS_CASUAL,
            FormalityLevel.BUSINESS,
            FormalityLevel.FORMAL,
            FormalityLevel.BLACK_TIE
        ]
        return abs(order.index(f1) - order.index(f2))
    
    def _formality_value(self, f: FormalityLevel) -> int:
        """Get numeric value for formality level."""
        order = [
            FormalityLevel.VERY_CASUAL,
            FormalityLevel.CASUAL,
            FormalityLevel.SMART_CASUAL,
            FormalityLevel.BUSINESS_CASUAL,
            FormalityLevel.BUSINESS,
            FormalityLevel.FORMAL,
            FormalityLevel.BLACK_TIE
        ]
        return order.index(f)
    
    def _get_suggested_categories(self, occasion: Occasion) -> List[str]:
        """Get suggested garment categories for occasion."""
        suggestions = {
            Occasion.CASUAL: ["top", "bottom", "sneakers", "jacket"],
            Occasion.BUSINESS: ["blazer", "dress_shirt", "trousers", "oxford_shoes"],
            Occasion.FORMAL: ["suit", "dress_shirt", "tie", "dress_shoes"],
            Occasion.SPORT: ["athletic_top", "athletic_bottom", "sneakers"],
            Occasion.EVENING: ["dress", "heels", "clutch", "statement_jewelry"],
            Occasion.DATE: ["nice_top", "jeans", "stylish_shoes"],
            Occasion.BEACH: ["swimwear", "coverup", "sandals", "sunglasses"]
        }
        return suggestions.get(occasion, ["top", "bottom", "shoes"])
