"""
Sandwich Rule Scorer
The classic TikTok/Instagram styling rule: match top color with shoes,
creating a visual "sandwich" with the bottom as the filling.

| Condition                          | Points |
|------------------------------------|--------|
| Top color == Shoes color           | +2     |
| Bottom is different from both      | +1     |
| Perfect sandwich (both conditions) | +3     |
"""
from typing import List, Optional, Tuple
from dataclasses import dataclass

from src.core.models import Garment, GarmentCategory
from src.core import get_logger

logger = get_logger(__name__)


@dataclass
class SandwichResult:
    """Result from sandwich rule analysis."""
    score: float  # 0-1 normalized
    points: int   # Raw points earned (max 3)
    has_sandwich: bool
    top_color: Optional[str]
    bottom_color: Optional[str]
    shoes_color: Optional[str]
    recommendation: str


class SandwichRuleScorer:
    """
    Scores outfits based on the "Sandwich Rule" for visual balance.
    
    The Rule:
    - Top and shoes should match in color
    - Bottom should be a different color (the "filling")
    - Creates visual cohesion and elongates the silhouette
    
    This is one of the most popular styling tips on social media because:
    1. It's easy to remember
    2. It creates instant polish
    3. It works with almost any style
    """
    
    # Colors considered "matching" even if not identical
    COLOR_FAMILIES = {
        "blue": ["blue", "navy", "cobalt", "royal blue", "sky blue", "denim", "indigo"],
        "black": ["black", "jet black", "onyx", "charcoal"],
        "white": ["white", "off-white", "cream", "ivory", "ecru"],
        "brown": ["brown", "tan", "camel", "cognac", "chocolate", "chestnut", "rust"],
        "green": ["green", "olive", "sage", "forest", "emerald", "khaki", "moss"],
        "red": ["red", "burgundy", "wine", "crimson", "maroon", "scarlet"],
        "pink": ["pink", "blush", "rose", "fuchsia", "coral", "salmon"],
        "gray": ["gray", "grey", "charcoal", "silver", "slate"],
        "beige": ["beige", "nude", "taupe", "sand", "camel", "tan"],
    }
    
    def __init__(self):
        self.max_points = 3
    
    def analyze_outfit(self, items: List[Garment]) -> SandwichResult:
        """
        Analyze outfit for sandwich rule compliance.
        
        Args:
            items: Outfit garments
            
        Returns:
            SandwichResult with score and analysis
        """
        # Find top, bottom, and shoes
        top = self._find_item_by_category(items, [GarmentCategory.TOP, GarmentCategory.OUTERWEAR])
        bottom = self._find_item_by_category(items, [GarmentCategory.BOTTOM])
        shoes = self._find_item_by_category(items, [GarmentCategory.SHOES])
        
        # Extract colors
        top_color = self._get_primary_color(top) if top else None
        bottom_color = self._get_primary_color(bottom) if bottom else None
        shoes_color = self._get_primary_color(shoes) if shoes else None
        
        # Check for missing items
        if not all([top_color, bottom_color, shoes_color]):
            return SandwichResult(
                score=0.5,
                points=0,
                has_sandwich=False,
                top_color=top_color,
                bottom_color=bottom_color,
                shoes_color=shoes_color,
                recommendation="Need top, bottom, and shoes to evaluate sandwich rule"
            )
        
        # Calculate sandwich score
        points = 0
        
        # Check if top and shoes match
        top_shoes_match = self._colors_match(top_color, shoes_color)
        if top_shoes_match:
            points += 2
        
        # Check if bottom is different (the "filling")
        bottom_different = (
            not self._colors_match(bottom_color, top_color) and
            not self._colors_match(bottom_color, shoes_color)
        )
        if bottom_different and top_shoes_match:
            points += 1
        
        has_sandwich = points >= 2
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            has_sandwich, top_shoes_match, bottom_different,
            top_color, bottom_color, shoes_color
        )
        
        return SandwichResult(
            score=points / self.max_points,
            points=points,
            has_sandwich=has_sandwich,
            top_color=top_color,
            bottom_color=bottom_color,
            shoes_color=shoes_color,
            recommendation=recommendation
        )
    
    def _find_item_by_category(
        self, 
        items: List[Garment], 
        categories: List[GarmentCategory]
    ) -> Optional[Garment]:
        """Find first item matching any of the given categories."""
        for item in items:
            if item.attributes.category in categories:
                return item
        return None
    
    def _get_primary_color(self, item: Garment) -> Optional[str]:
        """Get the primary color of an item."""
        if item and item.attributes.color:
            return item.attributes.color.primary.lower()
        return None
    
    def _colors_match(self, color1: str, color2: str) -> bool:
        """Check if two colors match (same or same family)."""
        if not color1 or not color2:
            return False
        
        c1 = color1.lower()
        c2 = color2.lower()
        
        # Exact match
        if c1 == c2:
            return True
        
        # Same family match
        for family, members in self.COLOR_FAMILIES.items():
            c1_in_family = c1 in members or any(m in c1 for m in members)
            c2_in_family = c2 in members or any(m in c2 for m in members)
            if c1_in_family and c2_in_family:
                return True
        
        return False
    
    def _generate_recommendation(
        self,
        has_sandwich: bool,
        top_shoes_match: bool,
        bottom_different: bool,
        top_color: str,
        bottom_color: str,
        shoes_color: str
    ) -> str:
        """Generate recommendation based on analysis."""
        if has_sandwich:
            return f"Perfect sandwich! {top_color.title()} top + {bottom_color.title()} bottom + {shoes_color.title()} shoes creates visual balance."
        
        if not top_shoes_match:
            return f"Try {top_color} shoes instead of {shoes_color} to create a visual 'sandwich' effect."
        
        if not bottom_different:
            return f"Your bottom ({bottom_color}) matches your top/shoes. Try a contrasting color for the 'filling'."
        
        return "Mix up your colors to create a top-bottom-shoes sandwich effect."


# Convenience functions
def check_sandwich_rule(items: List[Garment]) -> bool:
    """Quick check if outfit follows sandwich rule."""
    scorer = SandwichRuleScorer()
    result = scorer.analyze_outfit(items)
    return result.has_sandwich


def get_sandwich_score(items: List[Garment]) -> float:
    """Get sandwich rule score (0-1)."""
    scorer = SandwichRuleScorer()
    result = scorer.analyze_outfit(items)
    return result.score
