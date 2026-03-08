"""
Three Color Rule Scorer
Classic styling rule: an outfit should not exceed 3 distinct colors
(excluding neutrals like black, white, gray).

| Unique Colors | Score   | Notes                              |
|---------------|---------|-------------------------------------|
| 1-2           | 5/5     | Clean, minimal                      |
| 3             | 5/5     | Perfect balance                     |
| 4             | 3/5     | Acceptable, slightly busy           |
| 5+            | 1/5     | Penalty (unless Maximalist style)   |
"""
from typing import List, Set, Optional
from dataclasses import dataclass
from enum import Enum

from src.core.models import Garment
from src.core import get_logger

logger = get_logger(__name__)


class StyleAesthetic(str, Enum):
    """User's style aesthetic preference."""
    MINIMALIST = "minimalist"
    CLASSIC = "classic"
    CASUAL = "casual"
    MAXIMALIST = "maximalist"  # More colors allowed
    ECLECTIC = "eclectic"      # More colors allowed
    BOHEMIAN = "bohemian"      # Patterns and colors mix OK


@dataclass
class ThreeColorResult:
    """Result from three color rule analysis."""
    score: float  # 0-1 normalized
    points: int   # Raw points (max 5)
    unique_colors: int
    color_list: List[str]
    neutrals_used: List[str]
    within_limit: bool
    recommendation: str


class ThreeColorScorer:
    """
    Scores outfits based on the Three Color Rule.
    
    The Rule:
    - Limit outfit to 3 distinct colors maximum
    - Neutrals (black, white, gray, beige, nude) don't count
    - Creates cohesion and avoids "clown" effect
    
    Exceptions:
    - Maximalist/Eclectic styles can break this rule intentionally
    - Prints within a single garment count as 1 "color"
    """
    
    # Neutral colors that don't count toward the limit
    NEUTRAL_COLORS = {
        "white", "black", "gray", "grey", "beige", "cream", "ivory",
        "nude", "taupe", "charcoal", "off-white", "ecru", "bone",
        "silver", "gold"  # Metallics often treated as neutrals
    }
    
    # Color families for grouping similar colors
    COLOR_FAMILIES = {
        "blue": ["blue", "navy", "cobalt", "royal blue", "sky blue", "teal", "turquoise", "indigo", "azure", "cerulean", "denim"],
        "red": ["red", "burgundy", "wine", "crimson", "scarlet", "maroon", "cherry", "ruby"],
        "pink": ["pink", "blush", "rose", "fuchsia", "magenta", "coral", "salmon", "mauve", "dusty rose"],
        "green": ["green", "olive", "sage", "forest", "emerald", "khaki", "moss", "mint", "lime", "hunter"],
        "yellow": ["yellow", "mustard", "gold", "lemon", "sunflower", "canary", "amber"],
        "orange": ["orange", "rust", "terracotta", "tangerine", "peach", "apricot", "copper"],
        "purple": ["purple", "lavender", "violet", "plum", "lilac", "eggplant", "amethyst", "periwinkle"],
        "brown": ["brown", "tan", "camel", "cognac", "chocolate", "chestnut", "mocha", "espresso", "coffee"],
    }
    
    def __init__(self):
        self.max_points = 5
        self.color_limit = 3
    
    def analyze_outfit(
        self,
        items: List[Garment],
        user_aesthetic: Optional[StyleAesthetic] = None
    ) -> ThreeColorResult:
        """
        Analyze outfit for three color rule compliance.
        
        Args:
            items: Outfit garments
            user_aesthetic: Optional user style preference
            
        Returns:
            ThreeColorResult with score and analysis
        """
        if not items:
            return ThreeColorResult(
                score=0.5,
                points=0,
                unique_colors=0,
                color_list=[],
                neutrals_used=[],
                within_limit=True,
                recommendation="Add items to analyze"
            )
        
        # Extract all colors
        all_colors = self._extract_colors(items)
        
        # Separate neutrals from non-neutrals
        neutrals_used = [c for c in all_colors if self._is_neutral(c)]
        non_neutrals = [c for c in all_colors if not self._is_neutral(c)]
        
        # Group into color families
        unique_families = self._count_color_families(non_neutrals)
        
        # Adjust limit for maximalist styles
        adjusted_limit = self.color_limit
        if user_aesthetic in [StyleAesthetic.MAXIMALIST, StyleAesthetic.ECLECTIC, StyleAesthetic.BOHEMIAN]:
            adjusted_limit = 5
        
        # Calculate score
        within_limit = unique_families <= adjusted_limit
        points = self._calculate_points(unique_families, adjusted_limit, user_aesthetic)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            unique_families, adjusted_limit, within_limit, non_neutrals, user_aesthetic
        )
        
        return ThreeColorResult(
            score=points / self.max_points,
            points=points,
            unique_colors=unique_families,
            color_list=list(set(non_neutrals)),
            neutrals_used=list(set(neutrals_used)),
            within_limit=within_limit,
            recommendation=recommendation
        )
    
    def _extract_colors(self, items: List[Garment]) -> List[str]:
        """Extract all colors from outfit items."""
        colors = []
        for item in items:
            if item.attributes.color:
                colors.append(item.attributes.color.primary.lower())
                if item.attributes.color.secondary:
                    colors.append(item.attributes.color.secondary.lower())
                if item.attributes.color.accent:
                    colors.append(item.attributes.color.accent.lower())
        return colors
    
    def _is_neutral(self, color: str) -> bool:
        """Check if a color is neutral."""
        color_lower = color.lower()
        return color_lower in self.NEUTRAL_COLORS or any(
            neutral in color_lower for neutral in self.NEUTRAL_COLORS
        )
    
    def _count_color_families(self, colors: List[str]) -> int:
        """Count distinct color families in the list."""
        families_found = set()
        unmatched = []
        
        for color in colors:
            matched = False
            for family, members in self.COLOR_FAMILIES.items():
                if color in members or any(m in color for m in members):
                    families_found.add(family)
                    matched = True
                    break
            
            if not matched:
                unmatched.append(color)
        
        # Each unmatched color counts as its own family
        return len(families_found) + len(set(unmatched))
    
    def _calculate_points(
        self,
        unique_colors: int,
        limit: int,
        aesthetic: Optional[StyleAesthetic]
    ) -> int:
        """Calculate points based on color count."""
        if unique_colors <= 2:
            return 5  # Perfect minimal
        elif unique_colors == 3:
            return 5  # Ideal balance
        elif unique_colors == 4:
            if aesthetic in [StyleAesthetic.MAXIMALIST, StyleAesthetic.ECLECTIC]:
                return 4  # Acceptable for these styles
            return 3
        elif unique_colors == 5:
            if aesthetic in [StyleAesthetic.MAXIMALIST, StyleAesthetic.ECLECTIC, StyleAesthetic.BOHEMIAN]:
                return 3
            return 1
        else:
            if aesthetic == StyleAesthetic.MAXIMALIST:
                return 2  # Still somewhat acceptable
            return 0  # Too many colors
    
    def _generate_recommendation(
        self,
        unique_colors: int,
        limit: int,
        within_limit: bool,
        colors: List[str],
        aesthetic: Optional[StyleAesthetic]
    ) -> str:
        """Generate recommendation based on analysis."""
        if within_limit:
            if unique_colors <= 2:
                return f"Elegant minimal palette with {unique_colors} color{'s' if unique_colors > 1 else ''}. Very chic!"
            elif unique_colors == 3:
                return "Perfect! Three colors is the ideal balance for a cohesive look."
            else:
                return f"Working {unique_colors} colors with confidence! Great for your {aesthetic.value if aesthetic else 'bold'} style."
        
        # Over limit
        excess = unique_colors - limit
        if colors:
            # Suggest which colors to consolidate
            return f"Consider reducing by {excess} color{'s' if excess > 1 else ''}. Try swapping {colors[-1]} for a neutral."
        
        return f"Too many colors ({unique_colors}). Aim for {limit} non-neutral colors maximum."


# Convenience functions
def check_three_color_rule(items: List[Garment]) -> bool:
    """Quick check if outfit follows three color rule."""
    scorer = ThreeColorScorer()
    result = scorer.analyze_outfit(items)
    return result.within_limit


def count_outfit_colors(items: List[Garment]) -> int:
    """Count unique non-neutral colors in outfit."""
    scorer = ThreeColorScorer()
    result = scorer.analyze_outfit(items)
    return result.unique_colors


def get_three_color_score(items: List[Garment]) -> float:
    """Get three color rule score (0-1)."""
    scorer = ThreeColorScorer()
    result = scorer.analyze_outfit(items)
    return result.score
