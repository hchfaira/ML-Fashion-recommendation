"""
Skin Contrast Rule Scorer
Analyzes the contrast between skin tone, hair color, and outfit colors.

| Contrast Type | Best Outfit Style                     | Score Logic            |
|---------------|---------------------------------------|------------------------|
| High          | Saturated, complementary colors       | +3 if complementary    |
| Low           | Monochrome, soft gradients            | +3 if monochrome       |
| Medium        | Balanced mix                          | +2 for either approach |
"""
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from src.core.models import Garment, ColorInfo
from src.core import get_logger

logger = get_logger(__name__)


class ContrastType(str, Enum):
    """User's natural contrast type based on skin/hair combination."""
    HIGH = "high"      # Light skin + dark hair OR dark skin + light hair
    MEDIUM = "medium"  # Moderate contrast between features
    LOW = "low"        # Similar tones: blonde + fair OR dark skin + dark hair


@dataclass
class SkinContrastResult:
    """Result from skin contrast analysis."""
    score: float  # 0-1 normalized
    points: int   # Raw points earned
    contrast_type: ContrastType
    outfit_style: str  # "monochrome", "complementary", "neutral", etc.
    is_flattering: bool
    recommendation: str


class SkinContrastScorer:
    """
    Scores outfits based on how well they match the user's natural contrast.
    
    High Contrast (e.g., Snow White): Light skin + dark hair
    - Best with: Bold colors, black/white, saturated hues
    - Avoid: Washed out pastels, muddy tones
    
    Low Contrast (e.g., natural blonde, deep skin tones):
    - Best with: Monochrome, tonal dressing, soft gradients
    - Avoid: Harsh contrasting colors that overpower
    """
    
    # Neutral colors that don't count in contrast analysis
    NEUTRAL_COLORS = {
        "white", "black", "gray", "grey", "beige", "cream", 
        "ivory", "taupe", "charcoal", "off-white", "nude"
    }
    
    # High saturation colors for high contrast individuals
    HIGH_SATURATION_COLORS = {
        "red", "cobalt", "royal blue", "emerald", "fuchsia", 
        "purple", "orange", "turquoise", "hot pink", "electric blue",
        "crimson", "scarlet", "magenta", "violet"
    }
    
    # Soft/muted colors for low contrast individuals
    SOFT_COLORS = {
        "dusty rose", "sage", "powder blue", "lavender", "blush",
        "mauve", "seafoam", "champagne", "soft pink", "light blue",
        "mint", "peach", "coral", "periwinkle", "dusty blue"
    }
    
    # Complementary color pairs (simplified)
    COMPLEMENTARY_PAIRS = [
        ("blue", "orange"),
        ("red", "green"),
        ("yellow", "purple"),
        ("pink", "green"),
        ("turquoise", "coral"),
        ("navy", "gold"),
    ]
    
    def __init__(self):
        self.max_points = 5
    
    def analyze_outfit(
        self,
        items: List[Garment],
        user_contrast: ContrastType
    ) -> SkinContrastResult:
        """
        Analyze how well an outfit matches the user's contrast type.
        
        Args:
            items: Outfit garments
            user_contrast: User's skin/hair contrast level
            
        Returns:
            SkinContrastResult with score and recommendations
        """
        if not items:
            return SkinContrastResult(
                score=0.5,
                points=0,
                contrast_type=user_contrast,
                outfit_style="unknown",
                is_flattering=False,
                recommendation="Add items to analyze"
            )
        
        # Analyze outfit color characteristics
        colors = self._extract_colors(items)
        is_monochrome = self._is_monochrome(colors)
        has_complementary = self._has_complementary_colors(colors)
        has_high_saturation = self._has_high_saturation(colors)
        is_tonal = self._is_tonal(colors)
        
        # Determine outfit style
        outfit_style = self._determine_outfit_style(
            is_monochrome, has_complementary, has_high_saturation, is_tonal
        )
        
        # Score based on contrast type
        points, is_flattering = self._calculate_points(
            user_contrast, 
            is_monochrome, 
            has_complementary,
            has_high_saturation,
            is_tonal
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            user_contrast, outfit_style, is_flattering, points
        )
        
        return SkinContrastResult(
            score=points / self.max_points,
            points=points,
            contrast_type=user_contrast,
            outfit_style=outfit_style,
            is_flattering=is_flattering,
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
        return colors
    
    def _is_monochrome(self, colors: List[str]) -> bool:
        """Check if outfit is monochromatic (same color family or all neutrals)."""
        non_neutral = [c for c in colors if c not in self.NEUTRAL_COLORS]
        
        if not non_neutral:
            return True  # All neutrals = monochrome
        
        # Check if all non-neutral colors are similar
        if len(set(non_neutral)) == 1:
            return True
        
        # Check for tonal variations (e.g., light blue, navy, sky blue)
        base_colors = set()
        for color in non_neutral:
            for base in ["blue", "red", "green", "pink", "purple", "brown", "orange", "yellow"]:
                if base in color:
                    base_colors.add(base)
                    break
        
        return len(base_colors) <= 1
    
    def _is_tonal(self, colors: List[str]) -> bool:
        """Check if outfit uses tonal dressing (variations of same hue)."""
        non_neutral = [c for c in colors if c not in self.NEUTRAL_COLORS]
        
        if len(non_neutral) < 2:
            return True
        
        # Look for shade variations (light/dark of same color)
        color_families = {}
        for color in non_neutral:
            for base in ["blue", "red", "green", "pink", "purple", "brown", "orange", "yellow"]:
                if base in color:
                    color_families[base] = color_families.get(base, 0) + 1
                    break
        
        # If most colors belong to same family
        if color_families:
            max_family = max(color_families.values())
            return max_family >= len(non_neutral) * 0.7
        
        return False
    
    def _has_complementary_colors(self, colors: List[str]) -> bool:
        """Check if outfit contains complementary color pairs."""
        non_neutral = [c for c in colors if c not in self.NEUTRAL_COLORS]
        
        for c1, c2 in self.COMPLEMENTARY_PAIRS:
            has_c1 = any(c1 in color for color in non_neutral)
            has_c2 = any(c2 in color for color in non_neutral)
            if has_c1 and has_c2:
                return True
        
        return False
    
    def _has_high_saturation(self, colors: List[str]) -> bool:
        """Check if outfit has high saturation/vibrant colors."""
        non_neutral = [c for c in colors if c not in self.NEUTRAL_COLORS]
        
        high_sat_count = sum(
            1 for c in non_neutral 
            if c in self.HIGH_SATURATION_COLORS or any(
                sat in c for sat in ["bright", "electric", "hot", "vivid", "bold"]
            )
        )
        
        return high_sat_count >= 1
    
    def _determine_outfit_style(
        self,
        is_monochrome: bool,
        has_complementary: bool,
        has_high_saturation: bool,
        is_tonal: bool
    ) -> str:
        """Determine the overall color style of the outfit."""
        if is_monochrome and not has_high_saturation:
            return "monochrome_soft"
        elif is_monochrome and has_high_saturation:
            return "monochrome_bold"
        elif has_complementary:
            return "complementary"
        elif is_tonal:
            return "tonal"
        elif has_high_saturation:
            return "bold_mixed"
        else:
            return "neutral"
    
    def _calculate_points(
        self,
        contrast_type: ContrastType,
        is_monochrome: bool,
        has_complementary: bool,
        has_high_saturation: bool,
        is_tonal: bool
    ) -> tuple:
        """Calculate points based on contrast type and outfit characteristics."""
        points = 0
        is_flattering = False
        
        if contrast_type == ContrastType.HIGH:
            # High contrast people look great in bold, saturated looks
            if has_complementary:
                points += 3
                is_flattering = True
            if has_high_saturation:
                points += 2
                is_flattering = True
            if is_monochrome and not has_high_saturation:
                # Can work but not optimal
                points += 1
                
        elif contrast_type == ContrastType.LOW:
            # Low contrast people shine in monochrome and tonal
            if is_monochrome:
                points += 3
                is_flattering = True
            if is_tonal:
                points += 2
                is_flattering = True
            if has_complementary or has_high_saturation:
                # Can overpower natural coloring
                points += 0
                
        else:  # MEDIUM
            # Medium contrast is versatile
            if is_monochrome or is_tonal:
                points += 2
                is_flattering = True
            if has_complementary:
                points += 2
                is_flattering = True
            if has_high_saturation:
                points += 1
        
        # Cap at max points
        points = min(points, self.max_points)
        
        return points, is_flattering
    
    def _generate_recommendation(
        self,
        contrast_type: ContrastType,
        outfit_style: str,
        is_flattering: bool,
        points: int
    ) -> str:
        """Generate personalized recommendation."""
        if is_flattering:
            if contrast_type == ContrastType.HIGH:
                return f"Great choice! The {outfit_style.replace('_', ' ')} style highlights your natural contrast."
            elif contrast_type == ContrastType.LOW:
                return f"Perfect! The {outfit_style.replace('_', ' ')} approach complements your soft coloring."
            else:
                return f"Well balanced! This {outfit_style.replace('_', ' ')} look works well with your features."
        else:
            if contrast_type == ContrastType.HIGH:
                return "Try adding bolder colors or a black/white element to match your high contrast features."
            elif contrast_type == ContrastType.LOW:
                return "Consider a more monochromatic or tonal approach to complement your softer natural coloring."
            else:
                return "You can go either bold or soft - experiment with what feels right!"


# Convenience function
def score_skin_contrast(
    items: List[Garment],
    user_contrast: ContrastType
) -> float:
    """Quick function to get skin contrast score."""
    scorer = SkinContrastScorer()
    result = scorer.analyze_outfit(items, user_contrast)
    return result.score
