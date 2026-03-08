"""
Season Color Harmony Scorer
Determines if garment colors complement the user's skin undertone and color season.

Color Seasons:
- Spring (Warm, Light): Warm peachy tones, golden undertones
- Summer (Cool, Light): Cool pink tones, soft and muted colors  
- Autumn/Fall (Warm, Deep): Warm golden tones, rich earthy colors
- Winter (Cool, Deep): Cool blue/pink tones, high contrast colors
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.core.models import Garment, ColorInfo
from src.core import get_logger

logger = get_logger(__name__)


class SkinUndertone(str, Enum):
    """Skin undertone categories."""
    WARM = "warm"  # Yellow, golden, peachy undertones
    COOL = "cool"  # Pink, red, bluish undertones
    NEUTRAL = "neutral"  # Mix of warm and cool


class ColorSeason(str, Enum):
    """Color season categories based on skin tone analysis."""
    # Warm seasons
    SPRING = "spring"  # Warm + Light: Clear, warm, bright colors
    AUTUMN = "autumn"  # Warm + Deep: Muted, earthy, rich colors
    
    # Cool seasons
    SUMMER = "summer"  # Cool + Light: Soft, muted, cool colors
    WINTER = "winter"  # Cool + Deep: Bold, clear, high contrast


@dataclass
class SeasonColorResult:
    """Result of season color analysis."""
    score: float  # 0-1 normalized score
    points: int  # Raw points (-2 to +2 per item)
    matches: List[str]  # Colors that match
    clashes: List[str]  # Colors that clash
    recommendation: str


class SeasonColorHarmonyScorer:
    """
    Scores outfit colors based on user's color season.
    
    Scoring:
    - Match (+2 pts): Color belongs to user's season palette
    - Neutral (0 pts): Universal neutrals (white, navy, grey)
    - Clash (-2 pts): Color washes out or clashes with skin tone
    """
    
    def __init__(self):
        # Define palettes for each season
        self.season_palettes = self._initialize_palettes()
        
        # Universal neutrals that work for everyone
        self.universal_neutrals = {
            "white", "off-white", "ivory", "black", "navy", 
            "grey", "gray", "charcoal", "true red"
        }
        
        # Colors that clash with specific seasons
        self.season_clashes = self._initialize_clashes()
    
    def _initialize_palettes(self) -> Dict[ColorSeason, set]:
        """Initialize color palettes for each season."""
        return {
            ColorSeason.SPRING: {
                # Warm, clear, bright colors
                "coral", "peach", "salmon", "apricot", "warm pink",
                "golden yellow", "sunflower", "marigold", "butter yellow",
                "warm green", "lime", "apple green", "mint",
                "turquoise", "aqua", "light teal", "periwinkle",
                "camel", "warm beige", "tan", "golden brown",
                "ivory", "cream", "warm white"
            },
            ColorSeason.SUMMER: {
                # Cool, soft, muted colors
                "lavender", "soft pink", "rose", "mauve", "dusty rose",
                "powder blue", "sky blue", "soft blue", "periwinkle",
                "sage", "soft teal", "seafoam", "mint",
                "soft white", "dove grey", "blue-grey", "taupe",
                "plum", "raspberry", "soft burgundy",
                "cocoa", "rose brown", "cool beige"
            },
            ColorSeason.AUTUMN: {
                # Warm, muted, earthy colors
                "rust", "terracotta", "burnt orange", "pumpkin",
                "mustard", "gold", "amber", "honey",
                "olive", "moss", "forest green", "khaki",
                "burgundy", "wine", "brick red", "mahogany",
                "chocolate", "coffee", "caramel", "tan",
                "burnt sienna", "copper", "bronze",
                "cream", "warm beige", "oyster"
            },
            ColorSeason.WINTER: {
                # Cool, bold, clear colors
                "true red", "cherry", "crimson", "burgundy",
                "royal blue", "cobalt", "navy", "electric blue",
                "emerald", "pine", "forest green",
                "hot pink", "fuchsia", "magenta", "bright pink",
                "purple", "violet", "royal purple", "plum",
                "black", "white", "silver", "charcoal",
                "icy pink", "icy blue", "icy lavender"
            }
        }
    
    def _initialize_clashes(self) -> Dict[ColorSeason, set]:
        """Colors that typically clash with each season."""
        return {
            ColorSeason.SPRING: {
                # Muted, cool colors wash out Spring
                "dusty pink", "mauve", "burgundy", "wine",
                "charcoal", "black", "forest green", "olive drab"
            },
            ColorSeason.SUMMER: {
                # Bright, warm colors overwhelm Summer
                "bright orange", "golden yellow", "rust", "terracotta",
                "hot pink", "electric blue", "lime green", "true red"
            },
            ColorSeason.AUTUMN: {
                # Cool, icy colors clash with Autumn
                "icy pink", "icy blue", "lavender", "pastel pink",
                "silver", "grey", "powder blue", "soft pink"
            },
            ColorSeason.WINTER: {
                # Muted, warm earth tones look dull on Winter
                "mustard", "rust", "terracotta", "camel",
                "warm beige", "olive", "peach", "coral"
            }
        }
    
    def analyze_outfit(
        self,
        items: List[Garment],
        user_season: ColorSeason
    ) -> SeasonColorResult:
        """
        Analyze outfit colors for a user's color season.
        
        Args:
            items: Outfit garments
            user_season: User's color season
            
        Returns:
            SeasonColorResult with scoring and recommendations
        """
        if not items:
            return SeasonColorResult(
                score=0.5,
                points=0,
                matches=[],
                clashes=[],
                recommendation="Add items to analyze"
            )
        
        palette = self.season_palettes[user_season]
        clashes = self.season_clashes[user_season]
        
        total_points = 0
        matches = []
        clash_colors = []
        
        for item in items:
            color = item.attributes.color.primary.lower()
            
            # Check match
            if color in palette or self._is_similar_to_palette(color, palette):
                total_points += 2
                matches.append(color)
            # Check universal neutral
            elif color in self.universal_neutrals:
                total_points += 0  # Neutral
            # Check clash
            elif color in clashes or self._is_similar_to_palette(color, clashes):
                total_points -= 2
                clash_colors.append(color)
            else:
                # Unknown color - slight positive if warm/cool matches season
                if self._undertone_matches(color, user_season):
                    total_points += 1
                else:
                    total_points -= 1
        
        # Normalize score
        max_possible = len(items) * 2
        min_possible = len(items) * -2
        
        if max_possible == min_possible:
            score = 0.5
        else:
            score = (total_points - min_possible) / (max_possible - min_possible)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            user_season, matches, clash_colors, score
        )
        
        return SeasonColorResult(
            score=score,
            points=total_points,
            matches=matches,
            clashes=clash_colors,
            recommendation=recommendation
        )
    
    def get_best_colors(self, user_season: ColorSeason) -> List[str]:
        """Get the best colors for a user's season."""
        return list(self.season_palettes[user_season])[:10]
    
    def get_colors_to_avoid(self, user_season: ColorSeason) -> List[str]:
        """Get colors to avoid for a user's season."""
        return list(self.season_clashes[user_season])
    
    def _is_similar_to_palette(self, color: str, palette: set) -> bool:
        """Check if a color is similar to any in the palette."""
        # Simple substring matching for variations
        for palette_color in palette:
            if color in palette_color or palette_color in color:
                return True
        return False
    
    def _undertone_matches(self, color: str, season: ColorSeason) -> bool:
        """Check if color undertone matches season's warmth/coolness."""
        warm_indicators = {"yellow", "gold", "orange", "coral", "peach", "warm", "rust"}
        cool_indicators = {"blue", "pink", "purple", "lavender", "silver", "cool", "icy"}
        
        is_warm_color = any(ind in color for ind in warm_indicators)
        is_cool_color = any(ind in color for ind in cool_indicators)
        
        warm_seasons = {ColorSeason.SPRING, ColorSeason.AUTUMN}
        
        if season in warm_seasons:
            return is_warm_color and not is_cool_color
        else:
            return is_cool_color and not is_warm_color
    
    def _generate_recommendation(
        self,
        season: ColorSeason,
        matches: List[str],
        clashes: List[str],
        score: float
    ) -> str:
        """Generate styling recommendation."""
        season_name = season.value.capitalize()
        
        if score >= 0.8:
            return f"✅ Excellent! These colors perfectly complement your {season_name} palette."
        elif score >= 0.6:
            if clashes:
                return (f"⚠️ Good overall, but {', '.join(clashes)} may wash you out. "
                       f"Consider swapping for {', '.join(self.get_best_colors(season)[:3])}.")
            return f"👍 Nice color choices for your {season_name} palette."
        elif score >= 0.4:
            return (f"⚠️ Some colors don't suit your {season_name} coloring. "
                   f"Try: {', '.join(self.get_best_colors(season)[:5])}")
        else:
            return (f"❌ These colors may not be flattering for {season_name}. "
                   f"Your best colors: {', '.join(self.get_best_colors(season)[:5])}")


def analyze_season_harmony(
    items: List[Garment],
    user_season: ColorSeason
) -> float:
    """Quick function to get season harmony score."""
    scorer = SeasonColorHarmonyScorer()
    result = scorer.analyze_outfit(items, user_season)
    return result.score
