"""
Color Harmony Analyzer
Analyzes color relationships and harmony in outfits.
"""
from typing import List, Dict, Tuple, Optional, Set
import colorsys
import re

from config import get_config
from src.core.models import Garment, ColorInfo
from src.core import get_logger

logger = get_logger(__name__)


class ColorHarmonyAnalyzer:
    """
    Analyzes color harmony in outfits.
    
    Implements color theory principles:
    - Complementary colors
    - Analogous colors
    - Triadic colors
    - Monochromatic schemes
    - Neutral combinations
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Load color data from configuration
        color_data = self.config.get_data("color_data", default={})
        
        # Named color to HSL mapping from config
        self.color_hsl_map = color_data.get("color_hsl_map", self._initialize_color_map())
        
        # Neutral colors from config
        self.neutrals: Set[str] = set(color_data.get("neutrals", [
            "black", "white", "gray", "grey", "beige", "cream",
            "ivory", "tan", "khaki", "navy", "brown", "nude", "taupe"
        ]))
        
        # Harmony rules from config
        harmony_rules = color_data.get("harmony_rules", {})
        self._monochromatic_threshold = harmony_rules.get("monochromatic_threshold", 0.083)
        self._complementary_min = harmony_rules.get("complementary_min", 0.42)
        self._complementary_max = harmony_rules.get("complementary_max", 0.58)
        self._analogous_threshold = harmony_rules.get("analogous_threshold", 0.167)
        
        # Safe defaults from config
        self._safe_complementary_defaults = color_data.get(
            "safe_complementary_defaults", 
            ["navy", "white", "black"]
        )
        
        # Load scoring parameters
        scoring_params = self.config.get_parameters("model_parameters", "scoring.color_harmony", default={})
        self._score_all_neutrals = scoring_params.get("all_neutrals", 0.85)
        self._score_neutrals_accent = scoring_params.get("neutrals_with_accent", 0.9)
        self._score_monochromatic = scoring_params.get("monochromatic", 0.9)
        self._score_complementary = scoring_params.get("complementary", 0.85)
        self._score_analogous = scoring_params.get("analogous", 0.85)
        self._score_triadic = scoring_params.get("triadic", 0.8)
        self._score_clash = scoring_params.get("clash", 0.6)
        self._score_default = scoring_params.get("default", 0.7)
        self._neutral_bonus_per_item = scoring_params.get("neutral_bonus_per_item", 0.05)
        self._max_neutral_bonus = scoring_params.get("max_neutral_bonus", 0.1)
    
    def analyze_outfit_colors(self, items: List[Garment]) -> float:
        """
        Analyze color harmony of entire outfit.
        
        Args:
            items: List of garments in the outfit
            
        Returns:
            Color harmony score between 0 and 1
        """
        if len(items) < 2:
            return 0.7
        
        # Extract all colors
        colors = []
        for item in items:
            colors.append(item.attributes.color.primary.lower())
            if item.attributes.color.secondary:
                colors.append(item.attributes.color.secondary.lower())
        
        # Calculate harmony
        neutral_count = sum(1 for c in colors if c in self.neutrals)
        non_neutral_colors = [c for c in colors if c not in self.neutrals]
        
        # All neutrals = safe and harmonious
        if len(non_neutral_colors) == 0:
            return 0.85
        
        # Many neutrals + 1-2 accent colors = good
        if neutral_count >= len(colors) - 2:
            return 0.9
        
        # Check color relationships
        harmony_score = self._analyze_color_relationships(non_neutral_colors)
        
        # Bonus for neutral base
        neutral_bonus = min(0.1, neutral_count * 0.05)
        
        return min(1.0, harmony_score + neutral_bonus)
    
    def score_color_pair(self, color1: ColorInfo, color2: ColorInfo) -> float:
        """
        Score color compatibility between two items.
        
        Args:
            color1: First color info
            color2: Second color info
            
        Returns:
            Compatibility score
        """
        c1 = color1.primary.lower()
        c2 = color2.primary.lower()
        
        # Same color = monochromatic (good)
        if c1 == c2:
            return 0.85
        
        # Both neutrals = always works
        if c1 in self.neutrals and c2 in self.neutrals:
            return 0.9
        
        # One neutral = usually works
        if c1 in self.neutrals or c2 in self.neutrals:
            return 0.85
        
        # Check color relationship
        return self._get_color_relationship_score(c1, c2)
    
    def get_complementary_colors(self, color: str) -> List[str]:
        """Get colors that complement the given color."""
        color = color.lower()
        
        if color in self.neutrals:
            return ["any color works with neutrals"]
        
        # Get HSL and find complementary
        hsl = self.color_hsl_map.get(color)
        if not hsl:
            return ["navy", "white", "black"]  # Safe defaults
        
        h, s, l = hsl
        
        # Complementary (opposite on wheel)
        comp_h = (h + 0.5) % 1.0
        
        # Find closest named color to complementary
        complementary = self._find_closest_color(comp_h, s, l)
        
        # Neutrals always work
        return [complementary] + ["white", "black", "gray"]
    
    def _analyze_color_relationships(self, colors: List[str]) -> float:
        """Analyze relationships between multiple colors."""
        if not colors:
            return 0.7
        
        if len(colors) == 1:
            return 0.9  # Single color accent
        
        # Convert to HSL values
        hsl_values = []
        for color in colors:
            hsl = self.color_hsl_map.get(color)
            if hsl:
                hsl_values.append(hsl)
        
        if not hsl_values:
            return 0.6  # Unknown colors
        
        # Check for harmony patterns
        hues = [h for h, s, l in hsl_values]
        
        # Monochromatic (similar hues)
        if self._is_monochromatic(hues):
            return 0.9
        
        # Complementary (opposite hues)
        if self._is_complementary(hues):
            return 0.85
        
        # Analogous (adjacent hues)
        if self._is_analogous(hues):
            return 0.85
        
        # Triadic
        if self._is_triadic(hues):
            return 0.8
        
        # Default - check clash potential
        return 0.6 if self._colors_clash(hues) else 0.7
    
    def _is_monochromatic(self, hues: List[float]) -> bool:
        """Check if hues form a monochromatic scheme."""
        if len(hues) < 2:
            return True
        
        # All hues within 30 degrees
        return max(hues) - min(hues) < 0.083  # 30/360
    
    def _is_complementary(self, hues: List[float]) -> bool:
        """Check if hues are complementary."""
        if len(hues) != 2:
            return False
        
        diff = abs(hues[0] - hues[1])
        # Opposite on color wheel (180 degrees ± 30)
        return 0.42 < diff < 0.58 or diff > 0.92 or diff < 0.08
    
    def _is_analogous(self, hues: List[float]) -> bool:
        """Check if hues are analogous (adjacent on wheel)."""
        if len(hues) < 2:
            return True
        
        sorted_hues = sorted(hues)
        max_gap = 0
        for i in range(len(sorted_hues) - 1):
            gap = sorted_hues[i+1] - sorted_hues[i]
            max_gap = max(max_gap, gap)
        
        # All within 60 degrees
        return max_gap < 0.167  # 60/360
    
    def _is_triadic(self, hues: List[float]) -> bool:
        """Check if hues form a triadic scheme."""
        if len(hues) != 3:
            return False
        
        sorted_hues = sorted(hues)
        gap1 = sorted_hues[1] - sorted_hues[0]
        gap2 = sorted_hues[2] - sorted_hues[1]
        
        # Each gap should be roughly 120 degrees
        target = 0.333
        tolerance = 0.05
        
        return (abs(gap1 - target) < tolerance and 
                abs(gap2 - target) < tolerance)
    
    def _colors_clash(self, hues: List[float]) -> bool:
        """Check if colors potentially clash."""
        # Colors that are close but not close enough for analogous
        # can clash (roughly 45-75 degrees apart)
        for i, h1 in enumerate(hues):
            for h2 in hues[i+1:]:
                diff = abs(h1 - h2)
                if 0.125 < diff < 0.21:  # 45-75 degrees
                    return True
        return False
    
    def _get_color_relationship_score(self, c1: str, c2: str) -> float:
        """Get compatibility score based on color relationship."""
        hsl1 = self.color_hsl_map.get(c1)
        hsl2 = self.color_hsl_map.get(c2)
        
        if not hsl1 or not hsl2:
            return 0.6
        
        hues = [hsl1[0], hsl2[0]]
        
        if self._is_complementary(hues):
            return 0.85
        if self._is_analogous(hues):
            return 0.8
        if self._colors_clash(hues):
            return 0.4
        
        return 0.65
    
    def _find_closest_color(self, h: float, s: float, l: float) -> str:
        """Find the closest named color to given HSL values."""
        min_dist = float('inf')
        closest = "blue"  # Default
        
        for name, (ch, cs, cl) in self.color_hsl_map.items():
            # Hue is circular, so handle wraparound
            hue_diff = min(abs(h - ch), 1 - abs(h - ch))
            dist = hue_diff * 3 + abs(s - cs) + abs(l - cl)
            
            if dist < min_dist:
                min_dist = dist
                closest = name
        
        return closest
    
    def _initialize_color_map(self) -> Dict[str, Tuple[float, float, float]]:
        """Initialize color name to HSL mapping."""
        return {
            # Reds
            "red": (0.0, 1.0, 0.5),
            "burgundy": (0.95, 0.8, 0.3),
            "maroon": (0.0, 1.0, 0.25),
            "coral": (0.04, 1.0, 0.65),
            "salmon": (0.02, 0.9, 0.7),
            
            # Oranges
            "orange": (0.08, 1.0, 0.5),
            "peach": (0.06, 0.9, 0.75),
            "rust": (0.05, 0.8, 0.35),
            "terracotta": (0.05, 0.6, 0.45),
            
            # Yellows
            "yellow": (0.17, 1.0, 0.5),
            "gold": (0.14, 1.0, 0.45),
            "mustard": (0.13, 0.7, 0.45),
            
            # Greens
            "green": (0.33, 1.0, 0.35),
            "olive": (0.22, 0.6, 0.35),
            "sage": (0.28, 0.3, 0.55),
            "mint": (0.42, 0.5, 0.75),
            "emerald": (0.42, 0.9, 0.4),
            "forest": (0.33, 0.7, 0.25),
            "teal": (0.5, 1.0, 0.35),
            
            # Blues
            "blue": (0.58, 1.0, 0.5),
            "light blue": (0.55, 0.7, 0.7),
            "sky blue": (0.55, 0.8, 0.65),
            "royal blue": (0.61, 1.0, 0.45),
            "cobalt": (0.6, 1.0, 0.35),
            "powder blue": (0.55, 0.5, 0.8),
            
            # Purples
            "purple": (0.75, 1.0, 0.5),
            "lavender": (0.72, 0.5, 0.75),
            "violet": (0.78, 0.9, 0.5),
            "plum": (0.83, 0.6, 0.35),
            "mauve": (0.88, 0.3, 0.6),
            
            # Pinks
            "pink": (0.92, 1.0, 0.75),
            "hot pink": (0.92, 1.0, 0.55),
            "blush": (0.97, 0.5, 0.85),
            "rose": (0.95, 0.6, 0.55),
            "magenta": (0.83, 1.0, 0.5),
            "fuchsia": (0.83, 1.0, 0.5),
        }
