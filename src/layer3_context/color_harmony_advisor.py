"""
Color Harmony Advisor
=====================

Advises on color choices based on user's skin tone, undertone,
hair color, and personal contrast level.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

from config import get_config
from src.core.models import Garment

logger = logging.getLogger(__name__)


@dataclass
class ColorProfile:
    """User's color profile for styling."""
    skin_tone: str  # FAIR, LIGHT, MEDIUM, OLIVE, TAN, DARK, DEEP
    undertone: str  # WARM, COOL, NEUTRAL
    hair_color: str  # BLACK, BROWN, BLONDE, RED, etc.
    contrast_level: str  # LOW, MEDIUM, HIGH
    
    @property
    def season(self) -> str:
        """Determine color season (Spring, Summer, Autumn, Winter)."""
        if self.undertone == "WARM":
            if self.contrast_level in ["HIGH", "VERY_HIGH"]:
                return "AUTUMN"
            return "SPRING"
        elif self.undertone == "COOL":
            if self.contrast_level in ["HIGH", "VERY_HIGH"]:
                return "WINTER"
            return "SUMMER"
        else:  # NEUTRAL
            if self.contrast_level in ["HIGH", "VERY_HIGH"]:
                return "WINTER"
            return "SUMMER"


@dataclass
class ColorRecommendation:
    """Color recommendation results."""
    best_colors: List[str]
    good_colors: List[str]
    colors_to_avoid: List[str]
    neutral_colors: List[str]
    accent_colors: List[str]
    tips: List[str] = field(default_factory=list)


@dataclass
class ColorHarmonyScore:
    """Score for how well outfit colors harmonize with user."""
    score: float  # 0.0 to 1.0
    harmony_level: str  # "excellent", "good", "fair", "poor"
    matching_colors: List[str] = field(default_factory=list)
    clashing_colors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class ColorHarmonyAdvisor:
    """
    Advises on color choices based on user's coloring.
    
    Uses skin tone, undertone, hair color, and contrast level
    to recommend flattering colors for clothing.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize color harmony advisor.
        
        Args:
            config_path: Optional path to body profile config
        """
        self.config = get_config()
        self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[Path] = None):
        """Load color harmony configuration."""
        if config_path and config_path.exists():
            with open(config_path) as f:
                full_config = json.load(f)
        else:
            full_config = self.config.get_data("body_profile_config", default={})
        
        color_config = full_config.get("color_harmony", {})
        
        self.undertone_colors = color_config.get("undertone_colors", {
            "COOL": {
                "best": ["navy", "blue", "purple", "emerald", "pink", "silver", "gray"],
                "good": ["white", "black", "burgundy", "plum", "teal"],
                "avoid": ["orange", "yellow-orange", "warm_brown", "gold", "peach"]
            },
            "WARM": {
                "best": ["orange", "coral", "peach", "gold", "olive", "warm_brown", "cream"],
                "good": ["mustard", "rust", "terracotta", "camel", "ivory"],
                "avoid": ["blue-pink", "silver", "stark_white", "black", "cool_gray"]
            },
            "NEUTRAL": {
                "best": ["jade", "dusty_pink", "soft_white", "medium_gray"],
                "good": ["most_colors", "muted_tones"],
                "avoid": ["very_bright", "very_cool", "very_warm"]
            }
        })
        
        self.contrast_matching = color_config.get("contrast_matching", {
            "LOW": {
                "recommended": ["monochromatic", "tone_on_tone", "soft_colors"],
                "avoid": ["high_contrast", "black_and_white", "very_bright"]
            },
            "MEDIUM": {
                "recommended": ["balanced_contrast", "complementary", "medium_tones"],
                "avoid": ["extreme_contrast", "all_muted"]
            },
            "HIGH": {
                "recommended": ["bold_contrasts", "black_and_white", "jewel_tones"],
                "avoid": ["all_pastels", "low_contrast", "washed_out"]
            },
            "VERY_HIGH": {
                "recommended": ["dramatic_contrast", "bold_colors", "stark_combinations"],
                "avoid": ["muted_tones", "subtle_combinations"]
            }
        })
        
        self.skin_tone_colors = color_config.get("skin_tone_colors", {
            "FAIR": ["soft_pink", "light_blue", "lavender", "sage_green"],
            "LIGHT": ["rose", "periwinkle", "mint", "soft_coral"],
            "MEDIUM": ["terracotta", "teal", "burgundy", "olive"],
            "OLIVE": ["coral", "turquoise", "purple", "rust"],
            "TAN": ["orange", "gold", "emerald", "warm_red"],
            "DARK": ["bright_orange", "fuchsia", "emerald", "cobalt"],
            "DEEP": ["white", "bright_yellow", "electric_blue", "hot_pink"]
        })
        
        # Define color families for matching
        self.color_families = {
            "warm": ["orange", "coral", "peach", "gold", "rust", "terracotta", 
                     "cream", "ivory", "mustard", "warm_brown", "camel", "tan",
                     "warm_red", "salmon", "amber", "cognac"],
            "cool": ["blue", "navy", "purple", "pink", "silver", "gray", 
                     "burgundy", "plum", "teal", "emerald", "lavender", 
                     "periwinkle", "cool_gray", "slate", "mauve"],
            "neutral": ["white", "black", "gray", "beige", "taupe", "charcoal",
                        "off_white", "heather", "stone"]
        }
        
        logger.info("ColorHarmonyAdvisor configuration loaded")
    
    def create_color_profile(
        self,
        skin_tone: str,
        undertone: str,
        hair_color: str,
        contrast_level: str
    ) -> ColorProfile:
        """
        Create a color profile from user attributes.
        
        Args:
            skin_tone: User's skin tone
            undertone: User's undertone (warm/cool/neutral)
            hair_color: User's hair color
            contrast_level: User's contrast level
            
        Returns:
            ColorProfile instance
        """
        return ColorProfile(
            skin_tone=skin_tone.upper(),
            undertone=undertone.upper(),
            hair_color=hair_color.upper(),
            contrast_level=contrast_level.upper()
        )
    
    def get_color_recommendations(
        self,
        color_profile: ColorProfile
    ) -> ColorRecommendation:
        """
        Get color recommendations based on user's color profile.
        
        Args:
            color_profile: User's color profile
            
        Returns:
            ColorRecommendation with best, good, and avoid colors
        """
        # Get undertone-based colors
        undertone_rules = self.undertone_colors.get(
            color_profile.undertone, 
            self.undertone_colors.get("NEUTRAL", {})
        )
        
        best = list(undertone_rules.get("best", []))
        good = list(undertone_rules.get("good", []))
        avoid = list(undertone_rules.get("avoid", []))
        
        # Add skin tone specific colors
        skin_colors = self.skin_tone_colors.get(color_profile.skin_tone, [])
        best.extend([c for c in skin_colors if c not in best])
        
        # Adjust based on contrast level
        contrast_rules = self.contrast_matching.get(
            color_profile.contrast_level,
            self.contrast_matching.get("MEDIUM", {})
        )
        
        # Generate tips based on contrast
        tips = []
        if color_profile.contrast_level in ["LOW", "VERY_LOW"]:
            tips.append("Stick to monochromatic or tone-on-tone looks")
            tips.append("Avoid stark contrasts like pure black with pure white")
        elif color_profile.contrast_level in ["HIGH", "VERY_HIGH"]:
            tips.append("Bold contrasts work well for you")
            tips.append("Try black and white combinations")
            tips.append("Jewel tones will complement your natural coloring")
        else:
            tips.append("Balanced color combinations work well")
            tips.append("You can wear both soft and moderate contrasts")
        
        # Neutral colors that work for most
        neutrals = ["white", "black", "navy", "gray", "beige"]
        if color_profile.undertone == "WARM":
            neutrals = ["cream", "camel", "tan", "warm_gray", "chocolate"]
        elif color_profile.undertone == "COOL":
            neutrals = ["white", "black", "navy", "charcoal", "cool_gray"]
        
        # Accent colors based on season
        season = color_profile.season
        accent_colors = self._get_season_accents(season)
        
        return ColorRecommendation(
            best_colors=list(set(best)),
            good_colors=list(set(good)),
            colors_to_avoid=list(set(avoid)),
            neutral_colors=neutrals,
            accent_colors=accent_colors,
            tips=tips
        )
    
    def score_outfit_colors(
        self,
        garments: List[Garment],
        color_profile: ColorProfile
    ) -> ColorHarmonyScore:
        """
        Score how well outfit colors harmonize with user's coloring.
        
        Args:
            garments: List of garments in outfit
            color_profile: User's color profile
            
        Returns:
            ColorHarmonyScore with score and details
        """
        recommendations = self.get_color_recommendations(color_profile)
        
        matching = []
        clashing = []
        notes = []
        
        # Extract colors from garments
        outfit_colors = self._extract_outfit_colors(garments)
        
        score_adjustment = 0.0
        
        for color in outfit_colors:
            color_lower = color.lower()
            
            # Check if it's in best colors
            if self._color_matches(color_lower, recommendations.best_colors):
                matching.append(color)
                score_adjustment += 0.15
            elif self._color_matches(color_lower, recommendations.good_colors):
                matching.append(color)
                score_adjustment += 0.08
            elif self._color_matches(color_lower, recommendations.neutral_colors):
                # Neutrals are okay
                score_adjustment += 0.03
            elif self._color_matches(color_lower, recommendations.colors_to_avoid):
                clashing.append(color)
                score_adjustment -= 0.2
        
        # Check contrast within outfit
        outfit_contrast = self._analyze_outfit_contrast(outfit_colors)
        contrast_rules = self.contrast_matching.get(color_profile.contrast_level, {})
        
        if outfit_contrast in contrast_rules.get("recommended", []):
            notes.append(f"Good outfit contrast matches your personal contrast")
            score_adjustment += 0.1
        elif outfit_contrast in contrast_rules.get("avoid", []):
            notes.append(f"Outfit contrast doesn't match your personal contrast level")
            score_adjustment -= 0.1
        
        # Calculate final score
        base_score = 0.7
        final_score = max(0.0, min(1.0, base_score + score_adjustment))
        
        # Classify harmony level
        harmony_level = self._classify_harmony(final_score)
        
        return ColorHarmonyScore(
            score=final_score,
            harmony_level=harmony_level,
            matching_colors=matching,
            clashing_colors=clashing,
            notes=notes
        )
    
    def _get_season_accents(self, season: str) -> List[str]:
        """Get accent colors for color season."""
        season_accents = {
            "SPRING": ["coral", "peach", "warm_pink", "golden_yellow", "turquoise"],
            "SUMMER": ["dusty_pink", "lavender", "soft_blue", "sage", "mauve"],
            "AUTUMN": ["rust", "mustard", "olive", "terracotta", "burgundy"],
            "WINTER": ["fuchsia", "royal_blue", "emerald", "true_red", "icy_pink"]
        }
        return season_accents.get(season, season_accents["SUMMER"])
    
    def _extract_outfit_colors(self, garments: List[Garment]) -> List[str]:
        """Extract colors from garment list."""
        colors = []
        
        for garment in garments:
            # Check color attribute
            if hasattr(garment, 'color') and garment.color:
                colors.append(str(garment.color))
            
            # Check attributes dict
            if hasattr(garment, 'attributes') and garment.attributes:
                attrs = garment.attributes
                if isinstance(attrs, dict):
                    color = attrs.get('color') or attrs.get('Color')
                    if color:
                        colors.append(str(color))
                    
                    # Also check for primary_color, secondary_color
                    if attrs.get('primary_color'):
                        colors.append(str(attrs['primary_color']))
                    if attrs.get('secondary_color'):
                        colors.append(str(attrs['secondary_color']))
        
        return colors
    
    def _color_matches(self, color: str, color_list: List[str]) -> bool:
        """Check if color matches any in the list (flexible matching)."""
        color_normalized = color.lower().replace("_", " ").replace("-", " ")
        
        for target in color_list:
            target_normalized = target.lower().replace("_", " ").replace("-", " ")
            
            # Direct match
            if color_normalized == target_normalized:
                return True
            
            # Partial match (e.g., "light blue" matches "blue")
            if target_normalized in color_normalized or color_normalized in target_normalized:
                return True
        
        return False
    
    def _analyze_outfit_contrast(self, colors: List[str]) -> str:
        """Analyze contrast level within outfit colors."""
        if not colors:
            return "medium_contrast"
        
        # Simple heuristic based on color names
        has_light = any(c for c in colors if 'white' in c.lower() or 
                        'light' in c.lower() or 'cream' in c.lower())
        has_dark = any(c for c in colors if 'black' in c.lower() or 
                       'dark' in c.lower() or 'navy' in c.lower())
        
        if has_light and has_dark:
            return "high_contrast"
        elif has_light or has_dark:
            return "medium_contrast"
        else:
            return "low_contrast"
    
    def _classify_harmony(self, score: float) -> str:
        """Classify harmony level based on score."""
        if score >= 0.85:
            return "excellent"
        elif score >= 0.70:
            return "good"
        elif score >= 0.50:
            return "fair"
        else:
            return "poor"
    
    def get_color_tips(self, color_profile: ColorProfile) -> List[str]:
        """
        Get color tips for a color profile.
        
        Args:
            color_profile: User's color profile
            
        Returns:
            List of color styling tips
        """
        tips = []
        
        # Undertone tips
        if color_profile.undertone == "WARM":
            tips.append("Gold jewelry complements your warm undertone")
            tips.append("Earth tones and warm colors will enhance your complexion")
        elif color_profile.undertone == "COOL":
            tips.append("Silver jewelry complements your cool undertone")
            tips.append("Jewel tones and cool colors will enhance your complexion")
        else:
            tips.append("Both gold and silver jewelry work for you")
            tips.append("You can wear both warm and cool colors")
        
        # Contrast tips
        if color_profile.contrast_level in ["HIGH", "VERY_HIGH"]:
            tips.append("You can handle bold color combinations")
            tips.append("Black and white combinations will look striking on you")
        elif color_profile.contrast_level in ["LOW", "VERY_LOW"]:
            tips.append("Stick to colors in the same tonal range")
            tips.append("Avoid stark contrasts in your outfits")
        
        # Season tips
        season = color_profile.season
        tips.append(f"Your color season is {season} - explore {season.lower()} palettes")
        
        return tips
