"""
Morphology Advisor
Provides body-type aware style recommendations.
"""
from typing import List, Dict, Optional
from enum import Enum

from config import get_config
from src.core.models import Garment, GarmentCategory
from src.core import get_logger

logger = get_logger(__name__)


class BodyType(str, Enum):
    """Common body type classifications."""
    RECTANGLE = "rectangle"
    HOURGLASS = "hourglass"
    PEAR = "pear"
    APPLE = "apple"
    INVERTED_TRIANGLE = "inverted_triangle"
    ATHLETIC = "athletic"


class MorphologyAdvisor:
    """
    Provides body-type aware fashion advice.
    
    Different body types are flattered by different:
    - Silhouettes
    - Necklines
    - Proportions
    - Patterns
    
    This advisor helps select items that work best
    for each body type.
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Load morphology data from configuration
        morph_data = self.config.get_data("morphology_data", default={})
        
        # Recommended attributes per body type from config
        self.recommendations = self._build_recommendations(
            morph_data.get("body_type_recommendations", {})
        )
        
        # General recommendations fallback
        self._general_recommendations = morph_data.get("general_recommendations", {
            "silhouettes": ["balanced", "well-fitted"],
            "patterns": ["personal_preference"],
            "avoid": [],
            "tips": [
                "Focus on fit and comfort",
                "Wear what makes you feel confident"
            ]
        })
        
        # Priority areas per body type
        self._priority_areas = morph_data.get("priority_areas", {})
    
    def _build_recommendations(self, config_recs: Dict) -> Dict[BodyType, Dict]:
        """Build recommendations dict from config."""
        result = {}
        for body_type in BodyType:
            bt_key = body_type.value
            if bt_key in config_recs:
                result[body_type] = config_recs[bt_key]
            else:
                # Use defaults
                result[body_type] = self._get_default_for_body_type(body_type)
        return result
    
    def _get_default_for_body_type(self, body_type: BodyType) -> Dict:
        """Get default recommendations for a body type."""
        defaults = {
            BodyType.RECTANGLE: {
                "silhouettes": ["a-line", "fitted", "peplum", "structured"],
                "patterns": ["horizontal_stripes", "color_blocking"],
                "avoid": ["shapeless", "boxy"],
                "tips": ["Create curves with structured pieces", 
                        "Define waist with belts"]
            },
            BodyType.HOURGLASS: {
                "silhouettes": ["fitted", "wrap", "bodycon", "tailored"],
                "patterns": ["any"],
                "avoid": ["shapeless", "oversized", "boxy"],
                "tips": ["Emphasize your natural waist",
                        "Choose items that follow your curves"]
            },
            BodyType.PEAR: {
                "silhouettes": ["a-line", "fit_and_flare", "empire_waist"],
                "patterns": ["on_top", "solid_bottom"],
                "avoid": ["tight_bottoms", "hip_emphasis"],
                "tips": ["Draw attention to upper body",
                        "Choose darker colors on bottom"]
            },
            BodyType.APPLE: {
                "silhouettes": ["empire_waist", "v_neck", "a-line"],
                "patterns": ["vertical_stripes", "v_patterns"],
                "avoid": ["tight_waist", "clingy_fabrics"],
                "tips": ["Elongate torso with V-necks",
                        "Show off legs with shorter hemlines"]
            },
            BodyType.INVERTED_TRIANGLE: {
                "silhouettes": ["a-line", "wide_leg", "flared"],
                "patterns": ["volume_on_bottom"],
                "avoid": ["shoulder_pads", "boat_necks"],
                "tips": ["Balance broad shoulders with volume below",
                        "Choose V-necks to soften shoulders"]
            },
            BodyType.ATHLETIC: {
                "silhouettes": ["feminine", "ruffles", "curved"],
                "patterns": ["florals", "soft_patterns"],
                "avoid": ["too_sporty", "severe_cuts"],
                "tips": ["Add softness with feminine details",
                        "Create curves with strategic volume"]
            }
        }
        return defaults.get(body_type, self._general_recommendations)
    
    def score_for_body_type(
        self,
        garments: List[Garment],
        body_type: str
    ) -> float:
        """
        Score outfit suitability for body type.
        
        Args:
            garments: List of garments
            body_type: Body type string
            
        Returns:
            Score between 0 and 1
        """
        try:
            bt = BodyType(body_type.lower())
        except ValueError:
            return 0.7  # Unknown body type, neutral score
        
        if bt not in self.recommendations:
            return 0.7
        
        recs = self.recommendations[bt]
        scores = []
        
        for garment in garments:
            score = self._score_garment_for_body_type(garment, recs)
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.7
    
    def get_recommendations_for_body_type(
        self,
        body_type: str
    ) -> Dict:
        """
        Get detailed recommendations for body type.
        
        Args:
            body_type: Body type string
            
        Returns:
            Dictionary with recommendations
        """
        try:
            bt = BodyType(body_type.lower())
        except ValueError:
            return self._get_general_recommendations()
        
        if bt not in self.recommendations:
            return self._get_general_recommendations()
        
        recs = self.recommendations[bt]
        
        return {
            "body_type": bt.value,
            "flattering_silhouettes": recs["silhouettes"],
            "good_patterns": recs["patterns"],
            "styles_to_avoid": recs["avoid"],
            "styling_tips": recs["tips"],
            "priority_areas": self._get_priority_areas(bt)
        }
    
    def filter_by_body_type(
        self,
        garments: List[Garment],
        body_type: str
    ) -> List[Garment]:
        """
        Filter garments to those suitable for body type.
        
        Args:
            garments: Available garments
            body_type: Body type string
            
        Returns:
            Filtered list of suitable garments
        """
        try:
            bt = BodyType(body_type.lower())
        except ValueError:
            return garments  # Return all if unknown
        
        if bt not in self.recommendations:
            return garments
        
        recs = self.recommendations[bt]
        suitable = []
        
        for garment in garments:
            # Check if garment has avoided characteristics
            if self._should_avoid(garment, recs):
                continue
            
            # Include garments with good characteristics or neutral
            suitable.append(garment)
        
        return suitable
    
    def _score_garment_for_body_type(
        self,
        garment: Garment,
        recs: Dict
    ) -> float:
        """Score individual garment for body type."""
        score = 0.6  # Base neutral score
        
        silhouette = garment.attributes.silhouette
        if silhouette:
            silhouette = silhouette.lower()
            
            # Bonus for recommended silhouettes
            if silhouette in recs["silhouettes"]:
                score += 0.3
            
            # Penalty for avoided silhouettes
            if silhouette in recs.get("avoid", []):
                score -= 0.3
        
        # Check style tags
        style_tags = [s.lower() for s in garment.attributes.style_tags]
        
        for avoid in recs.get("avoid", []):
            if avoid in style_tags:
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _should_avoid(self, garment: Garment, recs: Dict) -> bool:
        """Check if garment should be avoided."""
        avoid_list = recs.get("avoid", [])
        
        silhouette = garment.attributes.silhouette
        if silhouette and silhouette.lower() in avoid_list:
            return True
        
        style_tags = [s.lower() for s in garment.attributes.style_tags]
        for avoid in avoid_list:
            if avoid in style_tags:
                return True
        
        return False
    
    def _get_priority_areas(self, body_type: BodyType) -> List[str]:
        """Get areas to emphasize or minimize."""
        priorities = {
            BodyType.RECTANGLE: {
                "emphasize": ["waist", "curves"],
                "minimize": []
            },
            BodyType.HOURGLASS: {
                "emphasize": ["waist", "curves"],
                "minimize": []
            },
            BodyType.PEAR: {
                "emphasize": ["shoulders", "bust"],
                "minimize": ["hips"]
            },
            BodyType.APPLE: {
                "emphasize": ["legs", "bust"],
                "minimize": ["waist"]
            },
            BodyType.INVERTED_TRIANGLE: {
                "emphasize": ["hips", "legs"],
                "minimize": ["shoulders"]
            },
            BodyType.ATHLETIC: {
                "emphasize": ["curves", "waist"],
                "minimize": []
            }
        }
        return priorities.get(body_type, {"emphasize": [], "minimize": []})
    
    def _get_general_recommendations(self) -> Dict:
        """Get general recommendations for unknown body type."""
        return {
            "body_type": "general",
            "flattering_silhouettes": ["fitted", "a-line", "straight"],
            "good_patterns": ["any balanced pattern"],
            "styles_to_avoid": [],
            "styling_tips": [
                "Focus on fit - clothes that fit well always look better",
                "Define your waist for a flattering silhouette",
                "Choose quality fabrics that drape well"
            ],
            "priority_areas": {"emphasize": [], "minimize": []}
        }
