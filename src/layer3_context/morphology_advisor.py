"""
Morphology Advisor
Provides body-type aware style recommendations.

Enhanced with body measurement integration for more precise
recommendations based on actual measurements.
"""
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass, field

from config import get_config
from src.core.models import Garment, GarmentCategory
from src.core import get_logger

if TYPE_CHECKING:
    from src.layer3_context.user_profile.models import BodyMetrics

logger = get_logger(__name__)


@dataclass
class EnhancedMorphologyScore:
    """Enhanced scoring with body measurements."""
    base_score: float
    measurement_adjustments: float
    final_score: float
    body_type: str
    notes: List[str] = field(default_factory=list)
    measurement_based_tips: List[str] = field(default_factory=list)


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

    # ========== ENHANCED METHODS WITH BODY MEASUREMENTS ==========
    
    def score_with_measurements(
        self,
        garments: List[Garment],
        body_type: str,
        body_metrics: Optional["BodyMetrics"] = None
    ) -> EnhancedMorphologyScore:
        """
        Score outfit with body measurements for enhanced precision.
        
        Args:
            garments: List of garments
            body_type: Body type string
            body_metrics: Optional body metrics with measurements
            
        Returns:
            EnhancedMorphologyScore with detailed analysis
        """
        # Get base body type score
        base_score = self.score_for_body_type(garments, body_type)
        
        measurement_adjustments = 0.0
        notes = []
        measurement_tips = []
        
        if body_metrics:
            # Adjust based on torso proportion
            torso_prop = getattr(body_metrics, 'torso_proportion', None)
            if torso_prop:
                adj, tip = self._adjust_for_torso(garments, torso_prop)
                measurement_adjustments += adj
                if tip:
                    measurement_tips.append(tip)
            
            # Adjust based on leg proportion
            leg_prop = getattr(body_metrics, 'leg_proportion', None)
            if leg_prop:
                adj, tip = self._adjust_for_legs(garments, leg_prop)
                measurement_adjustments += adj
                if tip:
                    measurement_tips.append(tip)
            
            # Adjust based on frame size
            frame_size = getattr(body_metrics, 'frame_size', None)
            if frame_size:
                adj, tip = self._adjust_for_frame(garments, frame_size)
                measurement_adjustments += adj
                if tip:
                    measurement_tips.append(tip)
            
            # Adjust based on shoulder-hip ratio
            shoulder_ratio = getattr(body_metrics, 'shoulder_hip_ratio', None)
            if shoulder_ratio:
                adj, note = self._adjust_for_shoulder_hip(garments, shoulder_ratio, body_type)
                measurement_adjustments += adj
                if note:
                    notes.append(note)
        
        final_score = max(0.0, min(1.0, base_score + measurement_adjustments))
        
        return EnhancedMorphologyScore(
            base_score=base_score,
            measurement_adjustments=measurement_adjustments,
            final_score=final_score,
            body_type=body_type,
            notes=notes,
            measurement_based_tips=measurement_tips
        )
    
    def get_enhanced_recommendations(
        self,
        body_type: str,
        body_metrics: Optional["BodyMetrics"] = None
    ) -> Dict[str, Any]:
        """
        Get recommendations enhanced with body measurements.
        
        Args:
            body_type: Body type string
            body_metrics: Optional body metrics
            
        Returns:
            Enhanced recommendations dictionary
        """
        base_recs = self.get_recommendations_for_body_type(body_type)
        
        if body_metrics:
            # Add measurement-specific tips
            measurement_tips = []
            
            # Torso-based tips
            torso_prop = getattr(body_metrics, 'torso_proportion', None)
            if torso_prop == 'short':
                measurement_tips.append("High-waisted bottoms will elongate your torso")
                measurement_tips.append("V-necks create vertical lines that lengthen")
            elif torso_prop == 'long':
                measurement_tips.append("Mid-rise bottoms balance your proportions")
                measurement_tips.append("Layered tops add interest to your torso")
            
            # Leg-based tips
            leg_prop = getattr(body_metrics, 'leg_proportion', None)
            if leg_prop == 'short':
                measurement_tips.append("High-waisted pants create an illusion of longer legs")
                measurement_tips.append("Pointed toe shoes elongate your silhouette")
            elif leg_prop == 'long':
                measurement_tips.append("You can wear any pant length beautifully")
                measurement_tips.append("Cropped pants showcase your leg length")
            
            # Frame size tips
            frame_size = getattr(body_metrics, 'frame_size', None)
            if frame_size == 'small':
                measurement_tips.append("Delicate accessories complement your frame")
                measurement_tips.append("Avoid oversized, bulky pieces")
            elif frame_size == 'large':
                measurement_tips.append("Substantial accessories suit your frame")
                measurement_tips.append("Bold patterns work well for you")
            
            # Proportion tips from BodyMetrics
            if hasattr(body_metrics, 'proportion_tips') and body_metrics.proportion_tips:
                measurement_tips.extend(body_metrics.proportion_tips)
            
            base_recs["measurement_based_tips"] = measurement_tips
            
            # Add size recommendations
            top_size = getattr(body_metrics, 'estimated_top_size', None)
            bottom_size = getattr(body_metrics, 'estimated_bottom_size', None)
            
            if top_size or bottom_size:
                base_recs["size_recommendations"] = {
                    "estimated_top_size": top_size,
                    "estimated_bottom_size": bottom_size
                }
        
        return base_recs
    
    def _adjust_for_torso(
        self,
        garments: List[Garment],
        torso_proportion: str
    ) -> tuple[float, Optional[str]]:
        """Adjust score based on torso proportion."""
        if torso_proportion == 'short':
            # Check for high-waisted items (bonus) or low-rise (penalty)
            for garment in garments:
                waist = self._get_garment_attribute(garment, 'waist', '')
                if 'high' in waist.lower():
                    return 0.1, None
                elif 'low' in waist.lower():
                    return -0.1, "Low-rise may not flatter short torso"
        elif torso_proportion == 'long':
            for garment in garments:
                waist = self._get_garment_attribute(garment, 'waist', '')
                if 'low' in waist.lower() or 'mid' in waist.lower():
                    return 0.05, None
        
        return 0.0, None
    
    def _adjust_for_legs(
        self,
        garments: List[Garment],
        leg_proportion: str
    ) -> tuple[float, Optional[str]]:
        """Adjust score based on leg proportion."""
        if leg_proportion == 'short':
            for garment in garments:
                # Check for vertical lines or elongating features
                pattern = self._get_garment_attribute(garment, 'pattern', '')
                if 'vertical' in pattern.lower():
                    return 0.1, None
                
                # Check for cropped pants (penalty)
                category = str(getattr(garment, 'category', '')).lower()
                if 'crop' in category:
                    return -0.1, "Cropped pants may shorten leg appearance"
        
        return 0.0, None
    
    def _adjust_for_frame(
        self,
        garments: List[Garment],
        frame_size: str
    ) -> tuple[float, Optional[str]]:
        """Adjust score based on frame size."""
        for garment in garments:
            weight = self._get_garment_attribute(garment, 'weight', '')
            
            if frame_size == 'small':
                if 'heavy' in weight.lower() or 'bulky' in weight.lower():
                    return -0.1, "Heavy fabrics may overwhelm small frame"
            elif frame_size == 'large':
                if 'delicate' in weight.lower() or 'sheer' in weight.lower():
                    return -0.05, "Very delicate fabrics may not suit large frame"
        
        return 0.0, None
    
    def _adjust_for_shoulder_hip(
        self,
        garments: List[Garment],
        shoulder_hip_ratio: float,
        body_type: str
    ) -> tuple[float, Optional[str]]:
        """Adjust score based on shoulder to hip ratio."""
        # Significant difference between shoulders and hips
        if shoulder_hip_ratio > 1.1:  # Broader shoulders
            for garment in garments:
                # Check for shoulder-emphasizing features
                neckline = self._get_garment_attribute(garment, 'neckline', '')
                if 'boat' in neckline.lower() or 'off_shoulder' in neckline.lower():
                    return -0.1, "This neckline may over-emphasize shoulders"
        elif shoulder_hip_ratio < 0.9:  # Broader hips
            for garment in garments:
                # Check for hip-emphasizing features
                silhouette = self._get_garment_attribute(garment, 'silhouette', '')
                if 'tight' in silhouette.lower() or 'bodycon' in silhouette.lower():
                    return -0.05, None
        
        return 0.0, None
    
    def _get_garment_attribute(
        self,
        garment: Garment,
        attr_name: str,
        default: str = ''
    ) -> str:
        """Safely get garment attribute."""
        if hasattr(garment, 'attributes') and garment.attributes:
            attrs = garment.attributes
            if isinstance(attrs, dict):
                return str(attrs.get(attr_name, default))
            elif hasattr(attrs, attr_name):
                return str(getattr(attrs, attr_name, default))
        return default
