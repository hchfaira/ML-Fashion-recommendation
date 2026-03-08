"""
Volume Balance Scorer
Ensures outfits balance volume (oversized/wide) with structure (fitted/slim).

The goal is to create visual harmony without overwhelming the person's frame.

Scoring:
- Wide + Slim (+2 pts): Wide-leg trousers with fitted top ✅
- Slim + Wide (+2 pts): Oversized hoodie with skinny jeans ✅
- Wide + Wide (-1 pt): Can look "sloppy" unless intentional ⚠️
- Slim + Slim (0 pts): Safe but sometimes lacks visual interest

Body Shape Considerations:
- Inverted Triangle: Volume on bottom balances broad shoulders
- Pear Shape: Volume/details on top balances hips
- Rectangle: Volume creates curves
- Hourglass: Fitted maintains natural proportions
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.core.models import Garment, GarmentCategory
from src.core import get_logger

logger = get_logger(__name__)


class VolumeLevel(str, Enum):
    """Volume levels for garments."""
    SLIM = "slim"           # Fitted, skinny, bodycon
    REGULAR = "regular"     # Standard fit
    RELAXED = "relaxed"     # Slightly loose
    WIDE = "wide"           # Wide-leg, oversized, voluminous


class BodyShape(str, Enum):
    """Body shape categories."""
    INVERTED_TRIANGLE = "inverted_triangle"  # Broad shoulders, narrow hips
    PEAR = "pear"                            # Narrow shoulders, wider hips
    RECTANGLE = "rectangle"                   # Balanced, minimal waist definition
    HOURGLASS = "hourglass"                  # Balanced with defined waist
    APPLE = "apple"                          # Fuller midsection
    ATHLETIC = "athletic"                    # Muscular, broad shoulders


@dataclass
class VolumeBalanceResult:
    """Result of volume balance analysis."""
    score: float  # 0-1 normalized
    points: int  # -1 to +2
    top_volume: VolumeLevel
    bottom_volume: VolumeLevel
    balance_type: str  # Description of balance
    recommendation: str
    body_shape_advice: Optional[str] = None


class VolumeBalanceScorer:
    """
    Scores volume balance in outfits.
    
    Follows the principle of contrasting volumes for visual interest
    while maintaining overall harmony.
    """
    
    def __init__(self):
        # Fit to volume mapping
        self.fit_to_volume = {
            "slim": VolumeLevel.SLIM,
            "skinny": VolumeLevel.SLIM,
            "fitted": VolumeLevel.SLIM,
            "bodycon": VolumeLevel.SLIM,
            "tight": VolumeLevel.SLIM,
            "regular": VolumeLevel.REGULAR,
            "standard": VolumeLevel.REGULAR,
            "straight": VolumeLevel.REGULAR,
            "relaxed": VolumeLevel.RELAXED,
            "loose": VolumeLevel.RELAXED,
            "comfortable": VolumeLevel.RELAXED,
            "oversized": VolumeLevel.WIDE,
            "wide": VolumeLevel.WIDE,
            "wide-leg": VolumeLevel.WIDE,
            "boxy": VolumeLevel.WIDE,
            "voluminous": VolumeLevel.WIDE,
            "baggy": VolumeLevel.WIDE,
            "flowy": VolumeLevel.WIDE,
            "palazzo": VolumeLevel.WIDE,
        }
        
        # Volume compatibility scoring
        # (top_volume, bottom_volume) -> (points, balance_type)
        self.volume_compatibility = {
            # Contrasting volumes = Best
            (VolumeLevel.WIDE, VolumeLevel.SLIM): (2, "balanced_contrast"),
            (VolumeLevel.SLIM, VolumeLevel.WIDE): (2, "balanced_contrast"),
            (VolumeLevel.RELAXED, VolumeLevel.SLIM): (2, "balanced_contrast"),
            (VolumeLevel.SLIM, VolumeLevel.RELAXED): (2, "balanced_contrast"),
            
            # Some contrast = Good
            (VolumeLevel.WIDE, VolumeLevel.REGULAR): (1, "slight_contrast"),
            (VolumeLevel.REGULAR, VolumeLevel.WIDE): (1, "slight_contrast"),
            (VolumeLevel.RELAXED, VolumeLevel.REGULAR): (1, "slight_contrast"),
            (VolumeLevel.REGULAR, VolumeLevel.RELAXED): (1, "slight_contrast"),
            
            # Similar volumes = Neutral to Risky
            (VolumeLevel.SLIM, VolumeLevel.SLIM): (0, "all_fitted"),
            (VolumeLevel.REGULAR, VolumeLevel.REGULAR): (0, "all_regular"),
            (VolumeLevel.RELAXED, VolumeLevel.RELAXED): (-1, "all_relaxed"),
            (VolumeLevel.WIDE, VolumeLevel.WIDE): (-1, "all_voluminous"),
        }
        
        # Body shape optimal volumes
        self.body_shape_preferences = {
            BodyShape.INVERTED_TRIANGLE: {
                "top_preference": VolumeLevel.SLIM,
                "bottom_preference": VolumeLevel.WIDE,
                "advice": "Add volume to lower body to balance broad shoulders"
            },
            BodyShape.PEAR: {
                "top_preference": VolumeLevel.RELAXED,
                "bottom_preference": VolumeLevel.SLIM,
                "advice": "Add volume/details on top to balance hips"
            },
            BodyShape.RECTANGLE: {
                "top_preference": VolumeLevel.RELAXED,
                "bottom_preference": VolumeLevel.RELAXED,
                "advice": "Volume or layers can help create curves and waist definition"
            },
            BodyShape.HOURGLASS: {
                "top_preference": VolumeLevel.REGULAR,
                "bottom_preference": VolumeLevel.REGULAR,
                "advice": "Fitted or regular fits maintain your natural proportions"
            },
            BodyShape.APPLE: {
                "top_preference": VolumeLevel.RELAXED,
                "bottom_preference": VolumeLevel.SLIM,
                "advice": "A-line or relaxed tops with slim bottoms elongate the silhouette"
            },
            BodyShape.ATHLETIC: {
                "top_preference": VolumeLevel.REGULAR,
                "bottom_preference": VolumeLevel.WIDE,
                "advice": "Wide-leg pants can soften a muscular frame"
            }
        }
    
    def analyze_outfit(
        self,
        items: List[Garment],
        body_shape: Optional[BodyShape] = None
    ) -> VolumeBalanceResult:
        """
        Analyze volume balance in an outfit.
        
        Args:
            items: List of garments
            body_shape: Optional body shape for personalized advice
            
        Returns:
            VolumeBalanceResult with scoring
        """
        # Separate tops and bottoms
        tops = [i for i in items if i.attributes.category in 
                {GarmentCategory.TOP, GarmentCategory.OUTERWEAR}]
        bottoms = [i for i in items if i.attributes.category == GarmentCategory.BOTTOM]
        
        if not tops or not bottoms:
            return VolumeBalanceResult(
                score=0.7,
                points=0,
                top_volume=VolumeLevel.REGULAR,
                bottom_volume=VolumeLevel.REGULAR,
                balance_type="incomplete",
                recommendation="Need both top and bottom to analyze volume balance"
            )
        
        # Get dominant volume for each
        top_volume = self._get_dominant_volume(tops)
        bottom_volume = self._get_dominant_volume(bottoms)
        
        # Get compatibility score
        combo = (top_volume, bottom_volume)
        points, balance_type = self.volume_compatibility.get(
            combo, (0, "unknown")
        )
        
        # Normalize score (-1 to +2) -> (0 to 1)
        score = (points + 1) / 3
        score = max(0, min(1, score))
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            top_volume, bottom_volume, balance_type, points
        )
        
        # Body shape specific advice
        body_advice = None
        if body_shape:
            body_advice = self._get_body_shape_advice(
                body_shape, top_volume, bottom_volume
            )
            # Adjust score based on body shape fit
            shape_bonus = self._calculate_body_shape_bonus(
                body_shape, top_volume, bottom_volume
            )
            score = min(1.0, score + shape_bonus)
        
        return VolumeBalanceResult(
            score=score,
            points=points,
            top_volume=top_volume,
            bottom_volume=bottom_volume,
            balance_type=balance_type,
            recommendation=recommendation,
            body_shape_advice=body_advice
        )
    
    def _get_dominant_volume(self, items: List[Garment]) -> VolumeLevel:
        """Get the dominant volume from a list of items."""
        volumes = []
        
        for item in items:
            fit = item.attributes.fit
            if fit:
                volume = self.fit_to_volume.get(fit.lower(), VolumeLevel.REGULAR)
            else:
                # Infer from silhouette
                silhouette = item.attributes.silhouette
                if silhouette:
                    volume = self.fit_to_volume.get(
                        silhouette.lower(), VolumeLevel.REGULAR
                    )
                else:
                    volume = VolumeLevel.REGULAR
            volumes.append(volume)
        
        # Return the most voluminous (visually dominant)
        volume_order = [VolumeLevel.WIDE, VolumeLevel.RELAXED, 
                       VolumeLevel.REGULAR, VolumeLevel.SLIM]
        
        for vol in volume_order:
            if vol in volumes:
                return vol
        
        return VolumeLevel.REGULAR
    
    def _generate_recommendation(
        self,
        top_vol: VolumeLevel,
        bottom_vol: VolumeLevel,
        balance_type: str,
        points: int
    ) -> str:
        """Generate styling recommendation."""
        if balance_type == "balanced_contrast":
            return (f"✅ Excellent volume balance! The {top_vol.value} top with "
                   f"{bottom_vol.value} bottom creates visual interest while staying balanced.")
        
        elif balance_type == "slight_contrast":
            return (f"👍 Good volume combination. The contrast between "
                   f"{top_vol.value} and {bottom_vol.value} works well.")
        
        elif balance_type == "all_fitted":
            return ("Sleek all-fitted look. Consider adding a statement accessory "
                   "or interesting textures for more visual interest.")
        
        elif balance_type == "all_regular":
            return "Classic balanced silhouette. Works well for most occasions."
        
        elif balance_type == "all_relaxed":
            return ("⚠️ Both pieces are relaxed. Define your waist with a belt "
                   "or opt for one more structured piece to avoid a shapeless look.")
        
        elif balance_type == "all_voluminous":
            return ("⚠️ Double volume can look overwhelming. Consider swapping "
                   "one piece for a more fitted option to create contrast.")
        
        return "Analyze your top and bottom volumes for better balance."
    
    def _get_body_shape_advice(
        self,
        body_shape: BodyShape,
        top_vol: VolumeLevel,
        bottom_vol: VolumeLevel
    ) -> str:
        """Get body shape specific advice."""
        prefs = self.body_shape_preferences.get(body_shape)
        if not prefs:
            return ""
        
        advice_parts = [prefs["advice"]]
        
        # Check if current outfit matches preferences
        top_match = (top_vol == prefs["top_preference"] or 
                    top_vol == VolumeLevel.REGULAR)
        bottom_match = (bottom_vol == prefs["bottom_preference"] or 
                       bottom_vol == VolumeLevel.REGULAR)
        
        if top_match and bottom_match:
            advice_parts.append("✅ This volume distribution suits your body shape well!")
        elif not top_match:
            advice_parts.append(
                f"Consider a {prefs['top_preference'].value} top for your {body_shape.value} shape."
            )
        elif not bottom_match:
            advice_parts.append(
                f"Consider {prefs['bottom_preference'].value} bottoms for your {body_shape.value} shape."
            )
        
        return " ".join(advice_parts)
    
    def _calculate_body_shape_bonus(
        self,
        body_shape: BodyShape,
        top_vol: VolumeLevel,
        bottom_vol: VolumeLevel
    ) -> float:
        """Calculate score bonus for body shape fit."""
        prefs = self.body_shape_preferences.get(body_shape)
        if not prefs:
            return 0
        
        bonus = 0
        if top_vol == prefs["top_preference"]:
            bonus += 0.1
        if bottom_vol == prefs["bottom_preference"]:
            bonus += 0.1
        
        return bonus
    
    def get_recommended_volumes(self, body_shape: BodyShape) -> Dict[str, str]:
        """Get recommended volumes for a body shape."""
        prefs = self.body_shape_preferences.get(body_shape, {})
        return {
            "top": prefs.get("top_preference", VolumeLevel.REGULAR).value,
            "bottom": prefs.get("bottom_preference", VolumeLevel.REGULAR).value,
            "advice": prefs.get("advice", "Balance volume between top and bottom")
        }


def calculate_volume_balance_score(
    items: List[Garment],
    body_shape: Optional[BodyShape] = None
) -> float:
    """Quick function to get volume balance score."""
    scorer = VolumeBalanceScorer()
    result = scorer.analyze_outfit(items, body_shape)
    return result.score
