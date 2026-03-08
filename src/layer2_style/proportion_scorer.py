"""
Proportion Scorer - Rule of Thirds
Implements the fashion Rule of Thirds for body proportions.

The Golden Ratio in fashion is 1/3 to 2/3 division:
- A short top (1/3) with high-waisted trousers (2/3) ✅
- A mini dress (1/3) with long legs visible (2/3) ✅
- An untucked long tunic over baggy pants (50/50) ❌

Dividing the body 50/50 often looks "stumpy."
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from src.core.models import Garment, GarmentCategory, LengthType, WaistRise
from src.core import get_logger

logger = get_logger(__name__)


class ProportionRatio(str, Enum):
    """Proportion ratio categories."""
    ONE_THIRD_TWO_THIRDS = "1/3 - 2/3"  # Ideal golden ratio
    TWO_THIRDS_ONE_THIRD = "2/3 - 1/3"  # Inverted but still good
    BALANCED_TUCKED = "balanced_tucked"  # 50/50 but tucked (acceptable)
    FIFTY_FIFTY = "50/50"  # Can look stumpy
    UNDEFINED = "undefined"


@dataclass
class ProportionResult:
    """Result of proportion analysis."""
    score: float  # 0-1 normalized
    points: int  # -2 to +3
    ratio: ProportionRatio
    top_portion: str  # Description of top portion
    bottom_portion: str  # Description of bottom portion
    recommendation: str


class ProportionScorer:
    """
    Scores outfit proportions based on the Rule of Thirds.
    
    Scoring:
    - Harmonious 1/3-2/3 split: +3 points
    - Balanced tucked: +1 point
    - 50/50 split: -2 points (disproportional)
    """
    
    def __init__(self):
        # Length mappings to visual portion
        self.top_length_portion = {
            LengthType.CROP: "short",       # ~1/3 visual
            LengthType.REGULAR: "medium",   # ~1/2 visual
            LengthType.LONGLINE: "long",    # ~2/3 visual
        }
        
        self.bottom_length_portion = {
            LengthType.MINI: "short",       # Shows 2/3 leg
            LengthType.KNEE_LENGTH: "medium",
            LengthType.MIDI: "long",        # Shows 1/3 leg
            LengthType.MAXI: "full",        # Shows minimal leg
            LengthType.ANKLE_LENGTH: "full",
            LengthType.REGULAR: "medium",   # Standard pants
        }
        
        # Waist rise impact on proportions
        self.waist_rise_impact = {
            WaistRise.HIGH_RISE: "elongates_legs",  # Creates 1/3-2/3
            WaistRise.MID_RISE: "neutral",
            WaistRise.LOW_RISE: "shortens_legs",    # Creates 2/3-1/3
        }
        
        # Optimal combinations for 1/3-2/3 ratio
        self.golden_ratio_combos = [
            # (top_length, waist_rise) -> Best proportions
            (LengthType.CROP, WaistRise.HIGH_RISE),      # Classic 1/3-2/3
            (LengthType.CROP, WaistRise.MID_RISE),       # Still good
            (LengthType.REGULAR, WaistRise.HIGH_RISE),   # Tucked in look
        ]
        
        # Bad combinations (50/50 or worse)
        self.bad_proportion_combos = [
            (LengthType.LONGLINE, WaistRise.LOW_RISE),   # Top heavy
            (LengthType.LONGLINE, None),                  # Untucked long
        ]
    
    def analyze_outfit(self, items: List[Garment]) -> ProportionResult:
        """
        Analyze outfit proportions.
        
        Args:
            items: List of garments
            
        Returns:
            ProportionResult with scoring
        """
        # Separate tops and bottoms
        tops = [i for i in items if i.attributes.category in 
                {GarmentCategory.TOP, GarmentCategory.OUTERWEAR}]
        bottoms = [i for i in items if i.attributes.category == GarmentCategory.BOTTOM]
        dresses = [i for i in items if i.attributes.category == GarmentCategory.DRESS]
        
        # Handle dress case
        if dresses:
            return self._analyze_dress_proportions(dresses[0])
        
        if not tops or not bottoms:
            return ProportionResult(
                score=0.7,
                points=0,
                ratio=ProportionRatio.UNDEFINED,
                top_portion="unknown",
                bottom_portion="unknown",
                recommendation="Need both top and bottom to analyze proportions"
            )
        
        # Get primary items
        top = tops[0]
        bottom = bottoms[0]
        
        return self._analyze_top_bottom_proportions(top, bottom)
    
    def _analyze_top_bottom_proportions(
        self,
        top: Garment,
        bottom: Garment
    ) -> ProportionResult:
        """Analyze proportions between top and bottom."""
        top_length = top.attributes.length_type
        bottom_rise = bottom.attributes.waist_rise
        bottom_length = bottom.attributes.length_type
        
        points = 0
        ratio = ProportionRatio.UNDEFINED
        
        # Determine top portion description
        if top_length == LengthType.CROP:
            top_portion = "short (1/3)"
        elif top_length == LengthType.LONGLINE:
            top_portion = "long (2/3)"
        else:
            top_portion = "medium (1/2)"
        
        # Determine bottom portion based on rise and length
        if bottom_rise == WaistRise.HIGH_RISE:
            bottom_portion = "high-waisted (elongates legs)"
        elif bottom_rise == WaistRise.LOW_RISE:
            bottom_portion = "low-rise (shortens legs)"
        else:
            bottom_portion = "mid-rise (neutral)"
        
        # Score the combination
        # Best: Crop top + High-rise = 1/3-2/3 golden ratio
        if top_length == LengthType.CROP and bottom_rise == WaistRise.HIGH_RISE:
            points = 3
            ratio = ProportionRatio.ONE_THIRD_TWO_THIRDS
            recommendation = "✅ Perfect 1/3-2/3 proportions! The crop top and high-waisted bottom create an elongating effect."
        
        # Good: Crop + Mid-rise
        elif top_length == LengthType.CROP and bottom_rise == WaistRise.MID_RISE:
            points = 2
            ratio = ProportionRatio.ONE_THIRD_TWO_THIRDS
            recommendation = "👍 Great proportions. Consider high-rise bottoms for even better leg elongation."
        
        # Good: Regular tucked into high-rise
        elif top_length == LengthType.REGULAR and bottom_rise == WaistRise.HIGH_RISE:
            points = 1
            ratio = ProportionRatio.BALANCED_TUCKED
            recommendation = "👍 Balanced proportions when tucked in. The high waist helps elongate."
        
        # Bad: Longline untucked with low-rise
        elif top_length == LengthType.LONGLINE and bottom_rise == WaistRise.LOW_RISE:
            points = -2
            ratio = ProportionRatio.FIFTY_FIFTY
            recommendation = "⚠️ This creates unflattering 50/50 proportions. Try tucking or switching to high-rise bottoms."
        
        # Bad: Longline with any (usually untucked)
        elif top_length == LengthType.LONGLINE:
            points = -1
            ratio = ProportionRatio.FIFTY_FIFTY
            recommendation = "⚠️ Long tops can create 50/50 proportions. Try a front tuck or half-tuck to define your waist."
        
        # Neutral: Regular with mid-rise
        elif top_length == LengthType.REGULAR and bottom_rise == WaistRise.MID_RISE:
            points = 0
            ratio = ProportionRatio.BALANCED_TUCKED
            recommendation = "Neutral proportions. Tuck in for better waist definition."
        
        # Default
        else:
            points = 0
            ratio = ProportionRatio.UNDEFINED
            recommendation = "Consider your top length and bottom rise for optimal proportions."
        
        # Normalize score
        # Points range from -2 to +3, normalize to 0-1
        score = (points + 2) / 5  # -2 -> 0, +3 -> 1
        score = max(0, min(1, score))
        
        return ProportionResult(
            score=score,
            points=points,
            ratio=ratio,
            top_portion=top_portion,
            bottom_portion=bottom_portion,
            recommendation=recommendation
        )
    
    def _analyze_dress_proportions(self, dress: Garment) -> ProportionResult:
        """Analyze dress proportions."""
        dress_length = dress.attributes.length_type
        
        if dress_length == LengthType.MINI:
            # Mini dress shows 2/3 legs = good proportions
            return ProportionResult(
                score=0.85,
                points=2,
                ratio=ProportionRatio.ONE_THIRD_TWO_THIRDS,
                top_portion="mini dress (1/3)",
                bottom_portion="legs visible (2/3)",
                recommendation="✅ Mini dress creates flattering 1/3-2/3 proportions."
            )
        
        elif dress_length == LengthType.MIDI:
            # Midi can be tricky - depends on where it hits
            return ProportionResult(
                score=0.6,
                points=0,
                ratio=ProportionRatio.FIFTY_FIFTY,
                top_portion="midi dress",
                bottom_portion="calves visible",
                recommendation="⚠️ Midi length can cut at an awkward point. Ensure it hits above or below the widest part of your calf."
            )
        
        elif dress_length == LengthType.MAXI:
            # Maxi with defined waist is good
            return ProportionResult(
                score=0.75,
                points=1,
                ratio=ProportionRatio.BALANCED_TUCKED,
                top_portion="maxi dress",
                bottom_portion="full length",
                recommendation="👍 Maxi dress works well with a defined waist. Add a belt to emphasize 1/3-2/3 proportions."
            )
        
        else:
            return ProportionResult(
                score=0.7,
                points=0,
                ratio=ProportionRatio.UNDEFINED,
                top_portion="dress",
                bottom_portion="unknown",
                recommendation="Consider dress length for optimal proportions."
            )
    
    def get_best_bottom_for_top(self, top: Garment) -> Dict[str, str]:
        """Suggest best bottom attributes for a given top."""
        top_length = top.attributes.length_type
        
        if top_length == LengthType.CROP:
            return {
                "waist_rise": "high_rise",
                "reason": "High-rise with crop top creates perfect 1/3-2/3 proportions"
            }
        elif top_length == LengthType.LONGLINE:
            return {
                "waist_rise": "high_rise",
                "fit": "slim",
                "reason": "Tuck the longline top or pair with slim high-rise to balance volume"
            }
        else:
            return {
                "waist_rise": "high_rise or mid_rise",
                "reason": "Tuck in for best waist definition"
            }


def calculate_proportion_score(items: List[Garment]) -> float:
    """Quick function to get proportion score."""
    scorer = ProportionScorer()
    result = scorer.analyze_outfit(items)
    return result.score
