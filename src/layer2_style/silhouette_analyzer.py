"""
Silhouette Analyzer
Analyzes silhouette balance and coherence in outfits.
"""
from typing import List, Dict, Optional
from enum import Enum

from src.core.models import Garment, GarmentCategory, LengthType, WaistRise
from src.core import get_logger

logger = get_logger(__name__)


class SilhouetteType(str, Enum):
    """Types of silhouettes."""
    FITTED = "fitted"
    RELAXED = "relaxed"
    OVERSIZED = "oversized"
    STRUCTURED = "structured"
    FLOWING = "flowing"
    STRAIGHT = "straight"
    A_LINE = "a-line"


class SilhouetteAnalyzer:
    """
    Analyzes silhouette combinations for visual balance.
    
    Good silhouette combinations create visual interest
    while maintaining overall harmony.
    """
    
    def __init__(self):
        # Silhouette compatibility rules
        # (top_silhouette, bottom_silhouette) -> score
        self.silhouette_compatibility = {
            # Fitted tops
            ("fitted", "fitted"): 0.7,
            ("fitted", "relaxed"): 0.85,
            ("fitted", "oversized"): 0.6,
            ("fitted", "straight"): 0.8,
            ("fitted", "a-line"): 0.9,
            ("fitted", "flowing"): 0.85,
            
            # Relaxed tops
            ("relaxed", "fitted"): 0.75,
            ("relaxed", "relaxed"): 0.7,
            ("relaxed", "straight"): 0.75,
            ("relaxed", "slim"): 0.85,
            
            # Oversized tops
            ("oversized", "fitted"): 0.9,
            ("oversized", "slim"): 0.95,
            ("oversized", "straight"): 0.8,
            ("oversized", "relaxed"): 0.6,
            ("oversized", "oversized"): 0.4,
            
            # Structured tops
            ("structured", "fitted"): 0.85,
            ("structured", "straight"): 0.9,
            ("structured", "relaxed"): 0.7,
            ("structured", "slim"): 0.85,
            
            # Flowing tops
            ("flowing", "fitted"): 0.8,
            ("flowing", "slim"): 0.85,
            ("flowing", "straight"): 0.75,
        }
        
        # Visual weight mapping
        self.silhouette_volume = {
            "fitted": 1,
            "slim": 1,
            "straight": 2,
            "relaxed": 3,
            "a-line": 3,
            "flowing": 3,
            "oversized": 4,
            "structured": 2
        }
        
        # Length compatibility for proportions
        # (top_length, bottom_length) -> score
        self.length_compatibility = {
            (LengthType.CROP, LengthType.REGULAR): 0.95,  # Best pairing - crop top with regular bottom
            (LengthType.CROP, LengthType.MIDI): 0.90,     # Crop top with midi skirt
            (LengthType.CROP, LengthType.MAXI): 0.90,     # Crop top with maxi
            (LengthType.REGULAR, LengthType.REGULAR): 0.8,
            (LengthType.REGULAR, LengthType.MIDI): 0.85,
            (LengthType.LONGLINE, LengthType.REGULAR): 0.75,
        }
        
        # Waist rise preferences by top length
        self.waist_rise_recommendations = {
            LengthType.CROP: [WaistRise.HIGH_RISE, WaistRise.MID_RISE],
            LengthType.REGULAR: [WaistRise.MID_RISE, WaistRise.HIGH_RISE],
            LengthType.LONGLINE: [WaistRise.HIGH_RISE],
        }
    
    def analyze_silhouette_balance(self, items: List[Garment]) -> float:
        """
        Analyze overall silhouette balance of an outfit.
        
        Args:
            items: List of garments
            
        Returns:
            Balance score between 0 and 1
        """
        # Separate by position
        tops = self._get_items_by_position(items, "upper")
        bottoms = self._get_items_by_position(items, "lower")
        
        if not tops or not bottoms:
            return 0.7  # Neutral if missing elements
        
        # Get dominant silhouettes
        top_silhouette = self._get_dominant_silhouette(tops)
        bottom_silhouette = self._get_dominant_silhouette(bottoms)
        
        if not top_silhouette or not bottom_silhouette:
            return 0.7
        
        # Check compatibility
        compatibility = self._get_silhouette_compatibility(
            top_silhouette, bottom_silhouette
        )
        
        # Check volume balance
        volume_score = self._analyze_volume_balance(top_silhouette, bottom_silhouette)
        
        # Check length and proportion balance (new)
        proportion_score = self._analyze_length_proportions(tops, bottoms)
        
        # Combine scores
        return 0.4 * compatibility + 0.3 * volume_score + 0.3 * proportion_score
    
    def _analyze_length_proportions(
        self,
        tops: List[Garment],
        bottoms: List[Garment]
    ) -> float:
        """
        Analyze length proportions between tops and bottoms.
        
        Args:
            tops: Upper body garments
            bottoms: Lower body garments
            
        Returns:
            Proportion score between 0 and 1
        """
        if not tops or not bottoms:
            return 0.7
        
        # Get length types
        top_lengths = [t.attributes.length_type for t in tops if t.attributes.length_type]
        bottom_lengths = [b.attributes.length_type for b in bottoms if b.attributes.length_type]
        waist_rises = [b.attributes.waist_rise for b in bottoms if b.attributes.waist_rise]
        
        if not top_lengths or not bottom_lengths:
            return 0.7  # Neutral if missing data
        
        # Check crop top with high-rise pairing
        if LengthType.CROP in top_lengths:
            if WaistRise.HIGH_RISE in waist_rises:
                return 0.95  # Excellent pairing
            elif WaistRise.MID_RISE in waist_rises:
                return 0.8
            else:
                return 0.6  # Low-rise with crop top is tricky
        
        # Check longline with waist rise
        if LengthType.LONGLINE in top_lengths:
            if WaistRise.HIGH_RISE in waist_rises:
                return 0.85
            else:
                return 0.7  # Can bunch up
        
        return 0.75  # Default for regular combinations
    
    def get_recommended_waist_rise(
        self,
        top: Garment
    ) -> List[WaistRise]:
        """
        Recommend waist rise based on top length.
        
        Args:
            top: The top garment
            
        Returns:
            List of recommended waist rises
        """
        top_length = top.attributes.length_type
        if top_length and top_length in self.waist_rise_recommendations:
            return self.waist_rise_recommendations[top_length]
        return [WaistRise.MID_RISE, WaistRise.HIGH_RISE]  # Safe defaults
    
    def get_recommended_silhouette(
        self,
        existing_item: Garment,
        target_category: GarmentCategory
    ) -> List[str]:
        """
        Get recommended silhouettes to pair with an existing item.
        
        Args:
            existing_item: The item already in the outfit
            target_category: Category of item being searched for
            
        Returns:
            List of recommended silhouette types
        """
        current_silhouette = existing_item.attributes.silhouette
        if not current_silhouette:
            return ["fitted", "relaxed", "straight"]  # Safe defaults
        
        current_silhouette = current_silhouette.lower()
        recommendations = []
        
        # Check all combinations
        for silhouette in ["fitted", "relaxed", "oversized", "straight", "flowing"]:
            if target_category in [GarmentCategory.TOP, GarmentCategory.OUTERWEAR]:
                key = (silhouette, current_silhouette)
            else:
                key = (current_silhouette, silhouette)
            
            score = self.silhouette_compatibility.get(key, 0.6)
            if score >= 0.75:
                recommendations.append(silhouette)
        
        return recommendations if recommendations else ["fitted", "straight"]
    
    def _get_items_by_position(
        self,
        items: List[Garment],
        position: str
    ) -> List[Garment]:
        """Get items by body position."""
        upper_categories = {GarmentCategory.TOP, GarmentCategory.OUTERWEAR}
        lower_categories = {GarmentCategory.BOTTOM}
        
        if position == "upper":
            return [i for i in items if i.attributes.category in upper_categories]
        elif position == "lower":
            return [i for i in items if i.attributes.category in lower_categories]
        
        return []
    
    def _get_dominant_silhouette(self, items: List[Garment]) -> Optional[str]:
        """Get the dominant silhouette from a list of items."""
        silhouettes = [i.attributes.silhouette for i in items 
                      if i.attributes.silhouette]
        
        if not silhouettes:
            return None
        
        # Return the silhouette with highest volume (most impactful)
        return max(silhouettes, key=lambda s: 
                  self.silhouette_volume.get(s.lower(), 2))
    
    def _get_silhouette_compatibility(
        self,
        top_sil: str,
        bottom_sil: str
    ) -> float:
        """Get compatibility score for silhouette pair."""
        top_sil = top_sil.lower()
        bottom_sil = bottom_sil.lower()
        
        # Check direct match
        score = self.silhouette_compatibility.get((top_sil, bottom_sil))
        if score is not None:
            return score
        
        # Check similar silhouettes
        similar_map = {
            "slim": "fitted",
            "tailored": "structured",
            "loose": "relaxed",
            "wide": "oversized"
        }
        
        top_sil = similar_map.get(top_sil, top_sil)
        bottom_sil = similar_map.get(bottom_sil, bottom_sil)
        
        return self.silhouette_compatibility.get((top_sil, bottom_sil), 0.6)
    
    def _analyze_volume_balance(self, top_sil: str, bottom_sil: str) -> float:
        """
        Analyze volume balance between top and bottom.
        
        Good balance typically means:
        - Volume contrast (one fitted, one relaxed)
        - Or similar moderate volumes
        """
        top_vol = self.silhouette_volume.get(top_sil.lower(), 2)
        bottom_vol = self.silhouette_volume.get(bottom_sil.lower(), 2)
        
        # Calculate volume difference
        diff = abs(top_vol - bottom_vol)
        
        # Some contrast is good
        if diff == 2:  # Moderate contrast
            return 0.9
        elif diff == 1:  # Slight contrast
            return 0.85
        elif diff == 0:  # Same volume
            if top_vol <= 2:  # Both moderate or fitted - OK
                return 0.8
            else:  # Both voluminous - risky
                return 0.5
        elif diff >= 3:  # Extreme contrast
            return 0.75  # Can work but bold
        
        return 0.7
