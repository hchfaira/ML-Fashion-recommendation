"""
Activity Analyzer
Scores outfit appropriateness based on activity level and physical context.
"""
from typing import List, Dict, Optional, Tuple
from enum import Enum

from src.core.models import (
    Garment, UserContext, GarmentCategory, FormalityLevel,
    ActivityLevel, TransportMode, EnergyLevel, TimeOfDay
)
from src.core import get_logger

logger = get_logger(__name__)


class ComfortFactor(str, Enum):
    """Factors affecting outfit comfort for activities."""
    MOBILITY = "mobility"           # Range of movement
    BREATHABILITY = "breathability" # Air flow
    WEATHER_PROTECTION = "protection"  # Against elements
    WEIGHT = "weight"               # Heavy vs light
    DURABILITY = "durability"       # Can handle activity
    EASY_CARE = "easy_care"         # Wrinkle/stain resistant


class ActivityAnalyzer:
    """
    Analyzes activity context to score outfit appropriateness.
    
    Features:
    - Activity level matching
    - Transport mode considerations
    - Energy level awareness
    - Time of day adjustments
    - Duration considerations
    """
    
    # Category comfort profiles for different activities
    CATEGORY_ACTIVITY_SCORES = {
        # Category: {activity_level: score}
        # Using DRESS for dresses
        GarmentCategory.DRESS: {
            ActivityLevel.INTENSE: 0.2,
            ActivityLevel.ACTIVE: 0.4,
            ActivityLevel.MODERATE: 0.7,
            ActivityLevel.LIGHT: 1.0,
            ActivityLevel.SEDENTARY: 1.0,
        },
        # Outerwear for formal/structured items
        GarmentCategory.OUTERWEAR: {
            ActivityLevel.INTENSE: 0.3,
            ActivityLevel.ACTIVE: 0.5,
            ActivityLevel.MODERATE: 0.8,
            ActivityLevel.LIGHT: 1.0,
            ActivityLevel.SEDENTARY: 1.0,
        },
    }
    
    # Transport mode considerations
    TRANSPORT_REQUIREMENTS = {
        TransportMode.WALKING: {
            "comfortable_shoes": True,
            "weather_ready": True,
            "mobility": 0.8,
            "bag_type": "hands_free"
        },
        TransportMode.CYCLING: {
            "comfortable_shoes": True,
            "weather_ready": True,
            "mobility": 1.0,
            "avoid_flowy": True,
            "bag_type": "backpack"
        },
        TransportMode.DRIVING: {
            "comfortable_shoes": False,  # Can change shoes
            "weather_ready": False,       # Car protected
            "mobility": 0.5,
            "bag_type": "any"
        },
        TransportMode.PUBLIC_TRANSIT: {
            "comfortable_shoes": True,
            "weather_ready": True,
            "mobility": 0.6,
            "bag_type": "secure"
        },
        TransportMode.MIXED: {
            "comfortable_shoes": True,
            "weather_ready": True,
            "mobility": 0.7,
            "bag_type": "versatile"
        },
    }
    
    def __init__(self):
        pass
    
    def analyze_activity_context(
        self,
        context: UserContext
    ) -> Dict:
        """
        Analyze activity-related context.
        
        Returns:
            Dict with activity analysis
        """
        return {
            "activity_level": context.activity_level or ActivityLevel.MODERATE,
            "transport_mode": context.transport_mode or TransportMode.MIXED,
            "duration_hours": context.duration_hours or 8.0,
            "energy_level": context.current_energy or EnergyLevel.NORMAL,
            "time_of_day": context.time_of_day or TimeOfDay.MORNING,
            "comfort_priority": context.comfort_priority or 0.5,
            "requirements": self._derive_requirements(context)
        }
    
    def score_garment_for_activity(
        self,
        garment: Garment,
        context: UserContext
    ) -> float:
        """
        Score how appropriate a garment is for the activity context.
        
        Args:
            garment: The garment to score
            context: User context with activity info
            
        Returns:
            Score from 0.0 to 1.0
        """
        scores = []
        
        # 1. Activity level compatibility
        activity_score = self._score_activity_level(garment, context)
        scores.append(("activity_level", activity_score, 0.3))
        
        # 2. Transport mode compatibility
        transport_score = self._score_transport_mode(garment, context)
        scores.append(("transport", transport_score, 0.2))
        
        # 3. Duration appropriateness
        duration_score = self._score_duration(garment, context)
        scores.append(("duration", duration_score, 0.2))
        
        # 4. Energy level match
        energy_score = self._score_energy_match(garment, context)
        scores.append(("energy", energy_score, 0.15))
        
        # 5. Time of day appropriateness
        time_score = self._score_time_of_day(garment, context)
        scores.append(("time", time_score, 0.15))
        
        # Weighted average
        total_weight = sum(w for _, _, w in scores)
        weighted_sum = sum(s * w for _, s, w in scores)
        
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return final_score
    
    def score_outfit_for_activity(
        self,
        outfit: List[Garment],
        context: UserContext
    ) -> Tuple[float, Dict]:
        """
        Score an entire outfit for activity appropriateness.
        
        Returns:
            Tuple of (score, breakdown)
        """
        if not outfit:
            return (0.5, {})
        
        # Score each garment
        garment_scores = [
            self.score_garment_for_activity(g, context)
            for g in outfit
        ]
        
        # Overall score is minimum (weakest link)
        # But don't penalize accessories too much
        core_items = [
            (g, s) for g, s in zip(outfit, garment_scores)
            if g.attributes.category not in [
                GarmentCategory.ACCESSORY,
                GarmentCategory.BAG
            ]
        ]
        
        if core_items:
            core_scores = [s for _, s in core_items]
            min_core = min(core_scores)
            avg_core = sum(core_scores) / len(core_scores)
            overall = (min_core + avg_core) / 2
        else:
            overall = sum(garment_scores) / len(garment_scores)
        
        breakdown = {
            "overall": overall,
            "garment_scores": {
                g.id: s for g, s in zip(outfit, garment_scores)
            },
            "requirements_met": self._check_requirements(outfit, context)
        }
        
        return (overall, breakdown)
    
    def get_activity_recommendations(
        self,
        context: UserContext
    ) -> List[str]:
        """
        Get activity-based outfit recommendations.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        activity = context.activity_level or ActivityLevel.MODERATE
        transport = context.transport_mode or TransportMode.MIXED
        duration = context.duration_hours or 8.0
        
        # Activity level recommendations
        if activity == ActivityLevel.INTENSE:
            recommendations.append("Choose breathable, moisture-wicking fabrics")
            recommendations.append("Opt for athletic or performance wear")
        elif activity == ActivityLevel.ACTIVE:
            recommendations.append("Select comfortable, flexible pieces")
            recommendations.append("Consider athleisure options")
        elif activity == ActivityLevel.SEDENTARY:
            recommendations.append("You can prioritize style over comfort")
            recommendations.append("Structured pieces work well")
        
        # Transport recommendations
        if transport == TransportMode.WALKING:
            recommendations.append("Wear comfortable walking shoes")
            recommendations.append("Consider a hands-free bag")
        elif transport == TransportMode.CYCLING:
            recommendations.append("Avoid flowy skirts or wide-leg pants")
            recommendations.append("Use a backpack for belongings")
        elif transport == TransportMode.PUBLIC_TRANSIT:
            recommendations.append("Choose easy-care fabrics that resist wrinkles")
        
        # Duration recommendations
        if duration > 10:
            recommendations.append("Long day ahead - prioritize comfort")
            recommendations.append("Layer for temperature changes throughout the day")
        elif duration < 3:
            recommendations.append("Short outing - you can prioritize style")
        
        return recommendations
    
    def _score_activity_level(
        self,
        garment: Garment,
        context: UserContext
    ) -> float:
        """Score garment for activity level."""
        activity = context.activity_level or ActivityLevel.MODERATE
        category = garment.attributes.category
        
        # Check category-specific scores
        if category in self.CATEGORY_ACTIVITY_SCORES:
            return self.CATEGORY_ACTIVITY_SCORES[category].get(activity, 0.5)
        
        # Default scoring based on comfort attributes
        attrs = garment.attributes
        
        if activity in [ActivityLevel.INTENSE, ActivityLevel.ACTIVE]:
            # Need stretchy, breathable - check material profile
            score = 0.5
            
            # Check stretch via material profile
            material = getattr(attrs, 'material', None)
            if material and material.stretch:
                stretch_val = getattr(material.stretch, 'value', str(material.stretch))
                if stretch_val in ['high', 'full']:
                    score += 0.2
            
            # Check material type
            if material and material.primary:
                if material.primary.lower() in ["cotton", "jersey", "technical", "lycra", "spandex"]:
                    score += 0.2
            
            # Check silhouette for structure
            silhouette = getattr(attrs, 'silhouette_profile', None)
            if silhouette and getattr(silhouette, 'is_structured', False):
                score -= 0.3
            
            return max(0, min(1, score))
        
        elif activity == ActivityLevel.SEDENTARY:
            # Any category works
            return 0.8
        
        else:
            # Moderate - most things work
            return 0.7
    
    def _score_transport_mode(
        self,
        garment: Garment,
        context: UserContext
    ) -> float:
        """Score garment for transport mode."""
        transport = context.transport_mode
        
        if not transport:
            return 0.8  # Default okay
        
        requirements = self.TRANSPORT_REQUIREMENTS.get(transport, {})
        score = 1.0
        attrs = garment.attributes
        
        # Check footwear - look for heel info in silhouette_profile or details
        if attrs.category == GarmentCategory.SHOES:
            if requirements.get("comfortable_shoes"):
                # Check silhouette profile for heel height
                silhouette = getattr(attrs, 'silhouette_profile', None)
                heel_height = getattr(silhouette, 'heel_height', None) if silhouette else None
                
                # Also check subcategory for heels
                subcategory = getattr(attrs, 'subcategory', '') or ''
                product_type = getattr(attrs, 'product_type', '') or ''
                
                is_heels = (
                    'heel' in subcategory.lower() or
                    'heel' in product_type.lower() or
                    (heel_height and heel_height > 3)
                )
                
                if is_heels:
                    score -= 0.4
        
        # Check for flowy items when cycling - look in silhouette profile
        if requirements.get("avoid_flowy"):
            silhouette = getattr(attrs, 'silhouette_profile', None)
            is_flowy = getattr(silhouette, 'is_flowy', False) if silhouette else False
            
            # Also check fit
            fit = getattr(attrs, 'fit', '') or ''
            if is_flowy or 'flowy' in fit.lower() or 'loose' in fit.lower():
                score -= 0.5
        
        # Check mobility requirements
        mobility_req = requirements.get("mobility", 0.5)
        if mobility_req > 0.7:
            # Need mobility - penalize restrictive items
            fit = getattr(attrs, 'fit', '') or ''
            if 'tight' in fit.lower() or 'fitted' in fit.lower():
                if attrs.category == GarmentCategory.BOTTOM:
                    score -= 0.2
        
        return max(0, min(1, score))
    
    def _score_duration(
        self,
        garment: Garment,
        context: UserContext
    ) -> float:
        """Score garment for activity duration."""
        duration = context.duration_hours or 8.0
        
        if duration > 10:
            # Long day - need comfort
            attrs = garment.attributes
            score = 0.5
            
            # Prefer comfortable materials
            comfort_materials = ["cotton", "jersey", "cashmere", "silk"]
            material = getattr(attrs, 'material', None)
            if material and material.primary:
                if material.primary.lower() in comfort_materials:
                    score += 0.3
            
            # Penalize potentially uncomfortable items (heels)
            if attrs.category == GarmentCategory.SHOES:
                subcategory = getattr(attrs, 'subcategory', '') or ''
                product_type = getattr(attrs, 'product_type', '') or ''
                if 'heel' in subcategory.lower() or 'heel' in product_type.lower():
                    score -= 0.4
            
            # Check silhouette for structure
            silhouette = getattr(attrs, 'silhouette_profile', None)
            if silhouette and getattr(silhouette, 'is_structured', False):
                score -= 0.1
            
            return max(0, min(1, score))
        
        elif duration < 3:
            # Short - anything goes
            return 0.9
        
        else:
            # Medium duration - slight comfort preference
            return 0.7
    
    def _score_energy_match(
        self,
        garment: Garment,
        context: UserContext
    ) -> float:
        """Score garment based on user's energy level."""
        energy_str = context.current_energy or "normal"
        
        # Low energy - prefer easy, comfortable outfits
        if energy_str == "low":
            if garment.attributes.category in [GarmentCategory.TOP, GarmentCategory.BOTTOM]:
                return 0.9  # Easy pieces
            if garment.attributes.category == GarmentCategory.OUTERWEAR:
                return 0.6  # Might require effort
            return 0.7
        
        # High energy - can handle more complex outfits
        elif energy_str == "high":
            return 0.9  # Everything works
        
        else:  # Normal
            return 0.8
    
    def _score_time_of_day(
        self,
        garment: Garment,
        context: UserContext
    ) -> float:
        """Score garment for time of day."""
        time_str = context.time_of_day or "morning"
        
        # Convert to enum if string
        if isinstance(time_str, str):
            try:
                time = TimeOfDay(time_str)
            except ValueError:
                time = TimeOfDay.MORNING
        else:
            time = time_str
        
        attrs = garment.attributes
        
        # Evening - darker colors, more formal acceptable
        if time == TimeOfDay.EVENING:
            if attrs.primary_color and attrs.primary_color.lower() in ["black", "navy", "burgundy"]:
                return 0.9
            return 0.7
        
        # Morning/Afternoon - most things work
        elif time in [TimeOfDay.MORNING, TimeOfDay.AFTERNOON]:
            return 0.8
        
        # Night - casual, comfortable
        elif time == TimeOfDay.NIGHT:
            # Just check for casual formality
            if garment.attributes.formality_level in [FormalityLevel.VERY_CASUAL, FormalityLevel.CASUAL]:
                return 1.0
            return 0.5
        
        return 0.7
    
    def _derive_requirements(
        self,
        context: UserContext
    ) -> List[str]:
        """Derive outfit requirements from activity context."""
        requirements = []
        
        activity = context.activity_level or ActivityLevel.MODERATE
        transport = context.transport_mode
        duration = context.duration_hours or 8.0
        
        if activity in [ActivityLevel.INTENSE, ActivityLevel.ACTIVE]:
            requirements.append("breathable_fabrics")
            requirements.append("flexible_fit")
        
        if transport == TransportMode.WALKING:
            requirements.append("comfortable_footwear")
        if transport == TransportMode.CYCLING:
            requirements.append("no_flowing_items")
            requirements.append("backpack_compatible")
        
        if duration > 10:
            requirements.append("all_day_comfort")
        
        # Check comfort constraints from context
        if context.special_requirements:
            requirements.extend(context.special_requirements)
        
        return requirements
    
    def _check_requirements(
        self,
        outfit: List[Garment],
        context: UserContext
    ) -> Dict[str, bool]:
        """Check which requirements are met by the outfit."""
        requirements = self._derive_requirements(context)
        results = {}
        
        for req in requirements:
            if req == "comfortable_footwear":
                footwear = [g for g in outfit if g.attributes.category == GarmentCategory.SHOES]
                # Check for heels based on subcategory/product_type
                def is_comfortable_shoe(g):
                    subcategory = getattr(g.attributes, 'subcategory', '') or ''
                    product_type = getattr(g.attributes, 'product_type', '') or ''
                    return not ('heel' in subcategory.lower() or 'heel' in product_type.lower())
                
                results[req] = any(is_comfortable_shoe(g) for g in footwear) if footwear else True
            
            elif req == "no_flowing_items":
                def is_flowy(g):
                    silhouette = getattr(g.attributes, 'silhouette_profile', None)
                    fit = getattr(g.attributes, 'fit', '') or ''
                    return (
                        (silhouette and getattr(silhouette, 'is_flowy', False)) or
                        'flowy' in fit.lower() or
                        'loose' in fit.lower()
                    )
                results[req] = not any(is_flowy(g) for g in outfit)
            
            elif req == "breathable_fabrics":
                breathable = ["cotton", "linen", "jersey", "technical"]
                def has_breathable_material(g):
                    material = getattr(g.attributes, 'material', None)
                    if material and material.primary:
                        return material.primary.lower() in breathable
                    return False
                results[req] = any(has_breathable_material(g) for g in outfit)
            
            else:
                # Default to True for unhandled requirements
                results[req] = True
        
        return results
