"""
Schedule Analyzer
Analyzes user's daily schedule to optimize outfit recommendations.
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime, time
from enum import Enum

from src.core.models import (
    Garment, UserContext, Occasion, FormalityLevel,
    ScheduleEvent
)
from src.core import get_logger

logger = get_logger(__name__)


class TransitionStrategy(str, Enum):
    """Strategy for handling multiple occasions."""
    SINGLE_OUTFIT = "single_outfit"       # One versatile outfit for all
    SMART_LAYERS = "smart_layers"         # Add/remove layers to transition
    ACCESSORY_SWAP = "accessory_swap"     # Change accessories to elevate
    FULL_CHANGE = "full_change"           # Recommend separate outfits


class ScheduleAnalyzer:
    """
    Analyzes user's schedule to provide context-aware recommendations.
    
    Features:
    - Multi-event day handling
    - Transition outfit recommendations
    - Time-appropriate suggestions
    - Formality range calculation
    """
    
    def __init__(self):
        # Formality mapping for occasions
        self.occasion_formality = {
            Occasion.CASUAL: 1,
            Occasion.DAILY_WEAR: 1,
            Occasion.WEEKEND: 1,
            Occasion.BEACH: 1,
            Occasion.GYM: 1,
            Occasion.SPORT: 2,
            Occasion.OUTDOOR: 2,
            Occasion.TRAVEL: 2,
            Occasion.WORK: 3,
            Occasion.BUSINESS: 4,
            Occasion.INTERVIEW: 4,
            Occasion.DATE: 3,
            Occasion.EVENING: 4,
            Occasion.COCKTAIL: 5,
            Occasion.EVENT: 4,
            Occasion.FORMAL: 5,
            Occasion.WEDDING: 5,
        }
    
    def analyze_schedule(
        self,
        context: UserContext
    ) -> Dict:
        """
        Analyze user's schedule and determine outfit requirements.
        
        Returns:
            Dict with schedule analysis and recommendations
        """
        schedule = context.schedule
        
        if not schedule:
            # Single occasion mode
            return {
                "mode": "single_occasion",
                "primary_occasion": context.occasion,
                "formality_range": self._get_formality_range(context.occasion),
                "transition_needed": False,
                "strategy": TransitionStrategy.SINGLE_OUTFIT
            }
        
        # Multi-event analysis
        occasions = [event.occasion for event in schedule if event.occasion]
        
        if not occasions:
            occasions = [context.occasion] if context.occasion else [Occasion.CASUAL]
        
        # Calculate formality range
        formality_levels = [
            self.occasion_formality.get(occ, 2) for occ in occasions
        ]
        min_formality = min(formality_levels)
        max_formality = max(formality_levels)
        formality_gap = max_formality - min_formality
        
        # Determine strategy
        if formality_gap <= 1:
            strategy = TransitionStrategy.SINGLE_OUTFIT
        elif formality_gap == 2:
            strategy = TransitionStrategy.SMART_LAYERS
        elif formality_gap == 3:
            strategy = TransitionStrategy.ACCESSORY_SWAP
        else:
            strategy = TransitionStrategy.FULL_CHANGE
        
        # Find primary event (longest or most formal)
        primary_event = self._find_primary_event(schedule)
        
        # Calculate time distribution
        indoor_ratio, outdoor_ratio = self._calculate_indoor_outdoor_ratio(schedule)
        
        return {
            "mode": "multi_occasion",
            "events": [
                {
                    "name": e.name,
                    "time": e.time,
                    "occasion": e.occasion.value if e.occasion else "casual",
                    "duration": e.duration_hours,
                    "indoor": e.indoor
                }
                for e in schedule
            ],
            "primary_occasion": primary_event.occasion if primary_event else context.occasion,
            "all_occasions": [o.value for o in occasions],
            "formality_range": {
                "min": min_formality,
                "max": max_formality,
                "gap": formality_gap
            },
            "transition_needed": formality_gap > 1,
            "strategy": strategy,
            "indoor_ratio": indoor_ratio,
            "outdoor_ratio": outdoor_ratio,
            "total_duration_hours": sum(e.duration_hours for e in schedule)
        }
    
    def get_required_formality_level(
        self,
        context: UserContext
    ) -> Tuple[FormalityLevel, FormalityLevel]:
        """
        Get min and max formality levels for the context.
        
        Returns:
            Tuple of (min_formality, max_formality)
        """
        analysis = self.analyze_schedule(context)
        
        if analysis["mode"] == "single_occasion":
            formality_range = analysis["formality_range"]
        else:
            formality_range = analysis["formality_range"]
        
        # Map numeric levels to FormalityLevel
        level_map = {
            1: FormalityLevel.CASUAL,
            2: FormalityLevel.SMART_CASUAL,
            3: FormalityLevel.BUSINESS_CASUAL,
            4: FormalityLevel.BUSINESS,
            5: FormalityLevel.FORMAL,
            6: FormalityLevel.BLACK_TIE
        }
        
        if isinstance(formality_range, dict):
            min_level = level_map.get(formality_range["min"], FormalityLevel.CASUAL)
            max_level = level_map.get(formality_range["max"], FormalityLevel.SMART_CASUAL)
        else:
            min_level = formality_range[0]
            max_level = formality_range[1]
        
        return (min_level, max_level)
    
    def recommend_transition_pieces(
        self,
        base_outfit: List[Garment],
        target_occasion: Occasion
    ) -> Dict:
        """
        Recommend pieces to add/swap for occasion transition.
        
        Args:
            base_outfit: Current outfit
            target_occasion: Occasion to transition to
            
        Returns:
            Dict with transition recommendations
        """
        current_formality = self._calculate_outfit_formality(base_outfit)
        target_formality = self.occasion_formality.get(target_occasion, 2)
        
        recommendations = {
            "direction": "elevate" if target_formality > current_formality else "casualize",
            "add": [],
            "remove": [],
            "swap": []
        }
        
        if target_formality > current_formality:
            # Elevate outfit
            recommendations["add"] = [
                "blazer or structured jacket",
                "statement jewelry",
                "dress shoes or heels"
            ]
            recommendations["swap"] = [
                ("sneakers", "loafers or heels"),
                ("t-shirt", "blouse or button-down"),
                ("casual bag", "structured handbag")
            ]
        else:
            # Casualize outfit
            recommendations["remove"] = [
                "blazer or jacket",
                "tie",
                "formal jewelry"
            ]
            recommendations["swap"] = [
                ("dress shoes", "clean sneakers"),
                ("button-down", "casual top"),
                ("structured bag", "crossbody or tote")
            ]
        
        return recommendations
    
    def _get_formality_range(
        self,
        occasion: Optional[Occasion]
    ) -> Tuple[FormalityLevel, FormalityLevel]:
        """Get formality range for a single occasion."""
        ranges = {
            Occasion.CASUAL: (FormalityLevel.VERY_CASUAL, FormalityLevel.CASUAL),
            Occasion.DAILY_WEAR: (FormalityLevel.CASUAL, FormalityLevel.SMART_CASUAL),
            Occasion.WEEKEND: (FormalityLevel.VERY_CASUAL, FormalityLevel.CASUAL),
            Occasion.WORK: (FormalityLevel.SMART_CASUAL, FormalityLevel.BUSINESS_CASUAL),
            Occasion.BUSINESS: (FormalityLevel.BUSINESS_CASUAL, FormalityLevel.BUSINESS),
            Occasion.INTERVIEW: (FormalityLevel.BUSINESS_CASUAL, FormalityLevel.BUSINESS),
            Occasion.DATE: (FormalityLevel.SMART_CASUAL, FormalityLevel.BUSINESS_CASUAL),
            Occasion.EVENING: (FormalityLevel.SMART_CASUAL, FormalityLevel.FORMAL),
            Occasion.FORMAL: (FormalityLevel.BUSINESS, FormalityLevel.BLACK_TIE),
            Occasion.WEDDING: (FormalityLevel.BUSINESS_CASUAL, FormalityLevel.FORMAL),
            Occasion.COCKTAIL: (FormalityLevel.SMART_CASUAL, FormalityLevel.FORMAL),
            Occasion.SPORT: (FormalityLevel.VERY_CASUAL, FormalityLevel.CASUAL),
            Occasion.GYM: (FormalityLevel.VERY_CASUAL, FormalityLevel.VERY_CASUAL),
            Occasion.BEACH: (FormalityLevel.VERY_CASUAL, FormalityLevel.VERY_CASUAL),
            Occasion.OUTDOOR: (FormalityLevel.VERY_CASUAL, FormalityLevel.CASUAL),
            Occasion.TRAVEL: (FormalityLevel.CASUAL, FormalityLevel.SMART_CASUAL),
            Occasion.EVENT: (FormalityLevel.SMART_CASUAL, FormalityLevel.FORMAL),
        }
        
        return ranges.get(occasion, (FormalityLevel.CASUAL, FormalityLevel.SMART_CASUAL))
    
    def _find_primary_event(
        self,
        schedule: List[ScheduleEvent]
    ) -> Optional[ScheduleEvent]:
        """Find the primary event (most important to dress for)."""
        if not schedule:
            return None
        
        # Score events by: formality × duration
        scored = []
        for event in schedule:
            formality = self.occasion_formality.get(event.occasion, 2) if event.occasion else 2
            score = formality * event.duration_hours
            scored.append((score, event))
        
        # Return highest scored event
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
    
    def _calculate_indoor_outdoor_ratio(
        self,
        schedule: List[ScheduleEvent]
    ) -> Tuple[float, float]:
        """Calculate ratio of indoor vs outdoor time."""
        if not schedule:
            return (1.0, 0.0)
        
        total_hours = sum(e.duration_hours for e in schedule)
        if total_hours == 0:
            return (1.0, 0.0)
        
        indoor_hours = sum(e.duration_hours for e in schedule if e.indoor)
        outdoor_hours = total_hours - indoor_hours
        
        return (indoor_hours / total_hours, outdoor_hours / total_hours)
    
    def _calculate_outfit_formality(
        self,
        garments: List[Garment]
    ) -> int:
        """Calculate overall formality level of an outfit."""
        if not garments:
            return 2
        
        formality_scores = {
            FormalityLevel.VERY_CASUAL: 1,
            FormalityLevel.CASUAL: 2,
            FormalityLevel.SMART_CASUAL: 3,
            FormalityLevel.BUSINESS_CASUAL: 4,
            FormalityLevel.BUSINESS: 5,
            FormalityLevel.FORMAL: 6,
            FormalityLevel.BLACK_TIE: 7
        }
        
        scores = [
            formality_scores.get(g.attributes.formality_level, 2)
            for g in garments
        ]
        
        # Outfit formality is roughly the average, pulled up by the most formal piece
        avg = sum(scores) / len(scores)
        max_score = max(scores)
        
        return int((avg + max_score) / 2)
