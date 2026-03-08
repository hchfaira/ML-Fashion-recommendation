"""
Wardrobe Rotation Service
Tracks item usage and promotes variety in outfit recommendations.
"""
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path

from src.core.models import Garment, UserContext, GarmentCategory
from src.core import get_logger

logger = get_logger(__name__)


class WardrobeRotationService:
    """
    Service to track garment usage and promote wardrobe variety.
    
    Features:
    - Track recently worn items
    - Suggest underutilized pieces
    - Seasonal rotation recommendations
    - "Shop your closet" suggestions
    """
    
    # Minimum days before re-wearing items by category
    DEFAULT_ROTATION_DAYS = {
        GarmentCategory.TOP: 3,
        GarmentCategory.BOTTOM: 2,
        GarmentCategory.DRESS: 5,
        GarmentCategory.OUTERWEAR: 1,
        GarmentCategory.SHOES: 2,
        GarmentCategory.ACCESSORY: 1,
        GarmentCategory.BAG: 3,
    }
    
    def __init__(self, history_file: Optional[Path] = None):
        """
        Initialize rotation service.
        
        Args:
            history_file: Optional path to persist usage history
        """
        self.history_file = history_file
        self.usage_history: Dict[str, List[datetime]] = defaultdict(list)
        self.favorite_combinations: List[List[str]] = []
        
        if history_file and history_file.exists():
            self._load_history()
    
    def record_wear(
        self,
        garment_ids: List[str],
        worn_date: Optional[datetime] = None
    ) -> None:
        """
        Record that items were worn on a specific date.
        
        Args:
            garment_ids: List of garment IDs that were worn
            worn_date: Date worn (defaults to today)
        """
        date = worn_date or datetime.now()
        
        for gid in garment_ids:
            self.usage_history[gid].append(date)
            # Keep last 100 wears per item
            self.usage_history[gid] = self.usage_history[gid][-100:]
        
        logger.debug(f"Recorded wear for {len(garment_ids)} items on {date.date()}")
        
        if self.history_file:
            self._save_history()
    
    def get_recently_worn(
        self,
        days: int = 7
    ) -> Set[str]:
        """
        Get IDs of items worn in the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Set of garment IDs
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        recently_worn = set()
        for gid, dates in self.usage_history.items():
            if any(d > cutoff for d in dates):
                recently_worn.add(gid)
        
        return recently_worn
    
    def calculate_freshness_score(
        self,
        garment: Garment,
        context: UserContext
    ) -> float:
        """
        Calculate how "fresh" a garment is (not recently worn).
        
        Higher score = more fresh (not worn recently)
        
        Args:
            garment: The garment to score
            context: User context with recently worn info
            
        Returns:
            Score from 0.0 (worn yesterday) to 1.0 (not worn recently)
        """
        garment_id = garment.id
        
        # Check context's recently worn items first
        if garment_id in (context.recently_worn_items or []):
            return 0.2  # Low but not zero
        
        # Check usage history
        wear_dates = self.usage_history.get(garment_id, [])
        
        if not wear_dates:
            return 1.0  # Never worn or no history = fresh
        
        # Get most recent wear
        last_worn = max(wear_dates)
        days_since = (datetime.now() - last_worn).days
        
        # Get category-specific rotation period
        rotation_days = self.DEFAULT_ROTATION_DAYS.get(
            garment.attributes.category,
            3  # Default
        )
        
        if days_since >= rotation_days * 2:
            return 1.0  # Very fresh
        elif days_since >= rotation_days:
            return 0.8  # Fresh enough
        elif days_since == 0:
            return 0.1  # Worn today
        else:
            # Linear interpolation
            return 0.2 + (days_since / rotation_days) * 0.6
    
    def score_outfit_freshness(
        self,
        outfit: List[Garment],
        context: UserContext
    ) -> float:
        """
        Calculate overall freshness score for an outfit.
        
        Args:
            outfit: List of garments in the outfit
            context: User context
            
        Returns:
            Average freshness score
        """
        if not outfit:
            return 1.0
        
        scores = [
            self.calculate_freshness_score(g, context)
            for g in outfit
        ]
        
        # Use geometric mean to penalize any recently worn items
        product = 1.0
        for s in scores:
            product *= s
        
        return product ** (1 / len(scores))
    
    def get_forgotten_gems(
        self,
        wardrobe: List[Garment],
        min_days: int = 30
    ) -> List[Garment]:
        """
        Find items that haven't been worn in a while.
        
        Args:
            wardrobe: Full wardrobe
            min_days: Minimum days since last wear
            
        Returns:
            List of "forgotten" garments
        """
        cutoff = datetime.now() - timedelta(days=min_days)
        forgotten = []
        
        for garment in wardrobe:
            wear_dates = self.usage_history.get(garment.id, [])
            
            if not wear_dates:
                # Never worn - definitely a gem!
                forgotten.append(garment)
            elif max(wear_dates) < cutoff:
                # Not worn recently
                forgotten.append(garment)
        
        logger.info(f"Found {len(forgotten)} forgotten gems from {len(wardrobe)} items")
        return forgotten
    
    def suggest_featured_item(
        self,
        wardrobe: List[Garment],
        context: UserContext
    ) -> Optional[Garment]:
        """
        Suggest an underutilized item to feature in today's outfit.
        
        Args:
            wardrobe: Full wardrobe
            context: User context
            
        Returns:
            Suggested garment to feature, or None
        """
        # Check if user already wants to feature something
        if context.items_to_feature:
            for gid in context.items_to_feature:
                matching = [g for g in wardrobe if g.id == gid]
                if matching:
                    return matching[0]
        
        # Find forgotten gems that match the occasion
        gems = self.get_forgotten_gems(wardrobe, min_days=21)
        
        if not gems:
            return None
        
        # Filter by occasion appropriateness
        occasion = context.occasion
        if occasion:
            def matches_occasion(g):
                # Check occasion_profile if available
                occasion_profile = getattr(g.attributes, 'occasion_profile', None)
                if occasion_profile:
                    suitable = getattr(occasion_profile, 'suitable_occasions', None)
                    if suitable:
                        return occasion in suitable
                # If no profile, allow it
                return True
            
            gems = [g for g in gems if matches_occasion(g)]
        
        # Return the oldest unworn item
        if gems:
            gem_dates = [
                (g, max(self.usage_history.get(g.id, [datetime.min])))
                for g in gems
            ]
            gem_dates.sort(key=lambda x: x[1])
            return gem_dates[0][0]
        
        return None
    
    def get_wear_statistics(
        self,
        wardrobe: List[Garment]
    ) -> Dict:
        """
        Get usage statistics for the wardrobe.
        
        Returns:
            Dict with usage statistics
        """
        stats = {
            "total_items": len(wardrobe),
            "never_worn": 0,
            "worn_this_week": 0,
            "worn_this_month": 0,
            "most_worn": [],
            "least_worn": [],
            "by_category": {}
        }
        
        week_ago = datetime.now() - timedelta(days=7)
        month_ago = datetime.now() - timedelta(days=30)
        
        wear_counts = []
        
        for garment in wardrobe:
            dates = self.usage_history.get(garment.id, [])
            total_wears = len(dates)
            
            if total_wears == 0:
                stats["never_worn"] += 1
            
            recent_week = sum(1 for d in dates if d > week_ago)
            recent_month = sum(1 for d in dates if d > month_ago)
            
            if recent_week > 0:
                stats["worn_this_week"] += 1
            if recent_month > 0:
                stats["worn_this_month"] += 1
            
            wear_counts.append({
                "id": garment.id,
                "name": getattr(garment, 'name', garment.id),
                "category": garment.attributes.category.value,
                "total_wears": total_wears,
                "recent_wears": recent_month
            })
            
            # By category stats
            cat = garment.attributes.category.value
            if cat not in stats["by_category"]:
                stats["by_category"][cat] = {
                    "count": 0,
                    "worn_this_month": 0
                }
            stats["by_category"][cat]["count"] += 1
            if recent_month > 0:
                stats["by_category"][cat]["worn_this_month"] += 1
        
        # Sort for most/least worn
        wear_counts.sort(key=lambda x: x["total_wears"], reverse=True)
        stats["most_worn"] = wear_counts[:5]
        stats["least_worn"] = [
            w for w in wear_counts
            if w["total_wears"] > 0
        ][-5:]
        
        return stats
    
    def _load_history(self) -> None:
        """Load usage history from file."""
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
            
            self.usage_history = defaultdict(list)
            for gid, dates in data.get("usage_history", {}).items():
                self.usage_history[gid] = [
                    datetime.fromisoformat(d) for d in dates
                ]
            self.favorite_combinations = data.get("favorites", [])
            
            logger.info(f"Loaded history for {len(self.usage_history)} items")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
    
    def _save_history(self) -> None:
        """Save usage history to file."""
        try:
            data = {
                "usage_history": {
                    gid: [d.isoformat() for d in dates]
                    for gid, dates in self.usage_history.items()
                },
                "favorites": self.favorite_combinations
            }
            
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
