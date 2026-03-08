"""
User History Manager
Manages user preferences and history for personalization.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict

from src.core.models import Garment, Outfit, ColorInfo
from src.core import get_logger

logger = get_logger(__name__)


class UserHistoryManager:
    """
    Manages user interaction history and learned preferences.
    
    Tracks:
    - Outfit selections
    - Color preferences
    - Style preferences
    - Feedback on recommendations
    """
    
    def __init__(self):
        # In-memory storage (would be database in production)
        self._user_preferences: Dict[str, Dict] = {}
        self._outfit_history: Dict[str, List[Dict]] = defaultdict(list)
        self._feedback: Dict[str, List[Dict]] = defaultdict(list)
    
    async def score_against_preferences(
        self,
        garments: List[Garment],
        user_id: str
    ) -> float:
        """
        Score outfit alignment with user preferences.
        
        Args:
            garments: List of garments
            user_id: User identifier
            
        Returns:
            Preference alignment score
        """
        preferences = await self.get_user_preferences(user_id)
        
        if not preferences:
            return 0.7  # Neutral for new users
        
        scores = []
        
        for garment in garments:
            # Color preference alignment
            color_score = self._score_color_preference(
                garment.attributes.color,
                preferences.get("preferred_colors", []),
                preferences.get("disliked_colors", [])
            )
            
            # Style preference alignment
            style_score = self._score_style_preference(
                garment.attributes.style_tags,
                preferences.get("preferred_styles", []),
                preferences.get("disliked_styles", [])
            )
            
            scores.append(0.5 * color_score + 0.5 * style_score)
        
        return sum(scores) / len(scores) if scores else 0.7
    
    async def get_user_preferences(self, user_id: str) -> Dict:
        """
        Get stored user preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of preferences
        """
        if user_id not in self._user_preferences:
            return {}
        
        return self._user_preferences[user_id]
    
    async def update_preferences(
        self,
        user_id: str,
        preferences: Dict
    ) -> None:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier
            preferences: New/updated preferences
        """
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {}
        
        self._user_preferences[user_id].update(preferences)
        logger.info(f"Updated preferences for user {user_id}")
    
    async def record_outfit_selection(
        self,
        user_id: str,
        outfit: Outfit,
        context: Dict = None
    ) -> None:
        """
        Record that user selected an outfit.
        
        Args:
            user_id: User identifier
            outfit: Selected outfit
            context: Selection context
        """
        record = {
            "outfit_id": outfit.id,
            "items": [item.garment.id for item in outfit.items],
            "timestamp": datetime.utcnow().isoformat(),
            "context": context or {}
        }
        
        self._outfit_history[user_id].append(record)
        
        # Update learned preferences
        await self._learn_from_selection(user_id, outfit)
    
    async def record_feedback(
        self,
        user_id: str,
        outfit_id: str,
        feedback_type: str,
        feedback_value: Any
    ) -> None:
        """
        Record user feedback on a recommendation.
        
        Args:
            user_id: User identifier
            outfit_id: Outfit that received feedback
            feedback_type: Type of feedback (rating, liked, disliked)
            feedback_value: Feedback value
        """
        record = {
            "outfit_id": outfit_id,
            "type": feedback_type,
            "value": feedback_value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._feedback[user_id].append(record)
        
        # Update preferences based on feedback
        await self._learn_from_feedback(user_id, record)
    
    async def get_outfit_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get recent outfit history for user.
        
        Args:
            user_id: User identifier
            limit: Maximum records to return
            
        Returns:
            List of outfit history records
        """
        history = self._outfit_history.get(user_id, [])
        return history[-limit:]
    
    async def get_similar_past_outfits(
        self,
        user_id: str,
        context: Dict
    ) -> List[Dict]:
        """
        Find past outfits in similar contexts.
        
        Args:
            user_id: User identifier
            context: Current context
            
        Returns:
            List of relevant past outfits
        """
        history = self._outfit_history.get(user_id, [])
        
        similar = []
        for record in history:
            if self._context_matches(record.get("context", {}), context):
                similar.append(record)
        
        return similar
    
    async def _learn_from_selection(
        self,
        user_id: str,
        outfit: Outfit
    ) -> None:
        """Learn preferences from outfit selection."""
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {
                "preferred_colors": [],
                "preferred_styles": [],
                "disliked_colors": [],
                "disliked_styles": [],
                "color_counts": defaultdict(int),
                "style_counts": defaultdict(int)
            }
        
        prefs = self._user_preferences[user_id]
        
        # Count colors and styles selected
        for item in outfit.items:
            color = item.garment.attributes.color.primary.lower()
            prefs.setdefault("color_counts", defaultdict(int))
            prefs["color_counts"][color] += 1
            
            for style in item.garment.attributes.style_tags:
                prefs.setdefault("style_counts", defaultdict(int))
                prefs["style_counts"][style.lower()] += 1
        
        # Update preferred lists (top 5)
        color_counts = prefs.get("color_counts", {})
        style_counts = prefs.get("style_counts", {})
        
        if color_counts:
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
            prefs["preferred_colors"] = [c for c, _ in sorted_colors[:5]]
        
        if style_counts:
            sorted_styles = sorted(style_counts.items(), key=lambda x: x[1], reverse=True)
            prefs["preferred_styles"] = [s for s, _ in sorted_styles[:5]]
    
    async def _learn_from_feedback(
        self,
        user_id: str,
        feedback: Dict
    ) -> None:
        """Learn preferences from explicit feedback."""
        # This would use the feedback to adjust preferences
        # Simplified for demo
        pass
    
    def _score_color_preference(
        self,
        color: ColorInfo,
        preferred: List[str],
        disliked: List[str]
    ) -> float:
        """Score color against preferences."""
        c = color.primary.lower()
        
        if c in [d.lower() for d in disliked]:
            return 0.3
        
        if c in [p.lower() for p in preferred]:
            return 0.95
        
        return 0.7  # Neutral
    
    def _score_style_preference(
        self,
        styles: List[str],
        preferred: List[str],
        disliked: List[str]
    ) -> float:
        """Score styles against preferences."""
        if not styles:
            return 0.7
        
        style_set = {s.lower() for s in styles}
        preferred_set = {p.lower() for p in preferred}
        disliked_set = {d.lower() for d in disliked}
        
        # Check for disliked
        if style_set & disliked_set:
            return 0.4
        
        # Check for preferred
        overlap = len(style_set & preferred_set)
        if overlap >= 2:
            return 0.95
        elif overlap == 1:
            return 0.8
        
        return 0.7
    
    def _context_matches(self, past_context: Dict, current_context: Dict) -> bool:
        """Check if contexts are similar."""
        # Match on occasion
        if past_context.get("occasion") == current_context.get("occasion"):
            return True
        
        # Match on similar weather
        past_temp = past_context.get("temperature")
        curr_temp = current_context.get("temperature")
        if past_temp and curr_temp and abs(past_temp - curr_temp) < 10:
            return True
        
        return False
