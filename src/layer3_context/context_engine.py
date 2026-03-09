"""
Context Engine - Main Orchestrator for Layer 3
Combines all contextual factors to filter and adjust recommendations.

Enhanced with body measurements integration for:
- Fit prediction based on body metrics
- Proportion harmony analysis
- Color harmony based on skin/hair analysis
"""
from typing import List, Dict, Any, Optional, Set, Union, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
from enum import Enum
import json

from src.core.models import (
    Garment, Outfit, UserContext, WeatherContext, 
    Occasion, Season, FormalityLevel
)
from src.core import get_logger
from .weather_service import WeatherService
from .occasion_analyzer import OccasionAnalyzer
from .morphology_advisor import MorphologyAdvisor
from .user_history import UserHistoryManager
from .schedule_analyzer import ScheduleAnalyzer, TransitionStrategy
from .wardrobe_rotation import WardrobeRotationService
from .activity_analyzer import ActivityAnalyzer
from .fit_predictor import FitPredictor, FitPrediction, OutfitFitPrediction
from .proportion_harmonizer import ProportionHarmonizer, ProportionAnalysis, ProportionScore
from .color_harmony_advisor import ColorHarmonyAdvisor, ColorProfile, ColorHarmonyScore

if TYPE_CHECKING:
    from .user_profile.models import BodyMetrics, StyleProfile

logger = get_logger(__name__)


class ContextCriteria(str, Enum):
    """Available context criteria for scoring."""
    WEATHER = "weather"
    OCCASION = "occasion"
    MORPHOLOGY = "morphology"
    PREFERENCE = "preference"
    ACTIVITY = "activity"
    FRESHNESS = "freshness"
    FIT = "fit"  # New: Size/fit compatibility
    PROPORTION = "proportion"  # New: Body proportion harmony
    COLOR_HARMONY = "color_harmony"  # New: Color matching based on skin/hair


# Default configuration path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "input" / "context_config.json"

# Default criteria - the most critical ones for outfit recommendations
DEFAULT_CRITERIA: List[ContextCriteria] = [
    ContextCriteria.OCCASION,
    ContextCriteria.WEATHER,
]

# Default weights for each criterion
DEFAULT_WEIGHTS: Dict[ContextCriteria, float] = {
    ContextCriteria.WEATHER: 0.15,
    ContextCriteria.OCCASION: 0.20,
    ContextCriteria.MORPHOLOGY: 0.15,
    ContextCriteria.PREFERENCE: 0.10,
    ContextCriteria.ACTIVITY: 0.05,
    ContextCriteria.FRESHNESS: 0.05,
    ContextCriteria.FIT: 0.10,
    ContextCriteria.PROPORTION: 0.10,
    ContextCriteria.COLOR_HARMONY: 0.10,
}

# Full criteria set for comprehensive analysis
ALL_CRITERIA: List[ContextCriteria] = list(ContextCriteria)


def load_context_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load context configuration from JSON file.
    
    Args:
        config_path: Path to config file. Uses DEFAULT_CONFIG_PATH if not provided.
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    
    # Return default config if file doesn't exist
    logger.warning(f"Config file not found at {path}, using defaults")
    return {
        "criteria": {
            "occasion": {"enabled": True, "weight": 0.30},
            "weather": {"enabled": True, "weight": 0.20},
            "morphology": {"enabled": False, "weight": 0.15},
            "preference": {"enabled": False, "weight": 0.15},
            "activity": {"enabled": False, "weight": 0.12},
            "freshness": {"enabled": False, "weight": 0.08},
        },
        "comprehensive_analysis": {
            "include_schedule": True,
            "include_activity": True,
            "include_season": True,
            "include_wardrobe_stats": True,
            "include_featured_item": True,
        },
        "scoring": {
            "style_context_blend": {"style_weight": 0.6, "context_weight": 0.4},
            "normalize_weights": True,
        }
    }


class ContextEngine:
    """
    Combines contextual factors to enhance recommendations.
    
    Context factors (configurable via JSON config or parameters):
    - Weather (temperature, conditions)
    - Occasion (work, casual, event)
    - Morphology (body type)
    - Preference (user history, past choices)
    - Activity (energy level, transport, duration)
    - Freshness (wardrobe rotation)
    
    Configuration can be provided via:
    1. config_path: Path to JSON configuration file
    2. criteria/weights: Direct parameters (override config file)
    
    Args:
        config_path: Path to JSON configuration file.
        criteria: List of ContextCriteria to use (overrides config file).
        weights: Dict of criterion weights (overrides config file).
        history_file: Optional path for wardrobe history persistence.
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        criteria: Optional[List[ContextCriteria]] = None,
        weights: Optional[Dict[ContextCriteria, float]] = None,
        history_file: Optional[Path] = None
    ):
        """
        Initialize ContextEngine with configurable criteria.
        
        Args:
            config_path: Path to JSON config file. Uses default if not provided.
            criteria: List of criteria to use. Overrides config file if provided.
            weights: Dict of criterion weights. Overrides config file if provided.
            history_file: Optional path for wardrobe history persistence.
        """
        # Load configuration from file
        self._config = load_context_config(config_path)
        self._config_path = config_path
        
        # Parse criteria and weights from config or use parameters
        if criteria is not None:
            # Direct parameter overrides config
            self._criteria: Set[ContextCriteria] = set(criteria)
        else:
            # Load from config file
            self._criteria = self._parse_criteria_from_config()
        
        if weights is not None:
            # Direct parameter overrides config
            self._weights: Dict[ContextCriteria, float] = weights.copy()
        else:
            # Load from config file
            self._weights = self._parse_weights_from_config()
        
        # Store comprehensive analysis config
        self._analysis_config = self._config.get("comprehensive_analysis", {})
        self._scoring_config = self._config.get("scoring", {})
        
        logger.info(f"ContextEngine initialized with criteria: {[c.value for c in self._criteria]}")
        logger.info(f"Weights: {[(c.value, w) for c, w in self._weights.items()]}")
        
        # Initialize services
        self.weather_service = WeatherService()
        self.occasion_analyzer = OccasionAnalyzer()
        self.morphology_advisor = MorphologyAdvisor()
        self.user_history = UserHistoryManager()
        
        # Enhanced services
        self.schedule_analyzer = ScheduleAnalyzer()
        self.wardrobe_rotation = WardrobeRotationService(history_file)
        self.activity_analyzer = ActivityAnalyzer()
        
        # Body measurement integration services
        self.fit_predictor = FitPredictor()
        self.proportion_harmonizer = ProportionHarmonizer()
        self.color_harmony_advisor = ColorHarmonyAdvisor()
        
        # Style profile cache (set via set_style_profile)
        self._style_profile: Optional["StyleProfile"] = None
    
    def _parse_criteria_from_config(self) -> Set[ContextCriteria]:
        """Parse enabled criteria from config."""
        criteria_config = self._config.get("criteria", {})
        enabled = set()
        
        for criterion_name, settings in criteria_config.items():
            if settings.get("enabled", False):
                try:
                    criterion = ContextCriteria(criterion_name)
                    enabled.add(criterion)
                except ValueError:
                    logger.warning(f"Unknown criterion in config: {criterion_name}")
        
        # Fall back to defaults if nothing enabled
        if not enabled:
            logger.warning("No criteria enabled in config, using defaults")
            return set(DEFAULT_CRITERIA)
        
        return enabled
    
    def _parse_weights_from_config(self) -> Dict[ContextCriteria, float]:
        """Parse criterion weights from config."""
        criteria_config = self._config.get("criteria", {})
        weights = {}
        
        for criterion_name, settings in criteria_config.items():
            try:
                criterion = ContextCriteria(criterion_name)
                weights[criterion] = settings.get("weight", DEFAULT_WEIGHTS.get(criterion, 0.1))
            except ValueError:
                continue
        
        # Ensure all criteria have weights
        for criterion in ContextCriteria:
            if criterion not in weights:
                weights[criterion] = DEFAULT_WEIGHTS.get(criterion, 0.1)
        
        return weights
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()
    
    @property
    def active_criteria(self) -> List[ContextCriteria]:
        """Get list of active criteria."""
        return list(self._criteria)
    
    @property
    def criterion_weights(self) -> Dict[ContextCriteria, float]:
        """Get current criterion weights."""
        return self._weights.copy()
    
    # Keep CRITERION_WEIGHTS for backward compatibility
    @property
    def CRITERION_WEIGHTS(self) -> Dict[ContextCriteria, float]:
        """Backward compatible access to weights."""
        return self._weights
    
    def set_criteria(self, criteria: List[ContextCriteria]) -> None:
        """
        Update the active criteria.
        
        Args:
            criteria: New list of criteria to use.
        """
        self._criteria = set(criteria)
        logger.info(f"Criteria updated to: {[c.value for c in self._criteria]}")
    
    def add_criterion(self, criterion: ContextCriteria) -> None:
        """Add a single criterion to the active set."""
        self._criteria.add(criterion)
        logger.debug(f"Added criterion: {criterion.value}")
    
    def remove_criterion(self, criterion: ContextCriteria) -> None:
        """Remove a single criterion from the active set."""
        self._criteria.discard(criterion)
        logger.debug(f"Removed criterion: {criterion.value}")
    
    def is_criterion_active(self, criterion: ContextCriteria) -> bool:
        """Check if a criterion is currently active."""
        return criterion in self._criteria
    
    def set_weight(self, criterion: ContextCriteria, weight: float) -> None:
        """
        Set the weight for a specific criterion.
        
        Args:
            criterion: The criterion to update.
            weight: New weight value (0.0 to 1.0).
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be between 0 and 1, got {weight}")
        self._weights[criterion] = weight
        logger.debug(f"Weight for {criterion.value} set to {weight}")
    
    def set_weights(self, weights: Dict[ContextCriteria, float]) -> None:
        """
        Set multiple criterion weights at once.
        
        Args:
            weights: Dict mapping criteria to weights.
        """
        for criterion, weight in weights.items():
            self.set_weight(criterion, weight)
    
    def reload_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Reload configuration from file.
        
        Args:
            config_path: Path to config file. Uses original path if not provided.
        """
        path = config_path or self._config_path
        self._config = load_context_config(path)
        self._criteria = self._parse_criteria_from_config()
        self._weights = self._parse_weights_from_config()
        self._analysis_config = self._config.get("comprehensive_analysis", {})
        self._scoring_config = self._config.get("scoring", {})
        logger.info(f"Configuration reloaded from {path}")
    
    async def apply_context(
        self,
        outfits: List[Outfit],
        context: UserContext
    ) -> List[Outfit]:
        """
        Apply contextual filtering and scoring to outfits.
        
        Args:
            outfits: List of candidate outfits
            context: User's current context
            
        Returns:
            Filtered and re-scored outfits
        """
        if not outfits:
            return []
        
        # Get weather if location provided
        weather = context.weather
        if not weather and context.location:
            weather = await self.weather_service.get_weather(context.location)
        
        scored_outfits = []
        
        for outfit in outfits:
            # Apply contextual modifiers
            context_score = await self._calculate_context_score(
                outfit, context, weather
            )
            
            # Update outfit scores
            outfit.occasion_match_score = context_score["occasion_score"]
            
            # Recalculate overall score with context
            outfit.overall_score = self._blend_scores(
                outfit.overall_score,
                context_score["total"]
            )
            
            scored_outfits.append((outfit, context_score))
        
        # Sort by new overall score
        scored_outfits.sort(key=lambda x: x[0].overall_score, reverse=True)
        
        return [outfit for outfit, _ in scored_outfits]
    
    async def filter_wardrobe_by_context(
        self,
        wardrobe: List[Garment],
        context: UserContext
    ) -> List[Garment]:
        """
        Pre-filter wardrobe items based on context.
        
        Args:
            wardrobe: Full wardrobe
            context: Current context
            
        Returns:
            Filtered list of suitable items
        """
        filtered = wardrobe.copy()
        
        # Weather filtering
        if context.weather:
            filtered = self._filter_by_weather(filtered, context.weather)
        
        # Occasion filtering
        if context.occasion:
            filtered = self._filter_by_occasion(filtered, context.occasion)
        
        # User preference filtering
        if context.user_id:
            filtered = await self._apply_user_preferences(
                filtered, context.user_id, context
            )
        
        return filtered
    
    async def _calculate_context_score(
        self,
        outfit: Outfit,
        context: UserContext,
        weather: Optional[WeatherContext]
    ) -> Dict[str, float]:
        """Calculate contextual score for an outfit based on active criteria."""
        scores = {
            "weather_score": 1.0,
            "occasion_score": 1.0,
            "morphology_score": 1.0,
            "preference_score": 1.0,
            "activity_score": 1.0,
            "freshness_score": 1.0,
            "fit_score": 1.0,
            "proportion_score": 1.0,
            "color_harmony_score": 1.0
        }
        
        garments = [item.garment for item in outfit.items]
        
        # Map criteria enum to score keys
        criterion_to_score_key = {
            ContextCriteria.WEATHER: "weather_score",
            ContextCriteria.OCCASION: "occasion_score",
            ContextCriteria.MORPHOLOGY: "morphology_score",
            ContextCriteria.PREFERENCE: "preference_score",
            ContextCriteria.ACTIVITY: "activity_score",
            ContextCriteria.FRESHNESS: "freshness_score",
            ContextCriteria.FIT: "fit_score",
            ContextCriteria.PROPORTION: "proportion_score",
            ContextCriteria.COLOR_HARMONY: "color_harmony_score"
        }
        
        # Weather appropriateness
        if self.is_criterion_active(ContextCriteria.WEATHER) and weather:
            scores["weather_score"] = self._score_weather_appropriateness(
                garments, weather
            )
        
        # Occasion appropriateness
        if self.is_criterion_active(ContextCriteria.OCCASION) and context.occasion:
            scores["occasion_score"] = self.occasion_analyzer.score_for_occasion(
                garments, context.occasion
            )
        
        # Morphology suitability (enhanced with body metrics)
        if self.is_criterion_active(ContextCriteria.MORPHOLOGY) and context.body_type:
            if self._style_profile and hasattr(self._style_profile, 'body_metrics'):
                enhanced_score = self.morphology_advisor.score_with_measurements(
                    garments, context.body_type, self._style_profile.body_metrics
                )
                scores["morphology_score"] = enhanced_score.final_score
            else:
                scores["morphology_score"] = self.morphology_advisor.score_for_body_type(
                    garments, context.body_type
                )
        
        # User preference alignment
        if self.is_criterion_active(ContextCriteria.PREFERENCE) and context.user_id:
            scores["preference_score"] = await self.user_history.score_against_preferences(
                garments, context.user_id
            )
        
        # Activity appropriateness
        if self.is_criterion_active(ContextCriteria.ACTIVITY) and (context.activity_level or context.transport_mode):
            activity_score, _ = self.activity_analyzer.score_outfit_for_activity(
                garments, context
            )
            scores["activity_score"] = activity_score
        
        # Wardrobe freshness
        if self.is_criterion_active(ContextCriteria.FRESHNESS):
            scores["freshness_score"] = self.wardrobe_rotation.score_outfit_freshness(
                garments, context
            )
        
        # NEW: Fit prediction based on body metrics
        if self.is_criterion_active(ContextCriteria.FIT) and self._style_profile:
            fit_prediction = self._score_fit(garments)
            scores["fit_score"] = fit_prediction.overall_score
        
        # NEW: Proportion harmony
        if self.is_criterion_active(ContextCriteria.PROPORTION) and self._style_profile:
            proportion_score = self._score_proportion_harmony(garments)
            scores["proportion_score"] = proportion_score.score
        
        # NEW: Color harmony
        if self.is_criterion_active(ContextCriteria.COLOR_HARMONY) and self._style_profile:
            color_score = self._score_color_harmony(garments)
            scores["color_harmony_score"] = color_score.score
        
        # Calculate weighted total based on active criteria only
        active_weights = {}
        for criterion in self._criteria:
            score_key = criterion_to_score_key[criterion]
            active_weights[score_key] = self._weights[criterion]
        
        # Normalize weights if configured
        should_normalize = self._scoring_config.get("normalize_weights", True)
        total_weight = sum(active_weights.values())
        
        if should_normalize and total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in active_weights.items()}
        else:
            normalized_weights = active_weights if active_weights else {}
        
        scores["total"] = sum(
            scores[k] * normalized_weights.get(k, 0) for k in scores if k != "total"
        )
        
        # Track which criteria were used
        scores["active_criteria"] = [c.value for c in self._criteria]
        
        return scores
    
    def _filter_by_weather(
        self,
        items: List[Garment],
        weather: WeatherContext
    ) -> List[Garment]:
        """Filter items based on weather conditions."""
        temp = weather.temperature_celsius
        condition = weather.condition.lower()
        
        # Determine suitable seasons
        if temp < 10:
            suitable_seasons = {Season.WINTER, Season.FALL}
        elif temp < 18:
            suitable_seasons = {Season.FALL, Season.SPRING}
        elif temp < 25:
            suitable_seasons = {Season.SPRING, Season.SUMMER}
        else:
            suitable_seasons = {Season.SUMMER}
        
        # Filter by season suitability
        filtered = []
        for item in items:
            item_seasons = set(item.attributes.season_suitable)
            if not item_seasons or item_seasons & suitable_seasons:
                filtered.append(item)
        
        # Additional filtering for rain
        if "rain" in condition or "wet" in condition:
            # Prefer items not sensitive to water
            # (This is simplified - would check material in production)
            pass
        
        return filtered
    
    def _filter_by_occasion(
        self,
        items: List[Garment],
        occasion: Occasion
    ) -> List[Garment]:
        """Filter items based on occasion."""
        # Map occasion to acceptable formality range
        formality_ranges = {
            Occasion.CASUAL: (FormalityLevel.VERY_CASUAL, FormalityLevel.SMART_CASUAL),
            Occasion.BUSINESS: (FormalityLevel.BUSINESS_CASUAL, FormalityLevel.BUSINESS),
            Occasion.FORMAL: (FormalityLevel.BUSINESS, FormalityLevel.BLACK_TIE),
            Occasion.SPORT: (FormalityLevel.VERY_CASUAL, FormalityLevel.CASUAL),
            Occasion.EVENING: (FormalityLevel.SMART_CASUAL, FormalityLevel.FORMAL),
            Occasion.DATE: (FormalityLevel.SMART_CASUAL, FormalityLevel.FORMAL),
            Occasion.BEACH: (FormalityLevel.VERY_CASUAL, FormalityLevel.CASUAL)
        }
        
        formality_order = [
            FormalityLevel.VERY_CASUAL,
            FormalityLevel.CASUAL,
            FormalityLevel.SMART_CASUAL,
            FormalityLevel.BUSINESS_CASUAL,
            FormalityLevel.BUSINESS,
            FormalityLevel.FORMAL,
            FormalityLevel.BLACK_TIE
        ]
        
        min_form, max_form = formality_ranges.get(
            occasion, 
            (FormalityLevel.CASUAL, FormalityLevel.SMART_CASUAL)
        )
        
        min_idx = formality_order.index(min_form)
        max_idx = formality_order.index(max_form)
        acceptable_formalities = set(formality_order[min_idx:max_idx+1])
        
        return [
            item for item in items
            if item.attributes.formality_level in acceptable_formalities
        ]
    
    async def _apply_user_preferences(
        self,
        items: List[Garment],
        user_id: str,
        context: UserContext
    ) -> List[Garment]:
        """Apply user-specific preferences."""
        # Filter out avoided colors
        if context.avoid_colors:
            avoid_set = {c.lower() for c in context.avoid_colors}
            items = [
                i for i in items
                if i.attributes.color.primary.lower() not in avoid_set
            ]
        
        # Boost preferred colors (don't filter, just note for scoring)
        # This is handled in scoring
        
        return items
    
    def _score_weather_appropriateness(
        self,
        garments: List[Garment],
        weather: WeatherContext
    ) -> float:
        """Score how appropriate outfit is for weather."""
        temp = weather.temperature_celsius
        
        # Check if garments are appropriate for temperature
        scores = []
        for garment in garments:
            seasons = garment.attributes.season_suitable
            
            if temp < 10:  # Cold
                if Season.WINTER in seasons or Season.FALL in seasons:
                    scores.append(1.0)
                elif Season.SPRING in seasons:
                    scores.append(0.6)
                else:
                    scores.append(0.3)
            elif temp < 20:  # Mild
                if Season.SPRING in seasons or Season.FALL in seasons:
                    scores.append(1.0)
                else:
                    scores.append(0.7)
            else:  # Warm
                if Season.SUMMER in seasons or Season.SPRING in seasons:
                    scores.append(1.0)
                elif Season.FALL in seasons:
                    scores.append(0.6)
                else:
                    scores.append(0.4)
        
        return sum(scores) / len(scores) if scores else 0.7
    
    def _blend_scores(self, style_score: float, context_score: float) -> float:
        """Blend style and context scores based on configuration."""
        blend_config = self._scoring_config.get("style_context_blend", {})
        style_weight = blend_config.get("style_weight", 0.6)
        context_weight = blend_config.get("context_weight", 0.4)
        return style_weight * style_score + context_weight * context_score
    
    def get_current_season(self, location: Optional[str] = None) -> Season:
        """Get current season based on date and location."""
        month = datetime.now().month
        
        # Northern hemisphere default
        if month in [12, 1, 2]:
            return Season.WINTER
        elif month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        else:
            return Season.FALL
    
    # ============================================
    # New Enhanced Context Methods
    # ============================================
    
    def analyze_schedule(self, context: UserContext) -> Dict:
        """
        Analyze user's schedule for multi-event days.
        
        Returns analysis including:
        - primary_occasion
        - transition_needed
        - strategy (single_outfit, smart_layers, etc.)
        """
        return self.schedule_analyzer.analyze_schedule(context)
    
    def get_transition_strategy(self, context: UserContext) -> TransitionStrategy:
        """Get recommended strategy for outfit transitions."""
        analysis = self.schedule_analyzer.analyze_schedule(context)
        return analysis.get("strategy", TransitionStrategy.SINGLE_OUTFIT)
    
    def get_transition_recommendations(
        self,
        base_outfit: List[Garment],
        target_occasion: Occasion
    ) -> Dict:
        """Get recommendations for transitioning outfit to new occasion."""
        return self.schedule_analyzer.recommend_transition_pieces(
            base_outfit, target_occasion
        )
    
    def record_outfit_worn(
        self,
        outfit: Outfit,
        worn_date: Optional[datetime] = None
    ) -> None:
        """Record that an outfit was worn."""
        garment_ids = [item.garment.id for item in outfit.items]
        self.wardrobe_rotation.record_wear(garment_ids, worn_date)
    
    def get_forgotten_gems(
        self,
        wardrobe: List[Garment],
        min_days: int = 30
    ) -> List[Garment]:
        """Find underutilized wardrobe items."""
        return self.wardrobe_rotation.get_forgotten_gems(wardrobe, min_days)
    
    def suggest_featured_item(
        self,
        wardrobe: List[Garment],
        context: UserContext
    ) -> Optional[Garment]:
        """Suggest an item to feature in today's outfit."""
        return self.wardrobe_rotation.suggest_featured_item(wardrobe, context)
    
    def get_wardrobe_statistics(
        self,
        wardrobe: List[Garment]
    ) -> Dict:
        """Get usage statistics for the wardrobe."""
        return self.wardrobe_rotation.get_wear_statistics(wardrobe)
    
    def get_activity_recommendations(
        self,
        context: UserContext
    ) -> List[str]:
        """Get activity-based outfit recommendations."""
        return self.activity_analyzer.get_activity_recommendations(context)
    
    def analyze_activity_context(
        self,
        context: UserContext
    ) -> Dict:
        """Get detailed activity analysis."""
        return self.activity_analyzer.analyze_activity_context(context)
    
    def get_comprehensive_context_analysis(
        self,
        context: UserContext,
        wardrobe: Optional[List[Garment]] = None
    ) -> Dict:
        """
        Get a comprehensive analysis of all context factors.
        
        Analysis components are controlled by the 'comprehensive_analysis'
        section in the configuration file.
        
        Returns:
            Dict with context analyses based on configuration
        """
        analysis = {
            "active_criteria": [c.value for c in self._criteria],
            "criteria_weights": {c.value: w for c, w in self._weights.items() if c in self._criteria},
        }
        
        # Include schedule analysis if configured
        if self._analysis_config.get("include_schedule", True):
            analysis["schedule"] = self.analyze_schedule(context)
        
        # Include activity analysis if configured
        if self._analysis_config.get("include_activity", True):
            analysis["activity"] = self.analyze_activity_context(context)
            analysis["recommendations"] = {
                "activity": self.get_activity_recommendations(context)
            }
        
        # Include season if configured
        if self._analysis_config.get("include_season", True):
            analysis["season"] = self.get_current_season(context.location)
        
        # Include wardrobe stats if configured and wardrobe provided
        if wardrobe and self._analysis_config.get("include_wardrobe_stats", True):
            analysis["wardrobe_stats"] = self.get_wardrobe_statistics(wardrobe)
        
        # Include featured item if configured and wardrobe provided
        if wardrobe and self._analysis_config.get("include_featured_item", True):
            featured = self.suggest_featured_item(wardrobe, context)
            if featured:
                analysis["featured_item"] = {
                    "id": featured.id,
                    "name": getattr(featured, 'name', featured.id),
                    "category": featured.attributes.category.value
                }
        
        return analysis
    
    # ============================================
    # Body Measurement Integration Methods
    # ============================================
    
    def set_style_profile(self, style_profile: "StyleProfile") -> None:
        """
        Set the user's style profile for body measurement-based scoring.
        
        Args:
            style_profile: StyleProfile from user_profile module
        """
        self._style_profile = style_profile
        logger.info("Style profile set for body measurement scoring")
        
        # Enable body measurement criteria if profile is set
        if style_profile.body_metrics:
            self.add_criterion(ContextCriteria.FIT)
            self.add_criterion(ContextCriteria.PROPORTION)
        
        if style_profile.skin_analysis or style_profile.hair_analysis:
            self.add_criterion(ContextCriteria.COLOR_HARMONY)
    
    def clear_style_profile(self) -> None:
        """Clear the style profile and disable body measurement criteria."""
        self._style_profile = None
        self.remove_criterion(ContextCriteria.FIT)
        self.remove_criterion(ContextCriteria.PROPORTION)
        self.remove_criterion(ContextCriteria.COLOR_HARMONY)
        logger.info("Style profile cleared")
    
    def _score_fit(self, garments: List[Garment]) -> OutfitFitPrediction:
        """Score outfit fit based on body metrics."""
        if not self._style_profile or not self._style_profile.body_metrics:
            return OutfitFitPrediction(
                overall_score=0.7,
                garment_scores={},
                average_fit_type="unknown",
                fit_notes=["No body metrics available"]
            )
        
        metrics = self._style_profile.body_metrics
        
        return self.fit_predictor.predict_outfit_fit(
            garments=garments,
            user_top_size=getattr(metrics, 'estimated_top_size', None),
            user_bottom_size=getattr(metrics, 'estimated_bottom_size', None),
            user_bmi_category=getattr(metrics, 'bmi_category', None)
        )
    
    def _score_proportion_harmony(self, garments: List[Garment]) -> ProportionScore:
        """Score outfit based on body proportion harmony."""
        if not self._style_profile or not self._style_profile.body_metrics:
            return ProportionScore(
                score=0.7,
                harmony_level="unknown",
                positive_factors=[],
                negative_factors=[],
                styling_tips=[]
            )
        
        metrics = self._style_profile.body_metrics
        
        # Get proportion analysis
        analysis = self.proportion_harmonizer.analyze_proportions(
            torso_ratio=getattr(metrics, 'torso_leg_ratio', None),
            leg_ratio=getattr(metrics, 'leg_proportion', None) if hasattr(metrics, 'leg_proportion') else None,
            frame_size=getattr(metrics, 'frame_size', None),
            shoulder_hip_ratio=getattr(metrics, 'shoulder_hip_ratio', None)
        )
        
        return self.proportion_harmonizer.score_outfit_harmony(garments, analysis)
    
    def _score_color_harmony(self, garments: List[Garment]) -> ColorHarmonyScore:
        """Score outfit based on color harmony with user's coloring."""
        if not self._style_profile:
            return ColorHarmonyScore(
                score=0.7,
                harmony_level="unknown",
                matching_colors=[],
                clashing_colors=[],
                notes=[]
            )
        
        # Build color profile from style profile
        skin = self._style_profile.skin_analysis
        hair = self._style_profile.hair_analysis
        
        if not skin and not hair:
            return ColorHarmonyScore(
                score=0.7,
                harmony_level="unknown",
                matching_colors=[],
                clashing_colors=[],
                notes=["No skin/hair analysis available"]
            )
        
        # Extract values with defaults
        skin_tone = str(getattr(skin, 'skin_tone', 'MEDIUM')) if skin else 'MEDIUM'
        undertone = str(getattr(skin, 'undertone', 'NEUTRAL')) if skin else 'NEUTRAL'
        hair_color = str(getattr(hair, 'hair_color', 'BROWN')) if hair else 'BROWN'
        contrast_level = str(getattr(skin, 'contrast_level', 'MEDIUM')) if skin else 'MEDIUM'
        
        color_profile = self.color_harmony_advisor.create_color_profile(
            skin_tone=skin_tone,
            undertone=undertone,
            hair_color=hair_color,
            contrast_level=contrast_level
        )
        
        return self.color_harmony_advisor.score_outfit_colors(garments, color_profile)
    
    def get_body_measurement_recommendations(
        self,
        garments: List[Garment]
    ) -> Dict[str, Any]:
        """
        Get comprehensive recommendations based on body measurements.
        
        Returns recommendations for:
        - Fit adjustments
        - Proportion styling
        - Color harmony
        
        Args:
            garments: List of garments to evaluate
            
        Returns:
            Dict with recommendations from all body measurement modules
        """
        recommendations = {}
        
        if not self._style_profile:
            return {"error": "No style profile set. Use set_style_profile() first."}
        
        # Fit recommendations
        if self._style_profile.body_metrics:
            fit_result = self._score_fit(garments)
            recommendations["fit"] = {
                "score": fit_result.overall_score,
                "fit_type": fit_result.average_fit_type,
                "notes": fit_result.fit_notes
            }
            
            # Proportion recommendations
            prop_result = self._score_proportion_harmony(garments)
            recommendations["proportions"] = {
                "score": prop_result.score,
                "harmony_level": prop_result.harmony_level,
                "positive": prop_result.positive_factors,
                "negative": prop_result.negative_factors,
                "tips": prop_result.styling_tips
            }
        
        # Color recommendations
        if self._style_profile.skin_analysis or self._style_profile.hair_analysis:
            color_result = self._score_color_harmony(garments)
            recommendations["color_harmony"] = {
                "score": color_result.score,
                "harmony_level": color_result.harmony_level,
                "matching": color_result.matching_colors,
                "clashing": color_result.clashing_colors,
                "notes": color_result.notes
            }
            
            # Add color recommendations
            if self._style_profile.skin_analysis:
                skin = self._style_profile.skin_analysis
                color_profile = self.color_harmony_advisor.create_color_profile(
                    skin_tone=str(getattr(skin, 'skin_tone', 'MEDIUM')),
                    undertone=str(getattr(skin, 'undertone', 'NEUTRAL')),
                    hair_color='BROWN',  # Default if no hair analysis
                    contrast_level=str(getattr(skin, 'contrast_level', 'MEDIUM'))
                )
                color_recs = self.color_harmony_advisor.get_color_recommendations(color_profile)
                recommendations["recommended_colors"] = {
                    "best": color_recs.best_colors,
                    "neutrals": color_recs.neutral_colors,
                    "avoid": color_recs.colors_to_avoid,
                    "tips": color_recs.tips
                }
        
        return recommendations
    
    def get_enhanced_outfit_analysis(
        self,
        outfit: Outfit,
        context: UserContext
    ) -> Dict[str, Any]:
        """
        Get enhanced outfit analysis including body measurements.
        
        Combines:
        - Context scoring (weather, occasion, activity, etc.)
        - Body measurement scoring (fit, proportions, colors)
        - Morphology recommendations
        
        Args:
            outfit: Outfit to analyze
            context: User context
            
        Returns:
            Comprehensive analysis dict
        """
        garments = [item.garment for item in outfit.items]
        
        analysis = {
            "outfit_id": getattr(outfit, 'id', None),
            "overall_score": outfit.overall_score,
        }
        
        # Standard context analysis
        analysis["context"] = {
            "active_criteria": [c.value for c in self._criteria],
            "weights": {c.value: w for c, w in self._weights.items() if c in self._criteria}
        }
        
        # Body measurement analysis if profile is set
        if self._style_profile:
            analysis["body_measurements"] = self.get_body_measurement_recommendations(garments)
            
            # Enhanced morphology if body metrics available
            if self._style_profile.body_metrics and context.body_type:
                enhanced_morph = self.morphology_advisor.get_enhanced_recommendations(
                    context.body_type,
                    self._style_profile.body_metrics
                )
                analysis["morphology"] = enhanced_morph
        
        return analysis

