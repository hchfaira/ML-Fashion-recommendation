# Layer 3: Context Engine
from .context_engine import (
    ContextEngine,
    ContextCriteria,
    DEFAULT_CRITERIA,
    ALL_CRITERIA,
    DEFAULT_WEIGHTS,
    load_context_config,
)
from .weather_service import WeatherService
from .occasion_analyzer import OccasionAnalyzer
from .morphology_advisor import MorphologyAdvisor, EnhancedMorphologyScore
from .user_history import UserHistoryManager
from .schedule_analyzer import ScheduleAnalyzer, TransitionStrategy
from .wardrobe_rotation import WardrobeRotationService
from .activity_analyzer import ActivityAnalyzer, ComfortFactor

# Body measurement integration modules
from .fit_predictor import FitPredictor, FitPrediction, OutfitFitPrediction
from .proportion_harmonizer import ProportionHarmonizer, ProportionAnalysis, ProportionScore
from .color_harmony_advisor import ColorHarmonyAdvisor, ColorProfile, ColorRecommendation, ColorHarmonyScore

__all__ = [
    "ContextEngine",
    "ContextCriteria",
    "DEFAULT_CRITERIA",
    "ALL_CRITERIA",
    "DEFAULT_WEIGHTS",
    "load_context_config",
    "WeatherService",
    "OccasionAnalyzer",
    "MorphologyAdvisor",
    "EnhancedMorphologyScore",
    "UserHistoryManager",
    # New enhanced services
    "ScheduleAnalyzer",
    "TransitionStrategy",
    "WardrobeRotationService",
    "ActivityAnalyzer",
    "ComfortFactor",
    # Body measurement integration
    "FitPredictor",
    "FitPrediction",
    "OutfitFitPrediction",
    "ProportionHarmonizer",
    "ProportionAnalysis",
    "ProportionScore",
    "ColorHarmonyAdvisor",
    "ColorProfile",
    "ColorRecommendation",
    "ColorHarmonyScore",
]
