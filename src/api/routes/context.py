"""
Context API Routes
Layer 3 context-aware features: weather, occasion, morphology, user history.
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, date

from src.core.models import Garment, Occasion, UserContext, WeatherContext
from src.layer3_context import (
    ContextEngine,
    WeatherService,
    OccasionAnalyzer,
    MorphologyAdvisor,
    UserHistoryManager,
    ScheduleAnalyzer,
    WardrobeRotationService,
    ActivityAnalyzer,
)
from src.core import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Lazy-loaded services
_context_engine = None
_weather_service = None
_occasion_analyzer = None
_morphology_advisor = None
_user_history_manager = None
_schedule_analyzer = None
_wardrobe_rotation = None
_activity_analyzer = None


def get_context_engine() -> ContextEngine:
    global _context_engine
    if _context_engine is None:
        _context_engine = ContextEngine()
    return _context_engine


def get_weather_service() -> WeatherService:
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service


def get_occasion_analyzer() -> OccasionAnalyzer:
    global _occasion_analyzer
    if _occasion_analyzer is None:
        _occasion_analyzer = OccasionAnalyzer()
    return _occasion_analyzer


def get_morphology_advisor() -> MorphologyAdvisor:
    global _morphology_advisor
    if _morphology_advisor is None:
        _morphology_advisor = MorphologyAdvisor()
    return _morphology_advisor


def get_user_history_manager() -> UserHistoryManager:
    global _user_history_manager
    if _user_history_manager is None:
        _user_history_manager = UserHistoryManager()
    return _user_history_manager


def get_schedule_analyzer() -> ScheduleAnalyzer:
    global _schedule_analyzer
    if _schedule_analyzer is None:
        _schedule_analyzer = ScheduleAnalyzer()
    return _schedule_analyzer


def get_wardrobe_rotation() -> WardrobeRotationService:
    global _wardrobe_rotation
    if _wardrobe_rotation is None:
        _wardrobe_rotation = WardrobeRotationService()
    return _wardrobe_rotation


def get_activity_analyzer() -> ActivityAnalyzer:
    global _activity_analyzer
    if _activity_analyzer is None:
        _activity_analyzer = ActivityAnalyzer()
    return _activity_analyzer


# ============== Request/Response Models ==============

class WeatherRequest(BaseModel):
    """Weather context request."""
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    date: Optional[date] = None


class ContextFilterRequest(BaseModel):
    """Request to filter wardrobe by context."""
    wardrobe: List[Garment]
    occasion: Optional[str] = None
    weather: Optional[WeatherContext] = None
    user_preferences: Optional[Dict[str, Any]] = None


class MorphologyRequest(BaseModel):
    """User morphology information."""
    body_shape: str  # "hourglass", "pear", "apple", "rectangle", "inverted_triangle"
    height_cm: Optional[int] = None
    preferred_fit: Optional[str] = None  # "fitted", "relaxed", "oversized"


class ScheduleEvent(BaseModel):
    """Calendar event for schedule analysis."""
    title: str
    start_time: datetime
    end_time: Optional[datetime] = None
    location: Optional[str] = None
    dress_code: Optional[str] = None
    event_type: Optional[str] = None  # "meeting", "lunch", "presentation", etc.


class ScheduleRequest(BaseModel):
    """Request for schedule-based outfit planning."""
    events: List[ScheduleEvent]
    wardrobe: List[Garment]
    date: date


class RotationRequest(BaseModel):
    """Request for wardrobe rotation analysis."""
    user_id: str
    wardrobe: List[Garment]
    days_to_consider: int = Field(default=30, ge=7, le=365)


class ActivityRequest(BaseModel):
    """Request for activity-based outfit filtering."""
    activities: List[str]  # ["walking", "sitting_long", "outdoor", etc.]
    wardrobe: List[Garment]
    comfort_priority: float = Field(default=0.5, ge=0, le=1)


class UserHistoryEntry(BaseModel):
    """Entry for user outfit history."""
    user_id: str
    outfit_garment_ids: List[str]
    occasion: Optional[str] = None
    date: datetime
    user_rating: Optional[int] = Field(default=None, ge=1, le=5)
    weather_temp: Optional[float] = None


# ============== Weather Endpoints ==============

@router.get("/weather")
async def get_weather(
    location: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
):
    """
    Get current weather for outfit recommendations.
    
    Provide either location name OR latitude/longitude.
    """
    if not location and (latitude is None or longitude is None):
        raise HTTPException(
            status_code=400,
            detail="Provide either location or latitude/longitude"
        )
    
    try:
        service = get_weather_service()
        
        if location:
            weather = await service.get_weather_by_location(location)
        else:
            weather = await service.get_weather_by_coords(latitude, longitude)
        
        return {
            "temperature_celsius": weather.temperature_celsius,
            "feels_like": weather.feels_like,
            "humidity": weather.humidity,
            "wind_speed": weather.wind_speed,
            "conditions": weather.conditions,
            "precipitation_chance": weather.precipitation_chance,
            "uv_index": weather.uv_index,
            "clothing_recommendations": weather.clothing_recommendations
        }
        
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weather/outfit-filter")
async def filter_by_weather(wardrobe: List[Garment], weather: WeatherContext):
    """
    Filter wardrobe items suitable for given weather.
    """
    try:
        service = get_weather_service()
        suitable_items = service.filter_by_weather(wardrobe, weather)
        
        return {
            "suitable_items": [g.id for g in suitable_items],
            "filtered_count": len(wardrobe) - len(suitable_items),
            "weather_summary": f"{weather.temperature_celsius}°C, {weather.conditions}"
        }
        
    except Exception as e:
        logger.error(f"Weather filtering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Occasion Endpoints ==============

@router.get("/occasions")
async def list_occasions():
    """List all supported occasions with formality levels."""
    analyzer = get_occasion_analyzer()
    return {
        "occasions": [
            {
                "name": occ.value,
                "formality_level": analyzer.get_formality_level(occ),
                "dress_code_hints": analyzer.get_dress_code_hints(occ)
            }
            for occ in Occasion
        ]
    }


@router.post("/occasions/analyze")
async def analyze_occasion(
    occasion: str,
    wardrobe: List[Garment]
):
    """
    Analyze which wardrobe items suit a specific occasion.
    
    Returns items ranked by appropriateness.
    """
    try:
        occ = Occasion(occasion)
        analyzer = get_occasion_analyzer()
        
        ranked_items = analyzer.rank_items_for_occasion(wardrobe, occ)
        
        return {
            "occasion": occasion,
            "suitable_items": [
                {"garment_id": g.id, "score": score, "reason": reason}
                for g, score, reason in ranked_items
            ],
            "dress_code": analyzer.get_dress_code_hints(occ)
        }
        
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid occasion. Valid: {[o.value for o in Occasion]}"
        )
    except Exception as e:
        logger.error(f"Occasion analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Morphology Endpoints ==============

@router.post("/morphology/advice")
async def get_morphology_advice(request: MorphologyRequest):
    """
    Get styling advice based on body morphology.
    
    Returns recommendations for:
    - Flattering silhouettes
    - Colors and patterns
    - What to accentuate/minimize
    """
    try:
        advisor = get_morphology_advisor()
        
        advice = advisor.get_advice(
            body_shape=request.body_shape,
            height_cm=request.height_cm,
            preferred_fit=request.preferred_fit
        )
        
        return {
            "body_shape": request.body_shape,
            "recommendations": advice.recommendations,
            "flattering_silhouettes": advice.flattering_silhouettes,
            "avoid": advice.avoid,
            "accentuate": advice.accentuate,
            "minimize": advice.minimize,
            "best_proportions": advice.best_proportions
        }
        
    except Exception as e:
        logger.error(f"Morphology advice failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/morphology/filter")
async def filter_by_morphology(
    wardrobe: List[Garment],
    body_shape: str,
    strictness: float = 0.5
):
    """
    Filter wardrobe items flattering for body shape.
    
    Strictness: 0.0 (lenient) to 1.0 (strict)
    """
    try:
        advisor = get_morphology_advisor()
        
        suitable = advisor.filter_flattering_items(
            wardrobe, body_shape, strictness
        )
        
        return {
            "suitable_items": [
                {"garment_id": g.id, "flattering_score": score}
                for g, score in suitable
            ],
            "body_shape": body_shape
        }
        
    except Exception as e:
        logger.error(f"Morphology filtering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Context Engine Endpoints ==============

@router.post("/filter")
async def filter_wardrobe_by_context(request: ContextFilterRequest):
    """
    Filter wardrobe by comprehensive context.
    
    Combines weather, occasion, user preferences, and more.
    """
    try:
        engine = get_context_engine()
        
        context = UserContext(
            occasion=Occasion(request.occasion) if request.occasion else None,
            weather=request.weather
        )
        
        filtered = await engine.filter_wardrobe_by_context(
            request.wardrobe, context
        )
        
        return {
            "filtered_items": [g.id for g in filtered],
            "original_count": len(request.wardrobe),
            "filtered_count": len(filtered),
            "context_applied": {
                "occasion": request.occasion,
                "weather": request.weather.model_dump() if request.weather else None
            }
        }
        
    except Exception as e:
        logger.error(f"Context filtering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Schedule Analysis Endpoints ==============

@router.post("/schedule/analyze")
async def analyze_schedule(request: ScheduleRequest):
    """
    Analyze daily schedule for outfit planning.
    
    Identifies transitions needed and suggests versatile pieces.
    """
    try:
        analyzer = get_schedule_analyzer()
        
        analysis = analyzer.analyze_day(request.events)
        
        return {
            "date": request.date.isoformat(),
            "events_count": len(request.events),
            "formality_range": analysis.formality_range,
            "transitions_needed": analysis.transitions_needed,
            "versatile_pieces_recommended": analysis.versatile_recommendations,
            "outfit_changes_suggested": analysis.outfit_changes,
            "strategy": analysis.strategy.value if hasattr(analysis.strategy, 'value') else analysis.strategy
        }
        
    except Exception as e:
        logger.error(f"Schedule analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Wardrobe Rotation Endpoints ==============

@router.post("/rotation/analyze")
async def analyze_wardrobe_rotation(request: RotationRequest):
    """
    Analyze wardrobe usage and rotation patterns.
    
    Identifies:
    - Overused items
    - Neglected pieces
    - Items to wear next
    """
    try:
        service = get_wardrobe_rotation()
        
        analysis = service.analyze_rotation(
            user_id=request.user_id,
            wardrobe=request.wardrobe,
            days=request.days_to_consider
        )
        
        return {
            "overused_items": [g.id for g in analysis.overused],
            "neglected_items": [g.id for g in analysis.neglected],
            "wear_next_suggestions": [g.id for g in analysis.suggestions],
            "rotation_score": analysis.rotation_score,
            "wardrobe_utilization": analysis.utilization_percent
        }
        
    except Exception as e:
        logger.error(f"Rotation analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rotation/record-wear")
async def record_item_worn(user_id: str, garment_id: str, date: Optional[date] = None):
    """Record that an item was worn."""
    try:
        service = get_wardrobe_rotation()
        service.record_wear(user_id, garment_id, date or datetime.now().date())
        
        return {"status": "recorded", "garment_id": garment_id}
        
    except Exception as e:
        logger.error(f"Recording wear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Activity Analysis Endpoints ==============

@router.post("/activity/filter")
async def filter_by_activity(request: ActivityRequest):
    """
    Filter wardrobe by planned activities.
    
    Activities: walking, sitting_long, outdoor, indoor, 
    formal_standing, physical_activity, etc.
    """
    try:
        analyzer = get_activity_analyzer()
        
        suitable = analyzer.filter_by_activities(
            request.wardrobe,
            request.activities,
            comfort_priority=request.comfort_priority
        )
        
        return {
            "suitable_items": [
                {"garment_id": g.id, "comfort_score": score}
                for g, score in suitable
            ],
            "activities_considered": request.activities
        }
        
    except Exception as e:
        logger.error(f"Activity filtering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activity/comfort-factors")
async def list_comfort_factors():
    """List all comfort factors considered for activities."""
    return {
        "comfort_factors": [
            "breathability",
            "stretch",
            "warmth",
            "water_resistance",
            "ease_of_movement",
            "formality_comfort"
        ],
        "activity_types": [
            "walking", "sitting_long", "outdoor", "indoor",
            "formal_standing", "physical_activity", "commuting"
        ]
    }


# ============== User History Endpoints ==============

@router.post("/history/record")
async def record_outfit_history(entry: UserHistoryEntry):
    """Record an outfit in user history."""
    try:
        manager = get_user_history_manager()
        
        manager.record_outfit(
            user_id=entry.user_id,
            garment_ids=entry.outfit_garment_ids,
            occasion=Occasion(entry.occasion) if entry.occasion else None,
            date=entry.date,
            rating=entry.user_rating,
            weather_temp=entry.weather_temp
        )
        
        return {"status": "recorded"}
        
    except Exception as e:
        logger.error(f"Recording history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{user_id}")
async def get_user_history(
    user_id: str,
    limit: int = 20,
    occasion: Optional[str] = None
):
    """Get user's outfit history."""
    try:
        manager = get_user_history_manager()
        
        occ = Occasion(occasion) if occasion else None
        history = manager.get_history(user_id, limit=limit, occasion=occ)
        
        return {
            "user_id": user_id,
            "outfits": history,
            "count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Fetching history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{user_id}/favorites")
async def get_favorite_outfits(user_id: str, limit: int = 10):
    """Get user's top-rated outfits."""
    try:
        manager = get_user_history_manager()
        favorites = manager.get_favorites(user_id, limit=limit)
        
        return {
            "user_id": user_id,
            "favorites": favorites
        }
        
    except Exception as e:
        logger.error(f"Fetching favorites failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{user_id}/patterns")
async def analyze_user_patterns(user_id: str):
    """Analyze user's style patterns and preferences."""
    try:
        manager = get_user_history_manager()
        patterns = manager.analyze_patterns(user_id)
        
        return {
            "user_id": user_id,
            "preferred_colors": patterns.preferred_colors,
            "preferred_styles": patterns.preferred_styles,
            "occasion_frequency": patterns.occasion_frequency,
            "average_rating": patterns.average_rating,
            "insights": patterns.insights
        }
        
    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
