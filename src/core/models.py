"""Shared data models and schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


# ============== Enums ==============

class GarmentCategory(str, Enum):
    """Categories of garments."""
    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"
    OUTERWEAR = "outerwear"
    SHOES = "shoes"
    ACCESSORY = "accessory"
    BAG = "bag"


class OutfitRole(str, Enum):
    """Role of a garment in an outfit."""
    BASE_LAYER = "base_layer"
    MAIN_PIECE = "main_piece"
    LAYERING_PIECE = "layering_piece"
    STATEMENT_PIECE = "statement_piece"
    SUPPORTING_PIECE = "supporting_piece"
    FOOTWEAR = "footwear"
    ACCESSORY = "accessory"


class Occasion(str, Enum):
    """Occasion types."""
    CASUAL = "casual"
    BUSINESS = "business"
    FORMAL = "formal"
    SPORT = "sport"
    EVENING = "evening"
    BEACH = "beach"
    DATE = "date"
    DAILY_WEAR = "daily_wear"
    WORK = "work"
    WEEKEND = "weekend"
    TRAVEL = "travel"
    EVENT = "event"
    INTERVIEW = "interview"
    WEDDING = "wedding"
    COCKTAIL = "cocktail"
    OUTDOOR = "outdoor"
    GYM = "gym"


class ActivityLevel(str, Enum):
    """Physical activity level for the day."""
    SEDENTARY = "sedentary"       # Office/desk work
    LIGHT = "light"               # Walking, light tasks
    MODERATE = "moderate"         # Active day, some walking
    ACTIVE = "active"             # Sports, outdoor activities
    INTENSIVE = "intensive"       # Gym, heavy exercise
    INTENSE = "intense"           # Alias for intensive


class EnergyLevel(str, Enum):
    """Current energy/mood level."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class TransportMode(str, Enum):
    """How the user will be traveling."""
    WALKING = "walking"
    CYCLING = "cycling"
    DRIVING = "driving"
    PUBLIC_TRANSIT = "public_transit"
    MIXED = "mixed"


class TimeOfDay(str, Enum):
    """Time of day for context."""
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


class BudgetLevel(str, Enum):
    """Budget consideration for recommendations."""
    BUDGET = "budget"
    MODERATE = "moderate"
    PREMIUM = "premium"
    LUXURY = "luxury"


class ComfortPriority(str, Enum):
    """How much to prioritize comfort vs style."""
    COMFORT_FIRST = "comfort_first"
    BALANCED = "balanced"
    STYLE_FIRST = "style_first"


class CultureContext(str, Enum):
    """Cultural context for dress codes."""
    WESTERN = "western"
    MIDDLE_EASTERN = "middle_eastern"
    EAST_ASIAN = "east_asian"
    SOUTH_ASIAN = "south_asian"
    LATIN = "latin"
    AFRICAN = "african"
    NORDIC = "nordic"
    MEDITERRANEAN = "mediterranean"


class Season(str, Enum):
    """Seasons."""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"


class FormalityLevel(str, Enum):
    """Formality levels."""
    VERY_CASUAL = "very_casual"
    CASUAL = "casual"
    SMART_CASUAL = "smart_casual"
    BUSINESS_CASUAL = "business_casual"
    BUSINESS = "business"
    FORMAL = "formal"
    BLACK_TIE = "black_tie"


class NecklineType(str, Enum):
    """Types of necklines."""
    V_NECK = "v_neck"
    CREWNECK = "crewneck"
    TURTLENECK = "turtleneck"
    BOAT_NECK = "boat_neck"
    COLLARED = "collared"
    SCOOP = "scoop"
    SQUARE = "square"
    HALTER = "halter"
    OFF_SHOULDER = "off_shoulder"
    COWL = "cowl"


class LengthType(str, Enum):
    """Length types for garments."""
    CROP = "crop"
    REGULAR = "regular"
    LONGLINE = "longline"
    MIDI = "midi"
    MAXI = "maxi"
    MINI = "mini"
    KNEE_LENGTH = "knee_length"
    ANKLE_LENGTH = "ankle_length"


class WaistRise(str, Enum):
    """Waist rise types for bottoms."""
    LOW_RISE = "low_rise"
    MID_RISE = "mid_rise"
    HIGH_RISE = "high_rise"


class ClosureType(str, Enum):
    """Closure types for garments."""
    ZIPPER = "zipper"
    BUTTONS = "buttons"
    SNAP = "snap"
    WRAP = "wrap"
    PULL_ON = "pull_on"
    HOOK_AND_EYE = "hook_and_eye"
    DRAWSTRING = "drawstring"
    VELCRO = "velcro"


class TransparencyLevel(str, Enum):
    """Transparency levels for fabrics."""
    OPAQUE = "opaque"
    SEMI_SHEER = "semi_sheer"
    SHEER = "sheer"


class ColorTemperature(str, Enum):
    """Color temperature classification."""
    WARM = "warm"
    COOL = "cool"
    NEUTRAL = "neutral"


class ColorDepth(str, Enum):
    """Color depth/lightness classification."""
    LIGHT = "light"
    MEDIUM = "medium"
    DARK = "dark"


class ContrastLevel(str, Enum):
    """Contrast level within garment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FabricWeight(str, Enum):
    """Fabric weight classification."""
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class StretchLevel(str, Enum):
    """Stretch/elasticity level."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StructureType(str, Enum):
    """Garment structure classification."""
    STRUCTURED = "structured"
    SEMI_STRUCTURED = "semi_structured"
    SOFT = "soft"
    FLOWING = "flowing"


class VolumeLevel(str, Enum):
    """Volume/fullness level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TrendAlignment(str, Enum):
    """How trend-forward the garment is."""
    TIMELESS = "timeless"
    TREND_FORWARD = "trend_forward"
    SEASONAL = "seasonal"


class StatementLevel(str, Enum):
    """How much of a statement piece."""
    BASIC = "basic"
    MODERATE = "moderate"
    STATEMENT = "statement"


class TemperatureSuitability(str, Enum):
    """Temperature suitability."""
    HOT = "hot"
    WARM = "warm"
    MILD = "mild"
    COLD = "cold"


# ============== Base Models ==============

class ColorProfile(BaseModel):
    """Enhanced color information for styling compatibility."""
    primary: str = Field(..., description="Primary color")
    secondary: Optional[str] = Field(None, description="Secondary color")
    accent: Optional[str] = Field(None, description="Accent color (legacy)")
    color_palette: List[str] = Field(default_factory=list, description="Full color palette")
    hex_codes: List[str] = Field(default_factory=list, description="Hex color codes")
    color_temperature: Optional[ColorTemperature] = Field(None, description="Warm/cool/neutral")
    color_depth: Optional[ColorDepth] = Field(None, description="Light/medium/dark")
    contrast_level: Optional[ContrastLevel] = Field(None, description="Internal contrast level")


# Backward compatibility alias
ColorInfo = ColorProfile


class MaterialProfile(BaseModel):
    """Enhanced material information for styling."""
    primary: str = Field(..., description="Primary material")
    secondary: Optional[str] = Field(None, description="Secondary material")
    texture: Optional[str] = Field(None, description="Texture description")
    fabric_weight: Optional[FabricWeight] = Field(None, description="Light/medium/heavy")
    stretch: Optional[StretchLevel] = Field(None, description="Stretch level")


# Backward compatibility alias
MaterialInfo = MaterialProfile


class PatternInfo(BaseModel):
    """Pattern information for a garment."""
    type: str = Field(default="solid", description="Pattern type (solid, stripes, checks, floral, dots, animal, geometric, etc.)")
    scale: Optional[str] = Field(None, description="Pattern scale (micro, small, medium, large, oversized)")
    density: Optional[str] = Field(None, description="Pattern density (sparse, moderate, dense)")
    description: Optional[str] = Field(None, description="Pattern description")
    
    # Colors within the pattern (for color callback analysis)
    pattern_colors: List[str] = Field(default_factory=list, description="Colors present in the pattern")


class SleeveType(BaseModel):
    """Sleeve information for tops and dresses."""
    length: str = Field(..., description="short, long, 3/4, sleeveless")
    style: Optional[str] = Field(None, description="raglan, puff, bell, bishop, cap, dolman")


class DetailTags(BaseModel):
    """Specific design elements and embellishments."""
    has_pockets: bool = False
    distressed: bool = False  # Ripped jeans, raw hems, frayed edges
    embellishments: List[str] = Field(default_factory=list)  # sequins, embroidery, ruffles, beading
    transparency: TransparencyLevel = TransparencyLevel.OPAQUE
    lined: bool = False
    reversible: bool = False


class SilhouetteProfile(BaseModel):
    """Silhouette and fit information for styling."""
    fit: Optional[str] = Field(None, description="skinny, slim, regular, relaxed, oversized")
    structure: Optional[StructureType] = Field(None, description="structured, soft, flowing")
    length: Optional[str] = Field(None, description="cropped, regular, long, mini, midi, maxi")
    volume: Optional[VolumeLevel] = Field(None, description="Low/medium/high volume")


class StyleIdentity(BaseModel):
    """Style aesthetic and identity markers."""
    aesthetic_styles: List[str] = Field(default_factory=list, description="minimalist, classic, streetwear, etc.")
    trend_alignment: Optional[TrendAlignment] = Field(None, description="timeless, trend_forward, seasonal")
    statement_level: Optional[StatementLevel] = Field(None, description="basic, moderate, statement")


class StylingCompatibility(BaseModel):
    """Compatibility hints for outfit building."""
    layering_compatibility: List[str] = Field(default_factory=list, description="works_under_jackets, works_over_shirts, etc.")
    matching_bottoms: List[str] = Field(default_factory=list, description="jeans, tailored_trousers, skirts, shorts")
    matching_tops: List[str] = Field(default_factory=list, description="tshirt, shirt, sweater, blouse")
    matching_outerwear: List[str] = Field(default_factory=list, description="blazer, trench, coat, denim_jacket")
    matching_shoes: List[str] = Field(default_factory=list, description="sneakers, loafers, boots, heels")


class OccasionProfile(BaseModel):
    """Occasion and formality information."""
    formality_level: FormalityLevel = FormalityLevel.CASUAL
    occasions: List[str] = Field(default_factory=list, description="daily_wear, work, weekend, evening, travel, event")


class SeasonalityInfo(BaseModel):
    """Seasonality and temperature information."""
    seasons: List[Season] = Field(default_factory=list)
    temperature_suitability: Optional[TemperatureSuitability] = Field(None, description="hot, warm, mild, cold")


class VersatilityInfo(BaseModel):
    """Versatility metrics for wardrobe planning."""
    versatility_score: float = Field(default=0.5, ge=0, le=1, description="How versatile this piece is (0-1)")
    capsule_wardrobe_friendly: bool = Field(default=False, description="Good for minimal wardrobe")


# ============== Garment Models ==============

class GarmentAttributes(BaseModel):
    """Extracted attributes of a garment for outfit compatibility."""
    # Core taxonomy
    category: GarmentCategory
    subcategory: Optional[str] = None
    product_type: Optional[str] = None
    outfit_role: Optional[OutfitRole] = None
    
    # Color profile (enhanced)
    color: ColorProfile
    
    # Material profile (enhanced)
    material: Optional[MaterialProfile] = None
    
    # Pattern
    pattern: PatternInfo = Field(default_factory=lambda: PatternInfo(type="solid"))
    
    # Silhouette profile
    silhouette_profile: Optional[SilhouetteProfile] = None
    
    # Legacy fields for backward compatibility
    silhouette: Optional[str] = None
    fit: Optional[str] = None
    
    # Style identity
    style_identity: Optional[StyleIdentity] = None
    style_tags: List[str] = Field(default_factory=list)  # Legacy, use style_identity.aesthetic_styles
    
    # Styling compatibility hints
    styling_compatibility: Optional[StylingCompatibility] = None
    
    # Occasion profile
    occasion_profile: Optional[OccasionProfile] = None
    formality_level: FormalityLevel = FormalityLevel.CASUAL  # Legacy
    
    # Seasonality
    seasonality: Optional[SeasonalityInfo] = None
    season_suitable: List[Season] = Field(default_factory=list)  # Legacy
    
    # Versatility metrics
    versatility: Optional[VersatilityInfo] = None
    
    # Extended garment details
    neckline: Optional[NecklineType] = None
    sleeves: Optional[SleeveType] = None
    length_type: Optional[LengthType] = None
    waist_rise: Optional[WaistRise] = None
    closure: Optional[ClosureType] = None
    details: DetailTags = Field(default_factory=DetailTags)
    
    # Extraction confidence
    confidence_score: float = Field(default=0.0, ge=0, le=1, description="Model confidence in extraction")


class Garment(BaseModel):
    """Complete garment model."""
    id: str
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    attributes: GarmentAttributes
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============== Outfit Models ==============

class OutfitItem(BaseModel):
    """An item in an outfit."""
    garment: Garment
    role: str = Field(..., description="Role in outfit (e.g., 'main_top', 'layering')")


class Outfit(BaseModel):
    """A complete outfit recommendation."""
    id: str
    items: List[OutfitItem]
    compatibility_score: float = Field(..., ge=0, le=1)
    style_coherence_score: float = Field(..., ge=0, le=1)
    occasion_match_score: float = Field(..., ge=0, le=1)
    overall_score: float = Field(..., ge=0, le=1)
    
    # Seven-point rule analysis
    seven_point_total: Optional[int] = Field(None, description="Total points (target: 7-10)")
    seven_point_harmonious: Optional[bool] = Field(None, description="Whether outfit is in optimal range")
    
    explanation: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============== Context Models ==============

class WeatherContext(BaseModel):
    """Weather context information."""
    temperature_celsius: float
    condition: str  # sunny, cloudy, rainy, snowy
    humidity: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    uv_index: Optional[int] = Field(None, ge=0, le=11, description="UV index 0-11")
    precipitation_chance: Optional[float] = Field(None, ge=0, le=100, description="% chance of rain")
    feels_like_celsius: Optional[float] = Field(None, description="Perceived temperature")


class ScheduleEvent(BaseModel):
    """A scheduled event for the day."""
    time: str = Field(..., description="Time of event (HH:MM)")
    name: str = Field(..., description="Event name")
    occasion: Optional[Occasion] = None
    duration_hours: float = Field(default=1.0, ge=0)
    indoor: bool = Field(default=True)
    dress_code: Optional[str] = None


class UserContext(BaseModel):
    """User context for personalization."""
    user_id: Optional[str] = None
    location: Optional[str] = None
    occasion: Optional[Occasion] = None
    weather: Optional[WeatherContext] = None
    body_type: Optional[str] = None
    style_preferences: List[str] = Field(default_factory=list)
    color_preferences: List[str] = Field(default_factory=list)
    avoid_colors: List[str] = Field(default_factory=list)
    
    # Personal coloring (for personalized color scoring)
    skin_undertone: Optional[str] = None  # warm, cool, neutral
    hair_color: Optional[str] = None  # for contrast analysis
    eye_color: Optional[str] = None
    color_season: Optional[str] = None  # spring, summer, autumn, winter
    contrast_type: Optional[str] = None  # high, medium, low (skin vs hair)
    
    # Body shape (for volume balance scoring)
    body_shape: Optional[str] = None  # hourglass, pear, apple, rectangle, inverted_triangle
    
    # Style aesthetic preference
    style_aesthetic: Optional[str] = None  # minimalist, maximalist, classic, bohemian, etc.
    
    # ============== NEW ENHANCED ATTRIBUTES ==============
    
    # Time context
    date: Optional[str] = Field(None, description="Date (YYYY-MM-DD)")
    time_of_day: Optional[str] = Field(None, description="morning, afternoon, evening, night")
    day_of_week: Optional[str] = Field(None, description="monday, tuesday, etc.")
    
    # Activity & Comfort
    activity_level: Optional[ActivityLevel] = Field(None, description="Physical activity level")
    comfort_priority: Optional[str] = Field(None, description="comfort_first, balanced, style_first")
    will_be_walking: bool = Field(default=False, description="If lots of walking expected")
    will_be_sitting: bool = Field(default=False, description="If sitting for long periods")
    needs_pockets: bool = Field(default=False, description="If pockets are needed")
    
    # Schedule & Multi-event
    schedule: List[ScheduleEvent] = Field(default_factory=list, description="Day's schedule")
    primary_event: Optional[str] = Field(None, description="Main event of the day")
    secondary_occasions: List[str] = Field(default_factory=list, description="Other activities planned")
    needs_transition_outfit: bool = Field(default=False, description="Needs to work for multiple occasions")
    
    # Social & Cultural
    meeting_new_people: bool = Field(default=False, description="First impressions matter")
    professional_setting: bool = Field(default=False, description="Work/professional context")
    conservative_environment: bool = Field(default=False, description="Need to dress conservatively")
    culture_context: Optional[str] = Field(None, description="Cultural context for dress codes")
    
    # Practical constraints
    budget_level: Optional[str] = Field(None, description="budget, moderate, premium, luxury")
    carrying_bag: bool = Field(default=True, description="Will have a bag")
    needs_outerwear: bool = Field(default=False, description="Will need jacket/coat")
    traveling_between_locations: bool = Field(default=False, description="Moving between places")
    public_transport: bool = Field(default=False, description="Using public transport")
    driving: bool = Field(default=False, description="Will be driving")
    
    # Mood & Intention
    mood: Optional[str] = Field(None, description="Current mood: confident, relaxed, energetic, etc.")
    impression_goal: Optional[str] = Field(None, description="What impression to make: professional, approachable, creative, etc.")
    special_requirements: List[str] = Field(default_factory=list, description="Any specific needs")
    
    # Photography/Events
    will_be_photographed: bool = Field(default=False, description="Photos expected")
    avoid_patterns_for_camera: bool = Field(default=False, description="Avoid moiré patterns for video")
    
    # Health & Comfort
    temperature_sensitivity: Optional[str] = Field(None, description="runs_hot, runs_cold, neutral")
    mobility_needs: Optional[str] = Field(None, description="Any mobility considerations")
    
    # Avoid items
    avoid_materials: List[str] = Field(default_factory=list, description="Materials to avoid (allergies, etc.)")
    avoid_styles: List[str] = Field(default_factory=list, description="Styles to avoid")
    
    # Recent wear tracking (don't repeat)
    recently_worn_ids: List[str] = Field(default_factory=list, description="IDs of recently worn items")
    recently_worn_items: List[str] = Field(default_factory=list, description="Alias for recently_worn_ids")
    days_since_last_wear: int = Field(default=3, description="Min days before repeating items")
    
    # Transport mode (enum-based)
    transport_mode: Optional[TransportMode] = Field(None, description="How user will travel")
    
    # Duration
    duration_hours: Optional[float] = Field(None, description="Expected duration of outing in hours")
    
    # Current energy level
    current_energy: Optional[str] = Field(None, description="low, normal, high")
    
    # Items to feature (for wardrobe rotation)
    items_to_feature: List[str] = Field(default_factory=list, description="IDs of items to highlight")


# ============== API Models ==============

class RecommendationRequest(BaseModel):
    """Request for outfit recommendations."""
    wardrobe_items: List[Garment] = Field(default_factory=list)
    base_item: Optional[Garment] = None  # Item to build outfit around
    context: UserContext = Field(default_factory=UserContext)
    num_recommendations: int = Field(default=3, ge=1, le=10)


class RecommendationResponse(BaseModel):
    """Response with outfit recommendations."""
    recommendations: List[Outfit]
    processing_time_ms: float
    context_applied: Dict[str, Any]
