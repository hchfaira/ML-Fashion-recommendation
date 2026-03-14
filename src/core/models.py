"""Shared data models and schemas."""
from pydantic import BaseModel, Field, computed_field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime, date, timezone


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


class WearContext(str, Enum):
    """The context/reason why a garment or outfit was worn."""
    DAILY = "daily"
    WORK = "work"
    SPORT = "sport"
    FORMAL = "formal"
    TRAVEL = "travel"
    SPECIAL_EVENT = "special_event"
    OTHER = "other"


class GarmentCondition(str, Enum):
    """Physical condition of a garment."""
    NEW = "new"
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    WORN = "worn"
    NEEDS_REPAIR = "needs_repair"


class UnusedReason(str, Enum):
    """Why a garment has not been worn."""
    FORGOTTEN = "forgotten"
    OUT_OF_SEASON = "out_of_season"
    POOR_FIT = "poor_fit"
    NO_MATCHING_ITEMS = "no_matching_items"
    STYLE_CHANGE = "style_change"
    UNKNOWN = "unknown"


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


# ============== History Models ==============

class WearRecord(BaseModel):
    """A single wear event for a garment or outfit."""
    worn_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    occasion: Optional[Occasion] = None
    wear_context: Optional[WearContext] = None
    weather_condition: Optional[str] = Field(None, description="sunny, rainy, cold, etc.")
    temperature_celsius: Optional[float] = None
    location: Optional[str] = None
    outfit_id: Optional[str] = Field(
        None,
        description="ID of the outfit this garment was part of when worn. "
                    "Use this to look up full outfit context — do NOT duplicate outfit data here.",
    )
    user_rating: Optional[int] = Field(None, ge=1, le=5, description="User satisfaction 1-5")
    notes: Optional[str] = None


class GarmentHistory(BaseModel):
    """Full wear and lifecycle history for a garment.

    Design principle: this is the **single source of truth** for a garment's
    wear data.  Each WearRecord optionally carries the ``outfit_id`` it
    belongs to so you can reconstruct "which outfits contained this garment"
    without storing any outfit data here.
    """
    acquired_at: Optional[datetime] = Field(None, description="When the garment was added to the wardrobe")
    acquisition_source: Optional[str] = Field(None, description="store, online, gift, thrift, etc.")
    purchase_price: Optional[float] = Field(None, ge=0, description="Original purchase price")
    condition: GarmentCondition = GarmentCondition.GOOD
    wear_records: List[WearRecord] = Field(default_factory=list, description="Chronological list of wear events")
    times_washed: int = Field(default=0, ge=0)
    last_cleaned_at: Optional[datetime] = None
    is_retired: bool = Field(default=False, description="Marked as removed from active wardrobe")
    retired_at: Optional[datetime] = None
    retirement_reason: Optional[str] = None

    @computed_field  # type: ignore[misc]
    @property
    def total_wears(self) -> int:
        return len(self.wear_records)

    @computed_field  # type: ignore[misc]
    @property
    def last_worn_at(self) -> Optional[datetime]:
        if not self.wear_records:
            return None
        return max(r.worn_at for r in self.wear_records)

    @computed_field  # type: ignore[misc]
    @property
    def days_since_last_worn(self) -> Optional[int]:
        if not self.wear_records:
            return None
        now = datetime.now(timezone.utc)
        last = self.last_worn_at
        if last and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (now - last).days if last else None

    @computed_field  # type: ignore[misc]
    @property
    def cost_per_wear(self) -> Optional[float]:
        if self.purchase_price is None or self.total_wears == 0:
            return None
        return round(self.purchase_price / self.total_wears, 2)

    @computed_field  # type: ignore[misc]
    @property
    def average_rating(self) -> Optional[float]:
        rated = [r.user_rating for r in self.wear_records if r.user_rating is not None]
        if not rated:
            return None
        return round(sum(rated) / len(rated), 1)

    def outfit_ids_worn_in(self) -> List[str]:
        """Return the distinct outfit IDs this garment has appeared in."""
        return list({r.outfit_id for r in self.wear_records if r.outfit_id})


class OutfitCreationRecord(BaseModel):
    """Metadata about when and how an outfit was created."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = Field(None, description="user, ai_suggestion, stylist")
    occasion_intent: Optional[Occasion] = None
    season_intent: Optional[Season] = None
    notes: Optional[str] = None


class OutfitHistory(BaseModel):
    """Full wear and lifecycle history for an outfit.

    Design principle: outfit wear records are the **authoritative** source
    for outfit-level metrics.  Garment-level wear data lives in each
    ``Garment.history`` and cross-references this outfit via
    ``WearRecord.outfit_id``.  There is NO duplication of garment data here.
    """
    creation: OutfitCreationRecord = Field(default_factory=OutfitCreationRecord)
    wear_records: List[WearRecord] = Field(
        default_factory=list,
        description="Each time this outfit was worn as a complete look",
    )
    is_favorite: bool = False
    is_archived: bool = Field(default=False, description="Outfit no longer active")
    archived_at: Optional[datetime] = None

    @computed_field  # type: ignore[misc]
    @property
    def total_wears(self) -> int:
        return len(self.wear_records)

    @computed_field  # type: ignore[misc]
    @property
    def last_worn_at(self) -> Optional[datetime]:
        if not self.wear_records:
            return None
        return max(r.worn_at for r in self.wear_records)

    @computed_field  # type: ignore[misc]
    @property
    def days_since_last_worn(self) -> Optional[int]:
        if not self.wear_records:
            return None
        now = datetime.now(timezone.utc)
        last = self.last_worn_at
        if last and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (now - last).days if last else None

    @computed_field  # type: ignore[misc]
    @property
    def average_rating(self) -> Optional[float]:
        rated = [r.user_rating for r in self.wear_records if r.user_rating is not None]
        if not rated:
            return None
        return round(sum(rated) / len(rated), 1)


class UnusedGarmentAlert(BaseModel):
    """Alert for a garment that has not been worn for a long time."""
    garment_id: str
    garment_description: str
    garment_attributes: Optional["GarmentAttributes"] = None
    days_since_last_worn: Optional[int] = Field(None, description="None means never worn")
    total_wears: int = 0
    acquired_at: Optional[datetime] = None
    days_owned: Optional[int] = None
    suggested_reason: UnusedReason = UnusedReason.UNKNOWN
    action_suggestion: str = Field(
        default="Consider wearing it or donating it.",
        description="Suggested user action: wear, donate, sell, store seasonally"
    )
    replacement_candidates: List[str] = Field(
        default_factory=list,
        description="Other wardrobe items that serve the same purpose"
    )


def scan_unused_garments(
    wardrobe: List["Garment"],
    threshold_days: int = 90,
) -> List[UnusedGarmentAlert]:
    """Scan a wardrobe and return alerts for garments not worn recently.

    This function derives all data purely from ``Garment.history`` —
    there is no need to iterate over outfit records separately because
    every wear event (outfit or standalone) is already mirrored into the
    garment's ``history.wear_records``.

    Args:
        wardrobe:        Full list of active garments.
        threshold_days:  Garments not worn within this many days are flagged.
                         Garments never worn are always included.

    Returns:
        Sorted list of ``UnusedGarmentAlert`` (longest unused first).
    """
    now = datetime.now(timezone.utc)
    alerts: List[UnusedGarmentAlert] = []

    for garment in wardrobe:
        if garment.history.is_retired:
            continue  # already removed, skip

        days = garment.history.days_since_last_worn  # None = never worn
        never_worn = days is None

        if not never_worn and days < threshold_days:
            continue  # worn recently — no alert

        # Determine likely reason
        if never_worn:
            reason = UnusedReason.FORGOTTEN
        elif garment.history.total_wears < 3:
            reason = UnusedReason.UNKNOWN
        else:
            # Heuristic: if all seasons in season_suitable are off-season, flag it
            current_month = now.month
            summer_months = {6, 7, 8}
            winter_months = {12, 1, 2}
            seasons = garment.attributes.season_suitable or []
            if seasons:
                if current_month in winter_months and Season.WINTER not in seasons:
                    reason = UnusedReason.OUT_OF_SEASON
                elif current_month in summer_months and Season.SUMMER not in seasons:
                    reason = UnusedReason.OUT_OF_SEASON
                else:
                    reason = UnusedReason.FORGOTTEN
            else:
                reason = UnusedReason.FORGOTTEN

        # days_owned
        days_owned: Optional[int] = None
        if garment.history.acquired_at:
            acq = garment.history.acquired_at
            if acq.tzinfo is None:
                acq = acq.replace(tzinfo=timezone.utc)
            days_owned = (now - acq).days

        action = (
            "You have never worn this item — consider wearing it soon or donating it."
            if never_worn
            else f"Not worn for {days} days — wear it or consider removing it from your wardrobe."
        )

        desc = (
            f"{garment.attributes.color.primary} "
            f"{garment.attributes.category.value}"
        )

        alerts.append(
            UnusedGarmentAlert(
                garment_id=garment.id,
                garment_description=desc,
                garment_attributes=garment.attributes,
                days_since_last_worn=days,
                total_wears=garment.history.total_wears,
                acquired_at=garment.history.acquired_at,
                days_owned=days_owned,
                suggested_reason=reason,
                action_suggestion=action,
            )
        )

    # Sort: never-worn first, then longest unused
    alerts.sort(key=lambda a: (a.days_since_last_worn is not None, -(a.days_since_last_worn or 0)))
    return alerts


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
    """Complete garment model with full lifecycle history."""
    id: str
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    attributes: GarmentAttributes
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # ---- Lifecycle history ----
    history: GarmentHistory = Field(default_factory=GarmentHistory)


# ============== Outfit Models ==============

class OutfitItem(BaseModel):
    """A reference to a garment within an outfit.

    Design principle
    ----------------
    ``garment_id`` is the authoritative foreign-key reference to
    ``Garment.id``.  The full ``Garment`` object can be optionally cached
    in ``garment`` at query time (e.g. after resolving from the wardrobe
    index) but is **never** persisted — storing full garment objects would
    duplicate the garment's data (and its history) across every outfit that
    contains it.

    Backward compatibility
    ----------------------
    Existing code that passes ``garment=<Garment>`` still works: the
    ``garment_id`` is automatically derived from ``garment.id`` via the
    model validator, so no call-site changes are required.
    """
    garment_id: str = Field(default="", description="Foreign-key reference to Garment.id")
    role: str = Field(
        ...,
        description=(
            "Styling role in this outfit: "
            "base_layer, main_top, bottom, outerwear, shoes, accessory, "
            "statement_piece, layering_piece, etc."
        ),
    )
    # Optional: resolved garment object (not persisted, only populated at query time)
    garment: Optional["Garment"] = Field(
        default=None,
        exclude=True,  # never serialised — always resolve from wardrobe index
        description="Resolved Garment object (transient, not persisted).",
    )

    from pydantic import model_validator

    @model_validator(mode="before")
    @classmethod
    def _derive_garment_id(cls, values: Any) -> Any:
        """Auto-populate garment_id from the garment object when provided."""
        if isinstance(values, dict):
            if not values.get("garment_id") and values.get("garment"):
                g = values["garment"]
                values["garment_id"] = g.id if hasattr(g, "id") else str(g)
        return values


class Outfit(BaseModel):
    """A saved outfit composed of garment references + full lifecycle history.

    Relationship model
    ------------------
    * ``items``  — list of (garment_id, role) references.  NO Garment objects
      are embedded here.  Resolve garments by looking up each id in the
      wardrobe at query time.
    * ``history`` — authoritative source for this outfit's wear timeline.
      Each ``WearRecord`` inside ``history.wear_records`` carries the same
      ``outfit_id`` (= ``self.id``) so garments can cross-reference back.

    How garment wear is tracked
    ---------------------------
    When the user wears this outfit, call ``record_wear()``:
      1. A ``WearRecord`` is appended to ``self.history.wear_records``.
      2. For every garment in ``self.items``, a mirrored ``WearRecord``
         (with ``outfit_id = self.id``) is appended to
         ``garment.history.wear_records``.
    This way ``Garment.history`` is always the single source of truth for
    per-garment statistics while ``Outfit.history`` drives outfit-level stats.
    """
    id: str
    name: Optional[str] = Field(None, description="User-facing name, e.g. 'Monday Work Look'")
    items: List[OutfitItem] = Field(
        ...,
        description="Ordered list of garment references (garment_id + role). "
                    "Resolve full garment data from the wardrobe index.",
    )
    compatibility_score: float = Field(..., ge=0, le=1)
    style_coherence_score: float = Field(..., ge=0, le=1)
    occasion_match_score: float = Field(..., ge=0, le=1)
    overall_score: float = Field(..., ge=0, le=1)

    # Seven-point rule analysis
    seven_point_total: Optional[int] = Field(None, description="Total points (target: 7-10)")
    seven_point_harmonious: Optional[bool] = Field(None, description="Whether outfit is in optimal range")

    explanation: Optional[str] = None
    tags: List[str] = Field(default_factory=list, description="Free-text tags: capsule, travel, work, etc.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ---- Lifecycle history (single source of truth for this outfit) ----
    history: OutfitHistory = Field(default_factory=OutfitHistory)

    # ------------------------------------------------------------------ #
    # Convenience helpers — derive data from history without duplicating   #
    # ------------------------------------------------------------------ #

    @computed_field  # type: ignore[misc]
    @property
    def garment_ids(self) -> List[str]:
        """Ordered list of garment IDs that make up this outfit."""
        return [item.garment_id for item in self.items]

    @computed_field  # type: ignore[misc]
    @property
    def total_wears(self) -> int:
        """How many times this outfit has been worn (from history)."""
        return self.history.total_wears

    @computed_field  # type: ignore[misc]
    @property
    def last_worn_at(self) -> Optional[datetime]:
        """Most recent wear date (from history)."""
        return self.history.last_worn_at

    @computed_field  # type: ignore[misc]
    @property
    def days_since_last_worn(self) -> Optional[int]:
        """Days elapsed since last worn; None if never worn."""
        return self.history.days_since_last_worn

    def record_wear(
        self,
        wardrobe: Dict[str, "Garment"],
        *,
        occasion: Optional[Occasion] = None,
        wear_context: Optional[WearContext] = None,
        weather_condition: Optional[str] = None,
        temperature_celsius: Optional[float] = None,
        location: Optional[str] = None,
        user_rating: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Record a wear event for this outfit AND all its constituent garments.

        Args:
            wardrobe: mapping of garment_id → Garment so each garment's
                      history can be updated in-place.
            All other args are forwarded to the ``WearRecord``.
        """
        record = WearRecord(
            worn_at=datetime.now(timezone.utc),
            occasion=occasion,
            wear_context=wear_context,
            weather_condition=weather_condition,
            temperature_celsius=temperature_celsius,
            location=location,
            outfit_id=self.id,
            user_rating=user_rating,
            notes=notes,
        )
        # 1. Update outfit history
        self.history.wear_records.append(record)

        # 2. Mirror wear into every constituent garment (with outfit_id link)
        for item in self.items:
            garment = wardrobe.get(item.garment_id)
            if garment is not None:
                garment.history.wear_records.append(record.model_copy())


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


# ============== Wardrobe Analysis Models ==============

class CategoryDistribution(BaseModel):
    """Distribution count for a single category."""
    category: str
    count: int
    percentage: float = Field(ge=0, le=100)


class WardrobeDistribution(BaseModel):
    """Distribution analysis of a wardrobe across multiple dimensions."""
    by_category: List[CategoryDistribution] = Field(default_factory=list)
    by_color: List[CategoryDistribution] = Field(default_factory=list)
    by_formality: List[CategoryDistribution] = Field(default_factory=list)
    by_season: List[CategoryDistribution] = Field(default_factory=list)
    by_material: List[CategoryDistribution] = Field(default_factory=list)
    by_pattern: List[CategoryDistribution] = Field(default_factory=list)
    total_items: int = 0


class WardrobeGap(BaseModel):
    """An identified gap or imbalance in the wardrobe."""
    gap_type: str = Field(..., description="category_missing, color_imbalance, formality_gap, season_gap, versatility_low")
    severity: str = Field(..., description="low, medium, high")
    description: str
    recommendation: str


class OccasionCoverage(BaseModel):
    """How well a wardrobe covers a specific occasion."""
    occasion: str
    coverage_score: float = Field(ge=0, le=1)
    suitable_items_count: int
    missing_categories: List[str] = Field(default_factory=list)
    suggestion: Optional[str] = None


class GarmentVersatility(BaseModel):
    """Versatility score for a single garment."""
    garment_id: str
    garment_description: str
    versatility_score: float = Field(ge=0, le=1)
    compatible_outfit_count: int
    compatible_categories: List[str] = Field(default_factory=list)
    compatible_occasions: List[str] = Field(default_factory=list)


class PurchaseSuggestion(BaseModel):
    """A suggested purchase to improve the wardrobe."""
    priority: int = Field(ge=1, description="1 = highest priority")
    category: str
    description: str
    reason: str
    estimated_outfit_increase: int = Field(default=0, description="How many new outfits this would enable")
    suggested_colors: List[str] = Field(default_factory=list)
    suggested_styles: List[str] = Field(default_factory=list)
    target_occasions: List[str] = Field(default_factory=list)


class WardrobeAnalysisResult(BaseModel):
    """Complete wardrobe analysis result."""
    distribution: WardrobeDistribution
    gaps: List[WardrobeGap] = Field(default_factory=list)
    occasion_coverage: List[OccasionCoverage] = Field(default_factory=list)
    top_versatile_items: List[GarmentVersatility] = Field(default_factory=list)
    purchase_suggestions: List[PurchaseSuggestion] = Field(default_factory=list)
    overall_score: float = Field(ge=0, le=1, description="Overall wardrobe health score")
    summary: str = ""


class OutfitDiagnosis(BaseModel):
    """Diagnosis of an outfit's weak points."""
    overall_score: float = Field(ge=0, le=1)
    grade: str
    weak_dimensions: List[Dict[str, Any]] = Field(default_factory=list, description="Dimensions scoring below threshold")
    strong_dimensions: List[Dict[str, Any]] = Field(default_factory=list, description="Well-performing dimensions")
    improvement_potential: float = Field(ge=0, le=1, description="How much the outfit can improve")


class AdditionSuggestion(BaseModel):
    """Suggestion to add a garment from the wardrobe to an outfit."""
    garment_id: str
    garment_description: str
    expected_score_change: float
    reason: str


class ReplacementSuggestion(BaseModel):
    """Suggestion to replace one garment with another from the wardrobe."""
    original_garment_id: str
    original_description: str
    replacement_garment_id: str
    replacement_description: str
    expected_score_change: float
    reason: str


class PurchaseTargeted(BaseModel):
    """Suggestion to buy a specific garment to improve a specific outfit."""
    category: str
    description: str
    reason: str
    expected_score_change: float
    suggested_attributes: Dict[str, Any] = Field(default_factory=dict)


class OutfitImprovementResult(BaseModel):
    """Complete outfit improvement analysis."""
    diagnosis: OutfitDiagnosis
    additions: List[AdditionSuggestion] = Field(default_factory=list)
    replacements: List[ReplacementSuggestion] = Field(default_factory=list)
    purchase_suggestions: List[PurchaseTargeted] = Field(default_factory=list)
    summary: str = ""


class RemovalImpact(BaseModel):
    """Impact analysis of removing a garment from the wardrobe."""
    garment_id: str
    garment_description: str
    garment_attributes: Optional["GarmentAttributes"] = None
    outfits_affected: int
    versatility_score: float = Field(ge=0, le=1)
    replacement_available: bool
    replacement_suggestions: List[str] = Field(default_factory=list)
    impact_level: str = Field(..., description="low, medium, high, critical")
    is_critical: bool = False
    has_replacement: bool = False
    summary: str = ""


# ============== Smart Removal Models ==============

class UserRemovalGoal(str, Enum):
    """User goals that shift how aggressively removal is recommended."""
    MINIMALIST = "minimalist"           # Tolerate losing outfits if wardrobe has redundancy
    MAXIMIZE_OPTIONS = "maximize_options"  # Penalise any loss of outfit diversity
    STYLE_UPGRADE = "style_upgrade"     # Favour removal of low-scored / low-rated items


class ReplacementCandidate(BaseModel):
    """A validated replacement for a garment being considered for removal."""
    garment_id: str
    garment_description: str
    coverage_match: float = Field(ge=0, le=1, description="0-1 how well it covers the role of the removed item")
    matched_seasons: List[Season] = Field(default_factory=list)
    matched_occasions: List[str] = Field(default_factory=list)
    formality_compatible: bool = True
    color_role_match: bool = True
    reason_not_perfect: Optional[str] = None


class ReplacementQuality(BaseModel):
    """Aggregated quality of available replacements."""
    valid_count: int = 0
    quality_level: str = Field(default="none", description="none, low, medium, high")
    best_candidate: Optional[ReplacementCandidate] = None
    all_candidates: List[ReplacementCandidate] = Field(default_factory=list)
    coverage_gaps: List[str] = Field(
        default_factory=list,
        description="What the replacements do NOT cover (e.g., 'winter season', 'formal occasions')",
    )


class SmartRemovalReason(BaseModel):
    """A single reason contributing to the removal verdict, with evidence."""
    signal_name: str = Field(..., description="E.g. 'recency_decay', 'replacement_quality', 'outfit_impact'")
    direction: str = Field(..., description="supports_removal, supports_keeping")
    weight: float = Field(ge=0, le=1, description="How much this signal mattered")
    score_contribution: float = Field(description="Positive = towards removal, negative = towards keeping")
    explanation: str = Field(..., description="One-sentence human explanation")


class DataGap(BaseModel):
    """Signals that certain data was unavailable, lowering confidence."""
    field: str
    description: str


class SmartRemovalVerdict(BaseModel):
    """Full smart-removal analysis output for a single garment."""
    # ---- Identity ----
    garment_id: str
    garment_description: str
    garment_attributes: Optional["GarmentAttributes"] = None

    # ---- Core scores ----
    regret_risk: float = Field(ge=0, le=1, description="0 = safe to remove, 1 = high regret risk")
    verdict: str = Field(..., description="SAFE_TO_REMOVE, CONSIDER, KEEP, DONATE")
    confidence: float = Field(ge=0, le=1, description="How reliable this verdict is given available data")

    # ---- Signal breakdown ----
    signals: Dict[str, float] = Field(
        default_factory=dict,
        description="Raw normalised signals: recency_score, frequency_score, versatility_score, etc.",
    )
    reasons: List[SmartRemovalReason] = Field(
        default_factory=list,
        description="Top reasons sorted by absolute contribution",
    )

    # ---- Outfit impact ----
    outfits_affected_weighted: float = Field(
        default=0.0,
        description="Quality-weighted outfit impact (not just count)",
    )
    outfits_affected_by_occasion: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown: {'casual': 5, 'work': 2, ...}",
    )

    # ---- Replacement quality ----
    replacement: ReplacementQuality = Field(default_factory=ReplacementQuality)

    # ---- Explainability ----
    what_you_lose: List[str] = Field(default_factory=list, description="Bullet points: what is lost")
    what_you_keep: List[str] = Field(default_factory=list, description="Bullet points: what remains")
    data_gaps: List[DataGap] = Field(default_factory=list, description="Missing data that lowers confidence")

    # ---- User goal used ----
    user_goal: UserRemovalGoal = UserRemovalGoal.MAXIMIZE_OPTIONS

    summary: str = ""

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


# ============== Capsule Wardrobe Models ==============

class GarmentCapsuleRole(str, Enum):
    """Role of a garment in a capsule wardrobe."""
    KEY_PIECE = "key_piece"       # High versatility — used in many outfits
    ACCEPTABLE = "acceptable"     # Mid versatility — useful but not critical
    ORPHAN = "orphan"             # Low versatility — rarely combined
    REDUNDANT = "redundant"       # Duplicate of another piece


# Alias for backward-compat and convenience
CapsuleGarmentRole = GarmentCapsuleRole


class CapsuleGarmentScore(BaseModel):
    """Versatility / capsule score for a single garment."""
    garment_id: str
    garment_description: str
    versatility_score: float = Field(ge=0.0, le=1.0)
    outfit_count: int = Field(default=0, description="Number of valid outfits this garment appears in")
    total_outfits: int = Field(default=1, description="Total outfits in wardrobe (denominator)")
    capsule_role: GarmentCapsuleRole = GarmentCapsuleRole.ACCEPTABLE
    compatible_garment_ids: List[str] = Field(default_factory=list)


class RedundantPair(BaseModel):
    """A pair of garments that are functionally similar (potential duplicate)."""
    garment_a_id: str
    garment_a_description: str
    garment_b_id: str
    garment_b_description: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class CapsuleSnapshot(BaseModel):
    """Point-in-time snapshot of wardrobe capsule health."""
    date: str = Field(description="ISO-8601 date string")
    cohesion_score: float = Field(ge=0.0, le=100.0)
    garments_count: int
    outfits_count: int
    key_pieces_count: int
    orphans_count: int
    action_taken: str = ""  # e.g. "Added black blazer", "Removed pink top"
    delta_score: float = 0.0
    delta_outfits: int = 0


class CapsuleAnalysisResult(BaseModel):
    """
    Full capsule wardrobe analysis result (F1).

    Extends the existing WardrobeAnalysisResult with capsule-specific metrics:
    cohesion score, colour palette, orphan detection, redundancy pairs,
    key pieces and a human-readable capsule recommendation.
    """
    # Core scores
    cohesion_score: float = Field(ge=0.0, le=100.0, description="0–100 capsule health score")
    color_cohesion_score: float = Field(ge=0.0, le=1.0)
    versatility_ratio: float = Field(ge=0.0, le=1.0, description="% pieces used in >3 outfits")
    redundancy_penalty: float = Field(ge=0.0, le=1.0)
    orphan_penalty: float = Field(ge=0.0, le=1.0)

    # Palette
    dominant_colors: List[str] = Field(default_factory=list)
    color_coverage_pct: float = Field(ge=0.0, le=1.0, description="% garments using dominant palette")

    # Piece classification
    garment_scores: List[CapsuleGarmentScore] = Field(default_factory=list)
    key_pieces: List[str] = Field(default_factory=list, description="Garment IDs with high versatility")
    orphan_pieces: List[str] = Field(default_factory=list, description="Garment IDs with low versatility")
    redundant_pairs: List[RedundantPair] = Field(default_factory=list)

    # Totals
    total_garments: int = 0
    total_outfits: int = 0

    # Recommendation
    capsule_profile: str = "standard"  # minimalist / standard / rich
    recommendation: str = ""
    projected_score_after_cleanup: float = Field(ge=0.0, le=100.0, default=0.0)


class MissingPieceRecommendation(BaseModel):
    """A single piece recommended to fill a capsule gap (F2)."""
    priority: int = Field(ge=1)
    category: str
    description: str
    reason: str
    impact_outfits: int = Field(default=0, description="Estimated new outfits this piece enables")
    suggested_colors: List[str] = Field(default_factory=list)
    profile_note: str = ""   # personalised note (morphology / season)
    llm_narration: str = ""  # LLM-generated explanation


class MissingPiecesResult(BaseModel):
    """Result of the missing-pieces recommender (F2)."""
    current_cohesion: float
    projected_cohesion: float
    recommendations: List[MissingPieceRecommendation] = Field(default_factory=list)
    summary: str = ""


class ReplacementVerdict(BaseModel):
    """Verdict for a redundant pair — which to keep (F3)."""
    garment_keep_id: str
    garment_keep_description: str = ""
    garment_remove_id: str
    garment_remove_description: str = ""
    versatility_gain: float = Field(default=0.0, description="Versatility score difference (winner − loser)")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    transition_timing: str = ""  # e.g. "Replace in September"
    llm_narration: str = ""


class ReplacementPlanResult(BaseModel):
    """Full replacement plan for all redundant pairs (F3)."""
    verdicts: List[ReplacementVerdict] = Field(default_factory=list)
    total_outfits_gained: int = 0
    summary: str = ""


class CapsuleOutfit(BaseModel):
    """An outfit selected / scored for its capsule quality (F4)."""
    rank: int = 0
    garment_ids: List[str] = Field(default_factory=list)
    garment_descriptions: List[str] = Field(default_factory=list)
    capsule_score: float = Field(ge=0.0, le=100.0, default=0.0)
    overall_style_score: float = Field(ge=0.0, le=1.0, default=0.0)
    pct_key_pieces: float = Field(ge=0.0, le=1.0, default=0.0)
    avg_versatility: float = Field(ge=0.0, le=1.0, default=0.0)
    tier: str = "creative"   # basic / semi_creative / creative
    occasions: List[str] = Field(default_factory=list)
    variations_count: int = 0
    llm_narration: str = ""


class CapsuleOutfitsResult(BaseModel):
    """Result of the capsule outfit generator (F4)."""
    outfits: List[CapsuleOutfit] = Field(default_factory=list)
    basic_count: int = 0
    semi_creative_count: int = 0
    creative_count: int = 0


class CapsuleEvolutionResult(BaseModel):
    """Result of the evolution tracker (F5)."""
    snapshots: List[CapsuleSnapshot] = Field(default_factory=list)
    delta_score: float = 0.0
    delta_outfits: int = 0
    delta_pieces: int = 0
    outfit_ratio: float = 1.0   # current / initial outfits
    pieces_ratio: float = 1.0   # initial / current pieces
    predicted_weeks_to_90: Optional[int] = None
    trend_direction: str = "stable"  # improving / declining / stable
    latest_cohesion: float = 0.0

