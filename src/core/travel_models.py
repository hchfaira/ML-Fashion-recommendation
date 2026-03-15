"""Travel and season wardrobe planning data models.

All models are Pydantic V2 dataclasses so they serialise cleanly to JSON
and integrate with the rest of the LLM-project pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TravelClimate(str, Enum):
    TROPICAL = "tropical"
    MEDITERRANEAN = "mediterranean"
    CONTINENTAL = "continental"
    COLD = "cold"
    ARID = "arid"
    UNKNOWN = "unknown"


class SeasonReadiness(str, Enum):
    READY = "ready"
    ADAPTABLE = "adaptable"
    STORE = "store"


class TransitionPhase(str, Enum):
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"


# ---------------------------------------------------------------------------
# Travel feature models
# ---------------------------------------------------------------------------

class TravelConstraints(BaseModel):
    """User-supplied travel constraints for the packing planner."""
    destination: str = Field(default="generic", description="City or country name (lower-case)")
    days: int = Field(default=7, ge=1, le=365)
    occasions: Dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of occasion name to number of days, e.g. {'casual':4, 'evening':2}",
    )
    max_pieces: int = Field(default=12, ge=3, le=50)
    travel_date: Optional[date] = None
    body_shape: Optional[str] = None
    season: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


class PackedGarment(BaseModel):
    """A garment selected for packing with its computed scores."""
    garment_id: str
    category: str
    description: str
    packing_score: float = Field(ge=0.0, le=1.0)
    occasion_tags: List[str] = Field(default_factory=list)
    outfits_enabled: int = Field(
        default=0, description="How many outfit combinations this piece enables"
    )
    weather_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class DayPlan(BaseModel):
    """Outfit assignment for a single travel day."""
    day: int
    occasion: str
    garment_ids: List[str]
    outfit_name: str = ""
    notes: str = ""


class PackingPlan(BaseModel):
    """Full output of the TravelWardrobePlanner."""
    destination: str
    days: int
    max_pieces: int
    packed_garments: List[PackedGarment]
    day_plans: List[DayPlan]
    packing_score: float = Field(ge=0.0, le=100.0)
    score_label: str = "fair"
    versatility_ratio: float = Field(
        default=0.0,
        description="outfit_combinations / pieces_packed; target > 2.0",
    )
    occasion_coverage: Dict[str, float] = Field(
        default_factory=dict,
        description="Fraction of days per occasion covered by packed garments",
    )
    warnings: List[str] = Field(default_factory=list)
    llm_narration: Optional[str] = None


# ---------------------------------------------------------------------------
# Seasonal audit models
# ---------------------------------------------------------------------------

class GarmentSeasonAudit(BaseModel):
    """Audit result for a single garment against a target season."""
    garment_id: str
    description: str
    category: str
    readiness: SeasonReadiness
    score: float = Field(ge=0.0, le=1.0, description="0=store, 1=fully ready")
    reasons: List[str] = Field(default_factory=list)


class SeasonCoverage(BaseModel):
    """Coverage statistics per category."""
    category: str
    required: int
    available: int
    gap: int = 0


class SeasonAuditResult(BaseModel):
    """Full output of the SeasonalAuditEngine."""
    target_season: str
    garment_audits: List[GarmentSeasonAudit]
    ready_count: int = 0
    adaptable_count: int = 0
    store_count: int = 0
    coverage_by_category: List[SeasonCoverage] = Field(default_factory=list)
    overall_readiness_pct: float = Field(ge=0.0, le=100.0, default=0.0)
    top_gaps: List[str] = Field(default_factory=list)
    llm_narration: Optional[str] = None


# ---------------------------------------------------------------------------
# Shopping list models
# ---------------------------------------------------------------------------

class ShoppingItem(BaseModel):
    """One prioritised shopping recommendation."""
    rank: int
    category: str
    description: str
    estimated_price_eur: float
    outfits_unlocked: int = Field(
        default=0, description="New outfit combinations this piece would enable"
    )
    roi: float = Field(
        default=0.0, description="outfits_unlocked / estimated_price_eur * 100"
    )
    urgency: str = Field(default="low", description="high | medium | low")
    color_suggestions: List[str] = Field(default_factory=list)
    reason: str = ""


class ShoppingListResult(BaseModel):
    """Full output of the ShoppingListOptimizer."""
    budget_eur: float
    items: List[ShoppingItem]
    total_estimated_cost: float
    projected_new_outfits: int
    within_budget: List[ShoppingItem] = Field(default_factory=list)
    over_budget: List[ShoppingItem] = Field(default_factory=list)
    llm_narration: Optional[str] = None


# ---------------------------------------------------------------------------
# Weekly rotation models
# ---------------------------------------------------------------------------

class RotationSlot(BaseModel):
    """One slot in the weekly plan (a day × session combination)."""
    day: str
    session: str = "full_day"
    occasion: str
    garment_ids: List[str]
    outfit_name: str = ""
    formality_level: int = 1


class WeeklyRotationPlan(BaseModel):
    """Full output of the WeeklyRotationPlanner."""
    week_label: str = "Week 1"
    slots: List[RotationSlot]
    repeat_rate: float = Field(
        default=0.0, description="Fraction of days with a repeated outfit (target < 0.2)"
    )
    coverage_pct: float = Field(
        default=0.0, description="Fraction of requested occasion types covered"
    )
    garment_utilisation: Dict[str, int] = Field(
        default_factory=dict,
        description="garment_id → number of times used this week",
    )
    warnings: List[str] = Field(default_factory=list)
    llm_narration: Optional[str] = None


# ---------------------------------------------------------------------------
# Season transition models
# ---------------------------------------------------------------------------

class TransitionAction(BaseModel):
    """A concrete action to take during wardrobe transition."""
    action: str = Field(description="store | keep | buy | layer")
    garment_id: Optional[str] = None
    description: str
    phase: TransitionPhase
    priority: int = Field(default=3, ge=1, le=5, description="1=highest priority")


class SeasonTransitionPlan(BaseModel):
    """Full output of the SeasonTransitionAdvisor."""
    current_season: str
    target_season: str
    actions: List[TransitionAction]
    store_ids: List[str] = Field(default_factory=list)
    keep_ids: List[str] = Field(default_factory=list)
    buy_descriptions: List[str] = Field(default_factory=list)
    color_shift_notes: List[str] = Field(default_factory=list)
    phase_summary: Dict[str, str] = Field(default_factory=dict)
    llm_narration: Optional[str] = None
