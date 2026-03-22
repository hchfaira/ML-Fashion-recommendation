"""Pydantic schemas for Mood Board API requests and responses."""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime


# ── Shared Outfit Schemas ──

class ShareOutfitRequest(BaseModel):
    """Request body for sharing an outfit."""
    outfit_data: Dict[str, Any] = Field(..., description="Snapshot of the outfit (garment_ids, scores, attributes)")
    title: Optional[str] = Field(None, max_length=255, description="Optional display title")
    occasion_tags: List[str] = Field(default_factory=list, description="Occasion tags, e.g. ['casual', 'summer']")
    style_tags: List[str] = Field(default_factory=list, description="Style tags, e.g. ['minimalist']")
    formality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    dominant_colors: List[str] = Field(default_factory=list)
    dominant_styles: Dict[str, float] = Field(default_factory=dict)
    embedding_vector: List[float] = Field(default_factory=list)
    is_public: bool = Field(default=True)


class SharedOutfitResponse(BaseModel):
    """Response for a shared outfit."""
    id: str
    owner_user_id: str
    outfit_data: Dict[str, Any]
    title: Optional[str]
    occasion_tags: List[str]
    style_tags: List[str]
    formality_score: Optional[float]
    dominant_colors: List[str]
    dominant_styles: Dict[str, float]
    likes_count: int
    saves_count: int
    is_public: bool
    shared_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ── Mood Board Schemas ──

class MoodBoardCreate(BaseModel):
    """Request body for creating a mood board."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)


class MoodBoardUpdate(BaseModel):
    """Request body for updating a mood board."""
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)


class MoodBoardResponse(BaseModel):
    """Response for a mood board."""
    id: str
    user_id: str
    name: str
    description: Optional[str]
    is_active: bool
    items_count: int
    created_at: datetime
    updated_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


# ── Mood Board Item Schemas ──

class SaveOutfitToBoardRequest(BaseModel):
    """Request body for saving a shared outfit into a mood board."""
    shared_outfit_id: str = Field(..., description="ID of the shared outfit to save")
    personal_note: Optional[str] = Field(None, max_length=500)


class MoodBoardItemResponse(BaseModel):
    """Response for a single mood board item."""
    id: str
    board_id: str
    shared_outfit_id: str
    personal_note: Optional[str]
    saved_at: datetime
    shared_outfit: Optional[SharedOutfitResponse] = None

    model_config = ConfigDict(from_attributes=True)


# ── Style Profile Schemas ──

class ColorFrequency(BaseModel):
    """A color with its frequency in the mood board."""
    color: str
    frequency: float = Field(..., ge=0.0, le=1.0)


class MoodBoardStyleProfileResponse(BaseModel):
    """The aggregated style profile of a mood board."""
    board_id: str
    dominant_colors: List[ColorFrequency]
    dominant_styles: Dict[str, float]
    formality_average: Optional[float]
    occasions_distribution: Dict[str, float]
    coherence_score: Optional[float]
    items_count: int
    is_stale: bool
    last_computed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


# ── Gap Analysis Schemas ──

class MoodBoardGapAnalysis(BaseModel):
    """Gap analysis between a mood board profile and the user's wardrobe."""
    alignment_score: float = Field(..., ge=0.0, le=100.0, description="Overall alignment 0-100")
    missing_colors: List[str] = Field(default_factory=list)
    missing_styles: List[str] = Field(default_factory=list)
    missing_piece_types: List[str] = Field(default_factory=list)
    well_covered: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None


# ── Outfit Recommendation with Mood Board Influence ──

class MoodBoardInfluence(BaseModel):
    """Metadata about how a mood board influenced a recommendation."""
    moodboard_score: float = Field(..., ge=0.0, le=1.0)
    matching_saved_outfit_ids: List[str] = Field(default_factory=list)
    color_alignment: List[str] = Field(default_factory=list)
    style_alignment: List[str] = Field(default_factory=list)
    boost_applied: float = Field(default=0.0)
    explanation: Optional[str] = None


class MoodBoardRecommendationResponse(BaseModel):
    """An outfit recommendation influenced by a mood board."""
    garment_ids: List[str]
    base_score: float
    final_score: float
    moodboard_influence: MoodBoardInfluence


# ── Discovery Feed ──

class DiscoveryFeedRequest(BaseModel):
    """Filters for the shared outfit discovery feed."""
    occasion_tag: Optional[str] = None
    style_tag: Optional[str] = None
    min_formality: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_formality: Optional[float] = Field(None, ge=0.0, le=1.0)
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=100)
