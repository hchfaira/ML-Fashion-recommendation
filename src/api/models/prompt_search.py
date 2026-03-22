"""Pydantic schemas for the Natural Language Outfit Search API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PromptSearchRequest(BaseModel):
    """Request body for a one-shot prompt-based outfit search."""
    prompt: str = Field(..., min_length=2, max_length=1000, description="Free-text outfit request")
    candidate_outfits: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Candidate outfit dicts from the user's wardrobe",
    )
    base_scores: Optional[List[float]] = Field(
        None,
        description="Pre-computed style scores (same length as candidate_outfits)",
    )
    max_results: int = Field(10, ge=1, le=50)
    preference: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="Weight of prompt score vs base style score (0=ignore prompt, 1=only prompt)",
    )
    explain: bool = Field(True, description="Whether to generate an LLM explanation")


class ConversationStartRequest(BaseModel):
    """Start a conversational outfit search session."""
    prompt: str = Field(..., min_length=2, max_length=1000)
    candidate_outfits: List[Dict[str, Any]] = Field(default_factory=list)
    base_scores: Optional[List[float]] = None
    max_results: int = Field(10, ge=1, le=50)
    preference: float = Field(0.4, ge=0.0, le=1.0)


class ConversationRefineRequest(BaseModel):
    """Refine an existing conversational search session."""
    refinement: str = Field(..., min_length=1, max_length=500, description="Follow-up message")
    candidate_outfits: List[Dict[str, Any]] = Field(default_factory=list)
    base_scores: Optional[List[float]] = None
    max_results: int = Field(10, ge=1, le=50)
    preference: float = Field(0.4, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class ParsedPromptResponse(BaseModel):
    """How the system interpreted the user's prompt."""
    raw_prompt: str
    colors: List[str]
    occasion: Optional[str]
    formality_range: List[float]  # [min, max]
    styles: List[str]
    season_hint: Optional[str]
    anchor_pieces: List[str]
    excluded_types: List[str]
    mood: Optional[str]
    confidence: float
    is_off_topic: bool
    clarification_needed: bool

    model_config = ConfigDict(from_attributes=True)


class WardrobeFiltersResponse(BaseModel):
    """Concrete wardrobe filter criteria derived from the prompt."""
    excluded_types: List[str]
    formality_min: float
    formality_max: float
    target_colors: List[str]
    color_families: List[str]
    target_styles: List[str]
    target_occasion: Optional[str]
    preferred_patterns: List[str]
    preferred_materials: List[str]
    season_hint: Optional[str]
    weight_color: float
    weight_style: float
    weight_occasion: float

    model_config = ConfigDict(from_attributes=True)


class ScoredOutfitResponse(BaseModel):
    """An outfit enriched with prompt relevance scores."""
    prompt_score: float = Field(..., description="How well the outfit matches the prompt (0-1)")
    final_score: float = Field(..., description="Blended prompt + base score (0-1)")
    outfit: Dict[str, Any] = Field(..., description="Original outfit dict")

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PromptSearchResponse(BaseModel):
    """Response for a prompt-based outfit search."""
    outfits: List[Dict[str, Any]] = Field(
        description="Ranked outfits (each includes prompt_score and final_score keys)"
    )
    prompt_interpretation: ParsedPromptResponse
    confidence: float
    explanation: str = Field("", description="LLM explanation for the top result")
    compromise_note: Optional[str] = Field(None, description="Compromise explanation when wardrobe is imperfect")
    missing_piece_suggestion: Optional[str] = Field(None, description="What to buy when no outfit matches")
    session_id: Optional[str] = Field(None, description="Session ID for conversational refinement")
    total_candidates: int
    is_off_topic: bool
    clarification_needed: bool

    model_config = ConfigDict(from_attributes=True)


class ParseOnlyResponse(BaseModel):
    """Debug response showing how a prompt was interpreted."""
    parsed: ParsedPromptResponse
    filters: WardrobeFiltersResponse

    model_config = ConfigDict(from_attributes=True)


class SuggestionsResponse(BaseModel):
    """Example search prompts."""
    suggestions: List[str]
