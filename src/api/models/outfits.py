"""Pydantic schemas for custom outfit API requests and responses."""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SubscriptionTier(str, Enum):
    """Available subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    PREMIUM = "premium"


# ── User Subscription Schemas ──

class UserSubscriptionCreate(BaseModel):
    """Request body for creating a user subscription."""
    user_id: str = Field(..., description="Unique user identifier")
    subscription_tier: SubscriptionTier = Field(default=SubscriptionTier.FREE)


class UserSubscriptionUpdate(BaseModel):
    """Request body for updating subscription."""
    is_premium: Optional[bool] = None
    subscription_tier: Optional[SubscriptionTier] = None
    expires_at: Optional[datetime] = None


class UserSubscriptionResponse(BaseModel):
    """Response for subscription queries."""
    id: str
    user_id: str
    is_premium: bool
    subscription_tier: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_renewed_at: Optional[datetime]
    custom_outfits_created: int
    analyses_performed: int
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


# ── Custom Outfit Schemas ──

class CustomOutfitCreate(BaseModel):
    """Request body for creating a custom outfit."""
    outfit_name: str = Field(..., min_length=1, max_length=255, description="Name of the outfit")
    garment_ids: List[str] = Field(..., description="List of garment IDs to include")
    user_season: Optional[str] = Field(None, description="User's color season (e.g., autumn, spring)")
    intended_occasion: Optional[str] = Field(None, description="Intended occasion (e.g., business, casual)")
    notes: Optional[str] = Field(None, max_length=1000, description="User notes about the outfit")


class CustomOutfitUpdate(BaseModel):
    """Request body for updating a custom outfit."""
    outfit_name: Optional[str] = Field(None, max_length=255)
    notes: Optional[str] = Field(None, max_length=1000)
    intended_occasion: Optional[str] = None


class CustomOutfitResponse(BaseModel):
    """Response for custom outfit queries."""
    id: str
    user_id: str
    outfit_name: str
    garment_ids: List[str]
    overall_score: Optional[float]
    score_grade: Optional[str]
    user_season: Optional[str]
    intended_occasion: Optional[str]
    notes: Optional[str]
    created_at: datetime
    saved_at: Optional[datetime]
    last_analyzed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class CustomOutfitWithAnalysis(CustomOutfitResponse):
    """Response including cached analysis."""
    latest_analysis: Optional["OutfitAnalysisResponse"] = None


# ── Outfit Analysis Schemas ──

class OutfitImprovementSuggestion(BaseModel):
    """Single improvement suggestion."""
    item: str = Field(..., description="Item to add or modify")
    reason: str = Field(..., description="Why this improvement helps")


class OutfitReplacement(BaseModel):
    """Replacement suggestion."""
    original: str = Field(..., description="Original garment or aspect")
    replacement: str = Field(..., description="Suggested replacement")
    reason: str = Field(..., description="Why this replacement is better")


class OutfitPurchase(BaseModel):
    """Purchase recommendation for outfit."""
    dimension: str = Field(..., description="Outfit dimension to improve")
    description: str = Field(..., description="What to purchase")
    reason: str = Field(..., description="Why this purchase helps")


class ImprovementExplanation(BaseModel):
    """Complete improvement explanation for an outfit."""
    additions: List[OutfitImprovementSuggestion] = Field(default_factory=list)
    replacements: List[OutfitReplacement] = Field(default_factory=list)
    purchases: List[OutfitPurchase] = Field(default_factory=list)


class OutfitAnalysisResponse(BaseModel):
    """Response for outfit analysis queries."""
    id: str
    outfit_id: str
    user_id: str
    analysis_type: str
    improvement_explanation: Optional[ImprovementExplanation]
    styling_tips: Optional[Dict[str, Any]]
    generated_at: datetime
    user_season: Optional[str]
    body_shape: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class OutfitAnalysisRequest(BaseModel):
    """Request to analyze a custom outfit (premium only)."""
    user_season: Optional[str] = Field(None, description="User's color season")
    body_shape: Optional[str] = Field(None, description="User's body shape")


# ── Error Responses ──

class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(..., description="Error message")
    error_code: str = Field(default="INTERNAL_ERROR")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UnauthorizedResponse(ErrorResponse):
    """401 Unauthorized response."""
    error_code: str = "UNAUTHORIZED"


class ForbiddenResponse(ErrorResponse):
    """403 Forbidden response."""
    error_code: str = "FORBIDDEN"


class NotFoundResponse(ErrorResponse):
    """404 Not Found response."""
    error_code: str = "NOT_FOUND"


class PremiumRequiredResponse(ErrorResponse):
    """403 Premium required response."""
    error_code: str = "PREMIUM_REQUIRED"
    upgrade_url: str = Field(default="https://example.com/upgrade")
