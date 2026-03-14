"""SQLAlchemy ORM models for outfit storage and user subscriptions."""
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from uuid import uuid4

from src.database import Base


class UserSubscription(Base):
    """Track user subscription status and premium access."""
    __tablename__ = "user_subscriptions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), unique=True, nullable=False, index=True)
    
    # Subscription details
    is_premium = Column(Boolean, default=False)
    subscription_tier = Column(String(50), default="free")  # free, basic, pro, premium
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)  # None if free or unlimited
    last_renewed_at = Column(DateTime, nullable=True)
    
    # Usage tracking
    custom_outfits_created = Column(Integer, default=0)
    analyses_performed = Column(Integer, default=0)
    
    # Relationships
    custom_outfits = relationship("CustomOutfit", back_populates="user")
    outfit_analyses = relationship("OutfitAnalysis", back_populates="user")

    def is_active(self) -> bool:
        """Check if subscription is currently active."""
        if not self.is_premium:
            return False
        if self.expires_at is None:
            return True
        return datetime.utcnow() < self.expires_at


class CustomOutfit(Base):
    """User's custom outfit created from selected garments."""
    __tablename__ = "custom_outfits"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), ForeignKey("user_subscriptions.user_id"), nullable=False, index=True)
    
    # Outfit details
    outfit_name = Column(String(255), nullable=False)
    garment_ids = Column(JSON, nullable=False)  # List of garment IDs
    
    # Scoring
    overall_score = Column(Float, nullable=True)
    score_grade = Column(String(2), nullable=True)  # A+, A, B+, B, C, etc.
    
    # Context
    user_season = Column(String(50), nullable=True)
    intended_occasion = Column(String(100), nullable=True)
    
    # Notes
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    saved_at = Column(DateTime, nullable=True)
    last_analyzed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("UserSubscription", back_populates="custom_outfits")
    analyses = relationship("OutfitAnalysis", back_populates="outfit")

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "outfit_name": self.outfit_name,
            "garment_ids": self.garment_ids,
            "overall_score": self.overall_score,
            "score_grade": self.score_grade,
            "user_season": self.user_season,
            "intended_occasion": self.intended_occasion,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "saved_at": self.saved_at.isoformat() if self.saved_at else None,
            "last_analyzed_at": self.last_analyzed_at.isoformat() if self.last_analyzed_at else None,
        }


class OutfitAnalysis(Base):
    """LLM-generated improvement explanation for a custom outfit (cached)."""
    __tablename__ = "outfit_analyses"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    outfit_id = Column(String(36), ForeignKey("custom_outfits.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("user_subscriptions.user_id"), nullable=False, index=True)
    
    # Analysis content (structured as JSON for flexibility)
    analysis_type = Column(String(50), default="improvement")  # improvement, explanation, styling_tip
    
    improvement_explanation = Column(JSON, nullable=True)  # Cached explanation from Layer 4 LLM
    # Structure: {
    #     "additions": [{"item": str, "reason": str}, ...],
    #     "replacements": [{"original": str, "replacement": str, "reason": str}, ...],
    #     "purchases": [{"dimension": str, "description": str, "reason": str}, ...],
    # }
    
    styling_tips = Column(JSON, nullable=True)  # Additional styling insights
    
    # Metadata
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_season = Column(String(50), nullable=True)
    body_shape = Column(String(50), nullable=True)
    
    # Relationships
    outfit = relationship("CustomOutfit", back_populates="analyses")
    user = relationship("UserSubscription", back_populates="outfit_analyses")

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "outfit_id": self.outfit_id,
            "user_id": self.user_id,
            "analysis_type": self.analysis_type,
            "improvement_explanation": self.improvement_explanation,
            "styling_tips": self.styling_tips,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "user_season": self.user_season,
            "body_shape": self.body_shape,
        }
