"""SQLAlchemy ORM models for outfit storage and user subscriptions."""
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, ForeignKey, Text, UniqueConstraint
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


# ── Mood Board Feature ──

class SharedOutfit(Base):
    """A outfit shared publicly by a user for others to discover and save."""
    __tablename__ = "shared_outfits"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    owner_user_id = Column(String(36), nullable=False, index=True)

    # Reference to the logical outfit (garment IDs + scores)
    outfit_data = Column(JSON, nullable=False)  # snapshot: garment_ids, scores, attributes

    # Display metadata
    title = Column(String(255), nullable=True)
    occasion_tags = Column(JSON, default=list)   # ["casual", "summer"]
    style_tags = Column(JSON, default=list)       # ["minimalist", "parisian"]
    formality_score = Column(Float, nullable=True)
    dominant_colors = Column(JSON, default=list)  # ["beige", "white"]
    dominant_styles = Column(JSON, default=dict)  # {"minimalist": 0.8}
    embedding_vector = Column(JSON, default=list) # serialized list[float]

    # Social counters (denormalized for query speed)
    likes_count = Column(Integer, default=0)
    saves_count = Column(Integer, default=0)

    is_public = Column(Boolean, default=True)
    shared_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    mood_board_items = relationship("MoodBoardItem", back_populates="shared_outfit")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "owner_user_id": self.owner_user_id,
            "outfit_data": self.outfit_data,
            "title": self.title,
            "occasion_tags": self.occasion_tags or [],
            "style_tags": self.style_tags or [],
            "formality_score": self.formality_score,
            "dominant_colors": self.dominant_colors or [],
            "dominant_styles": self.dominant_styles or {},
            "embedding_vector": self.embedding_vector or [],
            "likes_count": self.likes_count,
            "saves_count": self.saves_count,
            "is_public": self.is_public,
            "shared_at": self.shared_at.isoformat() if self.shared_at else None,
        }


class SharedOutfitLike(Base):
    """Tracks which users liked which shared outfits."""
    __tablename__ = "shared_outfit_likes"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    shared_outfit_id = Column(String(36), ForeignKey("shared_outfits.id"), nullable=False, index=True)
    liked_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "shared_outfit_id", name="uq_user_outfit_like"),)


class MoodBoard(Base):
    """A curated collection of shared outfits saved by a user."""
    __tablename__ = "mood_boards"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=False)  # Only one active board per user
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    items = relationship("MoodBoardItem", back_populates="board", cascade="all, delete-orphan")
    style_profile = relationship("MoodBoardStyleProfile", back_populates="board", uselist=False, cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active,
            "items_count": len(self.items) if self.items else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class MoodBoardItem(Base):
    """A single shared outfit saved inside a mood board."""
    __tablename__ = "mood_board_items"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    board_id = Column(String(36), ForeignKey("mood_boards.id"), nullable=False, index=True)
    shared_outfit_id = Column(String(36), ForeignKey("shared_outfits.id"), nullable=False, index=True)
    personal_note = Column(Text, nullable=True)
    saved_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (UniqueConstraint("board_id", "shared_outfit_id", name="uq_board_outfit"),)

    # Relationships
    board = relationship("MoodBoard", back_populates="items")
    shared_outfit = relationship("SharedOutfit", back_populates="mood_board_items")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "board_id": self.board_id,
            "shared_outfit_id": self.shared_outfit_id,
            "personal_note": self.personal_note,
            "saved_at": self.saved_at.isoformat() if self.saved_at else None,
        }


class MoodBoardStyleProfile(Base):
    """Aggregated style profile computed from all outfits saved in a mood board."""
    __tablename__ = "mood_board_style_profiles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    board_id = Column(String(36), ForeignKey("mood_boards.id"), nullable=False, unique=True, index=True)

    # Aggregated style signals
    dominant_colors = Column(JSON, default=list)   # [{"color": "beige", "frequency": 0.4}, ...]
    dominant_styles = Column(JSON, default=dict)   # {"minimalist": 0.72, "parisian": 0.45}
    formality_average = Column(Float, nullable=True)
    occasions_distribution = Column(JSON, default=dict)  # {"casual": 0.6, "work": 0.3}
    embedding_centroid = Column(JSON, default=list)  # list[float]
    coherence_score = Column(Float, nullable=True)   # 0-1: how homogeneous the board is
    items_count = Column(Integer, default=0)
    is_stale = Column(Boolean, default=True)  # True when board changed and needs recompute
    last_computed_at = Column(DateTime, nullable=True)

    # Relationship
    board = relationship("MoodBoard", back_populates="style_profile")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "board_id": self.board_id,
            "dominant_colors": self.dominant_colors or [],
            "dominant_styles": self.dominant_styles or {},
            "formality_average": self.formality_average,
            "occasions_distribution": self.occasions_distribution or {},
            "coherence_score": self.coherence_score,
            "items_count": self.items_count,
            "is_stale": self.is_stale,
            "last_computed_at": self.last_computed_at.isoformat() if self.last_computed_at else None,
        }
