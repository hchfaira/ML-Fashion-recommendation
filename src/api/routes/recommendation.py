"""
Recommendation API Routes
Main endpoints for outfit recommendations.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional

from src.core.models import (
    RecommendationRequest, RecommendationResponse, 
    Outfit, Garment, UserContext
)
from src.layer1_vision import VisionService
from src.layer2_style import StyleIntelligenceModel
from src.layer3_context import ContextEngine
from src.layer4_llm import OutfitExplainer
from src.core import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Lazy-loaded service instances (initialized on first use)
_style_model = None
_context_engine = None
_outfit_explainer = None


def get_style_model() -> StyleIntelligenceModel:
    """Get or create StyleIntelligenceModel instance."""
    global _style_model
    if _style_model is None:
        _style_model = StyleIntelligenceModel()
    return _style_model


def get_context_engine() -> ContextEngine:
    """Get or create ContextEngine instance."""
    global _context_engine
    if _context_engine is None:
        _context_engine = ContextEngine()
    return _context_engine


def get_outfit_explainer() -> OutfitExplainer:
    """Get or create OutfitExplainer instance."""
    global _outfit_explainer
    if _outfit_explainer is None:
        _outfit_explainer = OutfitExplainer()
    return _outfit_explainer


# For backward compatibility with tests that mock these names
style_model = None  # Use get_style_model() instead
context_engine = None  # Use get_context_engine() instead
outfit_explainer = None  # Use get_outfit_explainer() instead


@router.post("/outfit", response_model=RecommendationResponse)
async def get_outfit_recommendations(request: RecommendationRequest):
    """
    Get outfit recommendations based on wardrobe and context.
    
    This endpoint orchestrates all 4 layers:
    1. Uses embeddings from Layer 1 for similarity
    2. Scores outfits with Layer 2 style intelligence
    3. Applies context filtering from Layer 3
    4. Generates explanations with Layer 4
    """
    import time
    start_time = time.time()
    
    try:
        # Filter wardrobe by context
        ctx_engine = get_context_engine()
        available_items = await ctx_engine.filter_wardrobe_by_context(
            request.wardrobe_items,
            request.context
        )
        
        if len(available_items) < 2:
            raise HTTPException(
                status_code=400,
                detail="Not enough suitable items in wardrobe for recommendations"
            )
        
        # Generate outfit candidates
        model = get_style_model()
        outfits = []
        for _ in range(request.num_recommendations * 2):  # Generate extra for filtering
            outfit = await model.generate_outfit(
                wardrobe=available_items,
                anchor_item=request.base_item
            )
            outfits.append(outfit)
        
        # Apply context scoring
        scored_outfits = await ctx_engine.apply_context(
            outfits, request.context
        )
        
        # Take top N
        top_outfits = scored_outfits[:request.num_recommendations]
        
        # Generate explanations
        explainer = get_outfit_explainer()
        for outfit in top_outfits:
            explanation = await explainer.explain(
                outfit, request.context, detail_level="brief"
            )
            outfit.explanation = explanation.summary
        
        processing_time = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            recommendations=top_outfits,
            processing_time_ms=processing_time,
            context_applied={
                "occasion": request.context.occasion.value if request.context.occasion else None,
                "weather_considered": request.context.weather is not None,
                "items_filtered": len(request.wardrobe_items) - len(available_items)
            }
        )
        
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/match/{garment_id}")
async def find_matching_items(
    garment_id: str,
    wardrobe: List[Garment],
    top_k: int = 5
):
    """
    Find items that match well with a specific garment.
    """
    # Find the base garment
    base_garment = None
    for g in wardrobe:
        if g.id == garment_id:
            base_garment = g
            break
    
    if not base_garment:
        raise HTTPException(status_code=404, detail="Garment not found")
    
    # Find matches
    model = get_style_model()
    matches = await model.find_best_match(
        base_item=base_garment,
        candidates=[g for g in wardrobe if g.id != garment_id],
        top_k=top_k
    )
    
    return {
        "base_item": base_garment,
        "matches": [
            {"garment": g, "compatibility_score": score}
            for g, score in matches
        ]
    }


@router.post("/score")
async def score_outfit(items: List[Garment]):
    """
    Score a specific outfit combination.
    """
    if len(items) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 items to score"
        )
    
    model = get_style_model()
    scores = await model.score_outfit(items)
    
    return {
        "overall_score": scores["overall_score"],
        "breakdown": scores["breakdown"],
        "suggestions": scores.get("suggestions", [])
    }
