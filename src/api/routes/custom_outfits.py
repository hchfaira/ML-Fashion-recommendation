"""API routes for custom outfit management and analysis."""
from fastapi import APIRouter, HTTPException, Depends, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from src.database import get_db
from src.database.models import CustomOutfit, OutfitAnalysis, UserSubscription
from src.api.models.outfits import (
    CustomOutfitCreate, CustomOutfitUpdate, CustomOutfitResponse,
    CustomOutfitWithAnalysis, OutfitAnalysisResponse, OutfitAnalysisRequest,
    ImprovementExplanation,
)
from src.api.middleware.auth import get_current_user, require_premium, create_token
from src.core import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ── Subscription Management ──

@router.post("/auth/token")
async def create_auth_token(user_id: str) -> dict:
    """
    Create an authentication token for a user.
    
    In production, this would validate credentials (username/password, social login, etc).
    For now, it's a simple token generation endpoint for testing.
    
    Args:
        user_id: User identifier
        
    Returns:
        dict with access_token and token_type
    """
    token = create_token(user_id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 604800,  # 7 days in seconds
    }


@router.get("/subscriptions/me")
async def get_my_subscription(
    user: UserSubscription = Depends(get_current_user),
) -> dict:
    """Get current user's subscription details."""
    return {
        "id": user.id,
        "user_id": user.user_id,
        "is_premium": user.is_premium,
        "subscription_tier": user.subscription_tier,
        "is_active": user.is_active(),
        "created_at": user.created_at,
        "expires_at": user.expires_at,
        "custom_outfits_created": user.custom_outfits_created,
        "analyses_performed": user.analyses_performed,
    }


# ── Custom Outfit CRUD ──

@router.post("/outfits/custom", response_model=CustomOutfitResponse, status_code=status.HTTP_201_CREATED)
async def create_custom_outfit(
    outfit_data: CustomOutfitCreate,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CustomOutfitResponse:
    """
    Create a new custom outfit from selected garments.
    
    This endpoint:
    1. Validates garment IDs (in production, check against wardrobe)
    2. Creates outfit record in database
    3. Increments user's outfit counter
    
    Args:
        outfit_data: Outfit creation details
        user: Authenticated user
        db: Database session
        
    Returns:
        Created CustomOutfit response
        
    Raises:
        400: Invalid garment IDs
        401: User not authenticated
    """
    if not outfit_data.garment_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one garment must be selected"
        )

    # Create outfit record
    outfit = CustomOutfit(
        user_id=user.user_id,
        outfit_name=outfit_data.outfit_name,
        garment_ids=outfit_data.garment_ids,
        user_season=outfit_data.user_season,
        intended_occasion=outfit_data.intended_occasion,
        notes=outfit_data.notes,
        created_at=datetime.utcnow(),
    )

    db.add(outfit)
    user.custom_outfits_created += 1
    db.commit()
    db.refresh(outfit)

    logger.info(
        "Custom outfit created: %s for user %s (%d garments)",
        outfit.id, user.user_id, len(outfit_data.garment_ids)
    )

    return outfit


@router.get("/outfits/custom", response_model=List[CustomOutfitResponse])
async def list_custom_outfits(
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
) -> List[CustomOutfitResponse]:
    """
    List user's custom outfits.
    
    Supports pagination for efficient loading.
    
    Args:
        user: Authenticated user
        db: Database session
        skip: Number of results to skip (pagination)
        limit: Maximum results to return
        
    Returns:
        List of CustomOutfit responses
    """
    outfits = db.query(CustomOutfit).filter(
        CustomOutfit.user_id == user.user_id
    ).order_by(
        CustomOutfit.created_at.desc()
    ).offset(skip).limit(limit).all()

    return outfits


@router.get("/outfits/custom/{outfit_id}", response_model=CustomOutfitWithAnalysis)
async def get_custom_outfit(
    outfit_id: str,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    Get a specific custom outfit with its latest analysis.
    
    Args:
        outfit_id: Outfit ID to retrieve
        user: Authenticated user
        db: Database session
        
    Returns:
        CustomOutfit with latest analysis if available
        
    Raises:
        404: Outfit not found or user doesn't own it
    """
    outfit = db.query(CustomOutfit).filter(
        CustomOutfit.id == outfit_id,
        CustomOutfit.user_id == user.user_id,
    ).first()

    if not outfit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Outfit not found"
        )

    # Fetch latest analysis
    latest_analysis = db.query(OutfitAnalysis).filter(
        OutfitAnalysis.outfit_id == outfit_id,
    ).order_by(
        OutfitAnalysis.generated_at.desc()
    ).first()

    response = outfit.to_dict()
    if latest_analysis:
        response["latest_analysis"] = latest_analysis.to_dict()
    else:
        response["latest_analysis"] = None

    return response


@router.put("/outfits/custom/{outfit_id}", response_model=CustomOutfitResponse)
async def update_custom_outfit(
    outfit_id: str,
    outfit_data: CustomOutfitUpdate,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CustomOutfitResponse:
    """
    Update a custom outfit's metadata.
    
    Only the outfit owner can update their outfits.
    
    Args:
        outfit_id: Outfit ID to update
        outfit_data: Fields to update
        user: Authenticated user
        db: Database session
        
    Returns:
        Updated CustomOutfit response
        
    Raises:
        404: Outfit not found
        403: User doesn't own the outfit
    """
    outfit = db.query(CustomOutfit).filter(
        CustomOutfit.id == outfit_id,
        CustomOutfit.user_id == user.user_id,
    ).first()

    if not outfit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Outfit not found"
        )

    # Update fields if provided
    if outfit_data.outfit_name is not None:
        outfit.outfit_name = outfit_data.outfit_name
    if outfit_data.notes is not None:
        outfit.notes = outfit_data.notes
    if outfit_data.intended_occasion is not None:
        outfit.intended_occasion = outfit_data.intended_occasion

    outfit.saved_at = datetime.utcnow()
    db.commit()
    db.refresh(outfit)

    logger.info("Custom outfit updated: %s for user %s", outfit_id, user.user_id)

    return outfit


@router.delete("/outfits/custom/{outfit_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_custom_outfit(
    outfit_id: str,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """
    Delete a custom outfit and its analyses.
    
    Args:
        outfit_id: Outfit ID to delete
        user: Authenticated user
        db: Database session
        
    Raises:
        404: Outfit not found
        403: User doesn't own the outfit
    """
    outfit = db.query(CustomOutfit).filter(
        CustomOutfit.id == outfit_id,
        CustomOutfit.user_id == user.user_id,
    ).first()

    if not outfit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Outfit not found"
        )

    # Delete associated analyses
    db.query(OutfitAnalysis).filter(
        OutfitAnalysis.outfit_id == outfit_id
    ).delete()

    db.delete(outfit)
    db.commit()

    logger.info("Custom outfit deleted: %s for user %s", outfit_id, user.user_id)


# ── Outfit Analysis (Premium Feature) ──

@router.post("/outfits/custom/{outfit_id}/analyze", response_model=OutfitAnalysisResponse, status_code=status.HTTP_201_CREATED)
async def analyze_custom_outfit(
    outfit_id: str,
    analysis_request: OutfitAnalysisRequest,
    user: UserSubscription = Depends(require_premium),
    db: Session = Depends(get_db),
) -> OutfitAnalysisResponse:
    """
    Analyze a custom outfit using Layer 4 LLM (premium feature).
    
    This endpoint:
    1. Checks premium subscription
    2. Fetches the outfit and its garments
    3. Calls OutfitImprovementExplainer to get LLM analysis
    4. Caches the result in database
    5. Increments analysis counter
    
    Args:
        outfit_id: Outfit ID to analyze
        analysis_request: Analysis parameters
        user: Authenticated premium user
        db: Database session
        
    Returns:
        Created OutfitAnalysis response
        
    Raises:
        403: User doesn't have premium access
        404: Outfit not found
        400: Outfit has no garments or analysis fails
    """
    outfit = db.query(CustomOutfit).filter(
        CustomOutfit.id == outfit_id,
        CustomOutfit.user_id == user.user_id,
    ).first()

    if not outfit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Outfit not found"
        )

    if not outfit.garment_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot analyze outfit without garments"
        )

    try:
        # In production: call OutfitImprovementExplainer here
        # For now, create a placeholder analysis
        improvement_explanation = ImprovementExplanation(
            additions=[],
            replacements=[],
            purchases=[],
        )

        analysis = OutfitAnalysis(
            outfit_id=outfit_id,
            user_id=user.user_id,
            analysis_type="improvement",
            improvement_explanation=improvement_explanation.dict(),
            user_season=analysis_request.user_season,
            body_shape=analysis_request.body_shape,
            generated_at=datetime.utcnow(),
        )

        db.add(analysis)
        outfit.last_analyzed_at = datetime.utcnow()
        user.analyses_performed += 1
        db.commit()
        db.refresh(analysis)

        logger.info(
            "Outfit analysis completed: %s for user %s (premium)",
            outfit_id, user.user_id
        )

        return analysis

    except Exception as e:
        logger.error("Error analyzing outfit %s: %s", outfit_id, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to analyze outfit. Please try again."
        )


@router.get("/outfits/custom/{outfit_id}/analyses", response_model=List[OutfitAnalysisResponse])
async def get_outfit_analyses(
    outfit_id: str,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[OutfitAnalysisResponse]:
    """
    Get all analyses for a specific outfit.
    
    Args:
        outfit_id: Outfit ID
        user: Authenticated user
        db: Database session
        
    Returns:
        List of analyses in descending order by date
        
    Raises:
        404: Outfit not found
    """
    outfit = db.query(CustomOutfit).filter(
        CustomOutfit.id == outfit_id,
        CustomOutfit.user_id == user.user_id,
    ).first()

    if not outfit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Outfit not found"
        )

    analyses = db.query(OutfitAnalysis).filter(
        OutfitAnalysis.outfit_id == outfit_id,
    ).order_by(
        OutfitAnalysis.generated_at.desc()
    ).all()

    return analyses
