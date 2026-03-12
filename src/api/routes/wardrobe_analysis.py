"""
Wardrobe Analysis API Routes
Endpoints for wardrobe gap analysis, outfit improvement, and removal impact.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from src.core.models import (
    Garment,
    UserContext,
    Occasion,
    WardrobeAnalysisResult,
    OutfitImprovementResult,
    RemovalImpact,
    SmartRemovalVerdict,
    UserRemovalGoal,
)
from src.layer2_style.wardrobe_analyzer import WardrobeAnalyzer
from src.layer2_style.outfit_improver import OutfitImprover
from src.layer2_style.smart_removal_analyzer import SmartRemovalAnalyzer
from src.core import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Lazy-loaded service instances
_wardrobe_analyzer: Optional[WardrobeAnalyzer] = None
_outfit_improver: Optional[OutfitImprover] = None
_smart_removal_analyzer: Optional[SmartRemovalAnalyzer] = None


def get_wardrobe_analyzer() -> WardrobeAnalyzer:
    """Get or create WardrobeAnalyzer instance."""
    global _wardrobe_analyzer
    if _wardrobe_analyzer is None:
        _wardrobe_analyzer = WardrobeAnalyzer()
    return _wardrobe_analyzer


def get_outfit_improver() -> OutfitImprover:
    """Get or create OutfitImprover instance."""
    global _outfit_improver
    if _outfit_improver is None:
        _outfit_improver = OutfitImprover()
    return _outfit_improver


def get_smart_removal_analyzer() -> SmartRemovalAnalyzer:
    """Get or create SmartRemovalAnalyzer instance."""
    global _smart_removal_analyzer
    if _smart_removal_analyzer is None:
        _smart_removal_analyzer = SmartRemovalAnalyzer()
    return _smart_removal_analyzer


# ==================== Request / Response Models ====================

class WardrobeAnalysisRequest(BaseModel):
    """Request for full wardrobe analysis."""
    wardrobe: List[Garment] = Field(..., description="All garments in the wardrobe")
    context: Optional[UserContext] = Field(None, description="User context for personalized analysis")
    occasions: Optional[List[str]] = Field(
        None,
        description="Occasions to check coverage for (e.g. 'daily', 'work', 'date'). Defaults to common ones.",
    )
    top_k_versatile: int = Field(default=5, ge=1, le=20, description="Number of top versatile items to return")


class OutfitImprovementRequest(BaseModel):
    """Request for outfit improvement analysis."""
    outfit_garments: List[Garment] = Field(..., description="Garments in the outfit to improve")
    wardrobe: List[Garment] = Field(default_factory=list, description="Full wardrobe for finding alternatives")
    context: Optional[UserContext] = Field(None, description="User context for personalized scoring")
    profile: Optional[str] = Field(None, description="Scoring profile: default, minimalist, creative, business, casual")


class RemovalImpactRequest(BaseModel):
    """Request for garment removal impact analysis."""
    garment_id: str = Field(..., description="ID of the garment to simulate removing")
    wardrobe: List[Garment] = Field(..., description="Full wardrobe including the garment to remove")
    context: Optional[UserContext] = Field(None, description="User context")
    profile: Optional[str] = Field(None, description="Scoring profile")


class SimulateAdditionRequest(BaseModel):
    """Request for simulating a garment addition."""
    wardrobe: List[Garment] = Field(..., description="Current wardrobe")
    virtual_garment: Garment = Field(..., description="The garment to simulate adding")
    context: Optional[UserContext] = Field(None, description="User context")


class SmartRemovalRequest(BaseModel):
    """Request for smart removal analysis."""
    garment_id: Optional[str] = Field(None, description="Specific garment to analyse (omit for full wardrobe)")
    wardrobe: List[Garment] = Field(..., description="Full wardrobe")
    context: Optional[UserContext] = Field(None, description="User context")
    user_goal: UserRemovalGoal = Field(
        default=UserRemovalGoal.MAXIMIZE_OPTIONS,
        description="User goal: minimalist, maximize_options, style_upgrade",
    )


# ==================== Endpoints ====================

@router.post("/analyze", response_model=WardrobeAnalysisResult)
async def analyze_wardrobe(request: WardrobeAnalysisRequest):
    """
    Perform a complete wardrobe analysis.

    Returns distribution, gaps, occasion coverage, versatility scores, and purchase suggestions.
    """
    try:
        analyzer = get_wardrobe_analyzer()

        # Parse occasions
        occasions = None
        if request.occasions:
            parsed = []
            for occ_str in request.occasions:
                try:
                    parsed.append(Occasion(occ_str.lower()))
                except ValueError:
                    logger.warning(f"Unknown occasion '{occ_str}', skipping")
            if parsed:
                occasions = parsed

        result = analyzer.analyze(
            wardrobe=request.wardrobe,
            context=request.context,
            occasions=occasions,
            top_k_versatile=request.top_k_versatile,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Wardrobe analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Wardrobe analysis failed: {str(e)}")


@router.post("/improve", response_model=OutfitImprovementResult)
async def improve_outfit(request: OutfitImprovementRequest):
    """
    Analyze an outfit and suggest improvements.

    Returns diagnosis, addition/replacement suggestions, and purchase recommendations.
    """
    try:
        improver = get_outfit_improver()

        result = improver.improve(
            garments=request.outfit_garments,
            wardrobe=request.wardrobe,
            context=request.context,
            profile=request.profile,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Outfit improvement error: {e}")
        raise HTTPException(status_code=500, detail=f"Outfit improvement failed: {str(e)}")


@router.post("/impact-removal", response_model=RemovalImpact)
async def analyze_removal_impact(request: RemovalImpactRequest):
    """
    Analyze the impact of removing a garment from the wardrobe.

    Returns affected outfit count, versatility score, and replacement options.
    """
    try:
        improver = get_outfit_improver()

        result = improver.analyze_removal_impact(
            garment_id=request.garment_id,
            wardrobe=request.wardrobe,
            context=request.context,
            profile=request.profile,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Removal impact analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Removal impact analysis failed: {str(e)}")


@router.post("/simulate-addition")
async def simulate_addition(request: SimulateAdditionRequest):
    """
    Simulate adding a garment to the wardrobe and measure the impact.

    Returns before/after metrics, versatility score, and recommendation.
    """
    try:
        analyzer = get_wardrobe_analyzer()

        result = analyzer.simulate_addition(
            wardrobe=request.wardrobe,
            virtual_garment=request.virtual_garment,
            context=request.context,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Simulate addition error: {e}")
        raise HTTPException(status_code=500, detail=f"Simulate addition failed: {str(e)}")


@router.post("/smart-removal", response_model=SmartRemovalVerdict)
async def smart_removal_single(request: SmartRemovalRequest):
    """
    Smart removal analysis for a single garment.

    Returns a SmartRemovalVerdict with continuous regret-risk score,
    replacement quality, explainable reasons, and confidence.
    """
    if not request.garment_id:
        raise HTTPException(status_code=400, detail="garment_id is required for single-item analysis")
    try:
        analyzer = get_smart_removal_analyzer()
        result = analyzer.analyze(
            garment_id=request.garment_id,
            wardrobe=request.wardrobe,
            context=request.context,
            user_goal=request.user_goal,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Smart removal analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Smart removal analysis failed: {str(e)}")


@router.post("/smart-removal/wardrobe", response_model=List[SmartRemovalVerdict])
async def smart_removal_wardrobe(request: SmartRemovalRequest):
    """
    Smart removal analysis for every garment in the wardrobe.

    Returns a list of SmartRemovalVerdict sorted by regret_risk ascending
    (safest to remove first).
    """
    try:
        analyzer = get_smart_removal_analyzer()
        results = analyzer.analyze_wardrobe(
            wardrobe=request.wardrobe,
            context=request.context,
            user_goal=request.user_goal,
        )
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Smart removal wardrobe analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Smart removal wardrobe analysis failed: {str(e)}")
