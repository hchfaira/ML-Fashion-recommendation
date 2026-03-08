"""
Scoring API Routes
Advanced outfit scoring with all Layer 2 style intelligence features.

Features:
- Profile-based scoring configuration
- Dynamic criteria weighting
- Filtered score output based on profile settings
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from src.core.models import Garment, GarmentAttributes, Occasion, UserContext
from src.layer2_style import (
    OutfitScorecard,
    OutfitBuilder,
    TotalStyleScorer,
    SevenPointRuleScorer,
    SeasonColorHarmonyScorer,
    ProportionScorer,
    VolumeBalanceScorer,
    PatternMixingScorer,
    DesignPrinciplesScorer,
    CreativityScorer,
    SkinContrastScorer,
    SandwichRuleScorer,
    ThreeColorScorer,
    OccasionScorer,
    ColorSeason,
    SkinUndertone,
    BodyShape,
    # Scoring configuration
    ScoringConfigService,
    ScoringProfile,
    CriterionConfig,
    get_scoring_config_service,
    reload_scoring_config,
)
from src.core import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Lazy-loaded scorers
_total_style_scorer = None
_seven_point_scorer = None
_season_color_scorer = None
_proportion_scorer = None
_volume_balance_scorer = None
_pattern_mixing_scorer = None
_design_principles_scorer = None
_creativity_scorer = None
_skin_contrast_scorer = None
_sandwich_rule_scorer = None
_three_color_scorer = None
_occasion_scorer = None


def get_total_style_scorer() -> TotalStyleScorer:
    global _total_style_scorer
    if _total_style_scorer is None:
        _total_style_scorer = TotalStyleScorer()
    return _total_style_scorer


def get_seven_point_scorer() -> SevenPointRuleScorer:
    global _seven_point_scorer
    if _seven_point_scorer is None:
        _seven_point_scorer = SevenPointRuleScorer()
    return _seven_point_scorer


def get_season_color_scorer() -> SeasonColorHarmonyScorer:
    global _season_color_scorer
    if _season_color_scorer is None:
        _season_color_scorer = SeasonColorHarmonyScorer()
    return _season_color_scorer


def get_proportion_scorer() -> ProportionScorer:
    global _proportion_scorer
    if _proportion_scorer is None:
        _proportion_scorer = ProportionScorer()
    return _proportion_scorer


def get_volume_balance_scorer() -> VolumeBalanceScorer:
    global _volume_balance_scorer
    if _volume_balance_scorer is None:
        _volume_balance_scorer = VolumeBalanceScorer()
    return _volume_balance_scorer


def get_pattern_mixing_scorer() -> PatternMixingScorer:
    global _pattern_mixing_scorer
    if _pattern_mixing_scorer is None:
        _pattern_mixing_scorer = PatternMixingScorer()
    return _pattern_mixing_scorer


def get_design_principles_scorer() -> DesignPrinciplesScorer:
    global _design_principles_scorer
    if _design_principles_scorer is None:
        _design_principles_scorer = DesignPrinciplesScorer()
    return _design_principles_scorer


def get_creativity_scorer() -> CreativityScorer:
    global _creativity_scorer
    if _creativity_scorer is None:
        _creativity_scorer = CreativityScorer()
    return _creativity_scorer


def get_skin_contrast_scorer() -> SkinContrastScorer:
    global _skin_contrast_scorer
    if _skin_contrast_scorer is None:
        _skin_contrast_scorer = SkinContrastScorer()
    return _skin_contrast_scorer


def get_sandwich_rule_scorer() -> SandwichRuleScorer:
    global _sandwich_rule_scorer
    if _sandwich_rule_scorer is None:
        _sandwich_rule_scorer = SandwichRuleScorer()
    return _sandwich_rule_scorer


def get_three_color_scorer() -> ThreeColorScorer:
    global _three_color_scorer
    if _three_color_scorer is None:
        _three_color_scorer = ThreeColorScorer()
    return _three_color_scorer


def get_occasion_scorer() -> OccasionScorer:
    global _occasion_scorer
    if _occasion_scorer is None:
        _occasion_scorer = OccasionScorer()
    return _occasion_scorer


# ============== Request/Response Models ==============

class ScorecardRequest(BaseModel):
    """Request for outfit scorecard."""
    garments: List[Garment]
    occasion: Optional[str] = None
    user_skin_undertone: Optional[str] = None
    user_color_season: Optional[str] = None
    user_body_shape: Optional[str] = None
    profile: Optional[str] = Field(
        default=None,
        description="Scoring profile to use (default, minimalist, creative, business, casual). "
                    "Controls which criteria are calculated and their weights."
    )
    include_all_scores: bool = Field(
        default=False,
        description="If True, include all scores even those disabled by profile"
    )


class OutfitBuildRequest(BaseModel):
    """Request for building outfits from wardrobe."""
    wardrobe: List[Garment]
    occasion: Optional[str] = None
    max_outfits: int = Field(default=5, ge=1, le=20)
    anchor_item_id: Optional[str] = None
    style_preference: Optional[str] = None


class SevenPointRequest(BaseModel):
    """Request for 7-point rule analysis."""
    garments: List[Garment]
    accessories: Optional[List[str]] = None


class SeasonColorRequest(BaseModel):
    """Request for season color harmony analysis."""
    garments: List[Garment]
    skin_undertone: str  # "warm", "cool", "neutral"
    color_season: str  # "spring", "summer", "autumn", "winter"


class PatternMixingRequest(BaseModel):
    """Request for pattern mixing analysis."""
    garments: List[Garment]


class DesignPrinciplesRequest(BaseModel):
    """Request for design principles analysis."""
    garments: List[Garment]
    body_shape: Optional[str] = None


class CreativityRequest(BaseModel):
    """Request for creativity scoring."""
    garments: List[Garment]
    style_aesthetic: Optional[str] = None  # "classic", "trendy", "avant-garde"


# ============== Profile Management Endpoints ==============

@router.get("/profiles")
async def list_scoring_profiles():
    """
    List all available scoring profiles.
    
    Returns all profile names with their descriptions and
    summary of enabled criteria.
    """
    try:
        config_service = get_scoring_config_service()
        profiles = config_service.list_profiles()
        
        result = []
        for profile_name in profiles:
            profile = config_service.get_profile(profile_name)
            enabled_criteria = [
                name for name, cfg in profile.criteria.items()
                if cfg.enabled
            ]
            result.append({
                "name": profile_name,
                "description": profile.description,
                "enabled_criteria": enabled_criteria,
                "criteria_count": len(enabled_criteria)
            })
        
        return {
            "profiles": result,
            "total": len(result)
        }
        
    except Exception as e:
        logger.error(f"Failed to list profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/{profile_name}")
async def get_scoring_profile(profile_name: str):
    """
    Get detailed configuration for a specific scoring profile.
    
    Returns all criteria settings including:
    - Whether each criterion is enabled
    - Weight for each criterion
    - Display settings (show_in_summary, show_details)
    - Display names and descriptions
    """
    try:
        config_service = get_scoring_config_service()
        profile = config_service.get_profile(profile_name)
        
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{profile_name}' not found. "
                       f"Available: {config_service.list_profiles()}"
            )
        
        # Build detailed response
        criteria_details = {}
        for name, cfg in profile.criteria.items():
            criteria_details[name] = {
                "enabled": cfg.enabled,
                "weight": cfg.weight,
                "normalized_weight": None,  # Will be calculated
                "display_name": cfg.display_name,
                "description": cfg.description,
                "show_in_summary": cfg.show_in_summary,
                "show_details": cfg.show_details
            }
        
        # Calculate normalized weights
        normalized = config_service.normalize_weights(profile_name)
        for name, norm_weight in normalized.items():
            if name in criteria_details:
                criteria_details[name]["normalized_weight"] = round(norm_weight, 4)
        
        return {
            "name": profile_name,
            "description": profile.description,
            "criteria": criteria_details,
            "total_weight": sum(
                cfg.weight for cfg in profile.criteria.values() if cfg.enabled
            ),
            "enabled_count": sum(1 for cfg in profile.criteria.values() if cfg.enabled)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile {profile_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/{profile_name}/weights")
async def get_profile_weights(
    profile_name: str,
    normalized: bool = Query(
        default=True,
        description="Return normalized weights (sum to 1.0)"
    )
):
    """
    Get the weights for each criterion in a profile.
    
    Useful for understanding how the final score is calculated.
    """
    try:
        config_service = get_scoring_config_service()
        profile = config_service.get_profile(profile_name)
        
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{profile_name}' not found"
            )
        
        if normalized:
            weights = config_service.normalize_weights(profile_name)
            weights = {k: round(v, 4) for k, v in weights.items()}
        else:
            weights = {
                name: cfg.weight
                for name, cfg in profile.criteria.items()
                if cfg.enabled
            }
        
        return {
            "profile": profile_name,
            "normalized": normalized,
            "weights": weights,
            "total": round(sum(weights.values()), 4)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get weights for {profile_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profiles/reload")
async def reload_profiles():
    """
    Reload scoring profiles from configuration file.
    
    Use this endpoint after modifying the scoring_config.json file
    to apply changes without restarting the server.
    """
    try:
        config_service = reload_scoring_config()
        profiles = config_service.list_profiles()
        
        return {
            "status": "success",
            "message": "Scoring profiles reloaded successfully",
            "available_profiles": profiles
        }
        
    except Exception as e:
        logger.error(f"Failed to reload profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Scoring Endpoints ==============

@router.post("/scorecard")
async def get_outfit_scorecard(request: ScorecardRequest):
    """
    Get comprehensive outfit scorecard with all scoring dimensions.
    
    This is the main endpoint for detailed outfit analysis combining:
    - Color harmony
    - Formality matching
    - Silhouette balance
    - Pattern compatibility
    - Occasion appropriateness
    - And more...
    
    Use the `profile` parameter to select a scoring profile:
    - **default**: Full analysis with all criteria
    - **minimalist**: Focus on essential rules (7-point, color harmony)
    - **creative**: Emphasizes creativity and pattern mixing
    - **business**: Professional dress code focus
    - **casual**: Relaxed everyday styling
    
    Returns a complete breakdown of scores and recommendations,
    filtered according to the selected profile's settings.
    """
    if len(request.garments) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 garments to score"
        )
    
    try:
        # Create scorecard with profile
        scorecard = OutfitScorecard(
            garments=request.garments,
            profile=request.profile
        )
        
        # Set user context if provided
        if request.occasion:
            try:
                scorecard.occasion = Occasion(request.occasion)
            except ValueError:
                pass
        
        # Calculate all scores (respects profile)
        result = scorecard.calculate_all_scores()
        
        return result.to_json(include_all_scores=request.include_all_scores)
        
    except Exception as e:
        logger.error(f"Scorecard calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/build-outfits")
async def build_outfits(request: OutfitBuildRequest):
    """
    Automatically build optimal outfits from a wardrobe.
    
    Uses OutfitBuilder to generate outfit combinations
    ranked by compatibility and style scores.
    """
    if len(request.wardrobe) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 garments in wardrobe"
        )
    
    try:
        builder = OutfitBuilder(wardrobe=request.wardrobe)
        
        # Find anchor item if specified
        anchor = None
        if request.anchor_item_id:
            for g in request.wardrobe:
                if g.id == request.anchor_item_id:
                    anchor = g
                    break
        
        # Build outfits
        candidates = builder.build_outfits(
            anchor_item=anchor,
            max_outfits=request.max_outfits
        )
        
        return {
            "outfits": [
                {
                    "garments": [g.id for g in c.garments],
                    "score": c.total_score,
                    "breakdown": c.score_breakdown
                }
                for c in candidates
            ],
            "total_generated": len(candidates)
        }
        
    except Exception as e:
        logger.error(f"Outfit building failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/total-style")
async def get_total_style_score(garments: List[Garment]):
    """
    Get complete style report with grade (A+ to F).
    
    Combines all scoring rules into a single comprehensive grade.
    """
    if len(garments) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 garments"
        )
    
    try:
        scorer = get_total_style_scorer()
        report = scorer.score_outfit(garments)
        
        return {
            "grade": report.grade,
            "total_score": report.total_score,
            "breakdown": report.breakdown.model_dump() if hasattr(report.breakdown, 'model_dump') else report.breakdown,
            "strengths": report.strengths,
            "improvements": report.improvements,
            "summary": report.summary
        }
        
    except Exception as e:
        logger.error(f"Total style scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/seven-point-rule")
async def check_seven_point_rule(request: SevenPointRequest):
    """
    Analyze outfit against the 7-point rule.
    
    The 7-point rule suggests optimal accessorizing
    for a balanced, polished look.
    """
    try:
        scorer = get_seven_point_scorer()
        result = scorer.score(request.garments, accessories=request.accessories)
        
        return {
            "score": result.score,
            "current_points": result.current_points,
            "target_points": result.target_points,
            "is_balanced": result.is_balanced,
            "suggestions": result.suggestions
        }
        
    except Exception as e:
        logger.error(f"Seven point rule check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/season-color-harmony")
async def check_season_color_harmony(request: SeasonColorRequest):
    """
    Check if outfit colors match user's color season and skin undertone.
    
    Based on seasonal color analysis theory.
    """
    try:
        scorer = get_season_color_scorer()
        
        undertone = SkinUndertone(request.skin_undertone)
        season = ColorSeason(request.color_season)
        
        result = scorer.score(
            request.garments,
            skin_undertone=undertone,
            color_season=season
        )
        
        return {
            "score": result.score,
            "harmony_level": result.harmony_level,
            "matching_colors": result.matching_colors,
            "clashing_colors": result.clashing_colors,
            "recommendations": result.recommendations
        }
        
    except Exception as e:
        logger.error(f"Season color harmony check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proportion-analysis")
async def analyze_proportions(garments: List[Garment], body_shape: Optional[str] = None):
    """
    Analyze outfit proportions (Rule of Thirds).
    
    Evaluates visual balance and proportion ratios.
    """
    try:
        scorer = get_proportion_scorer()
        
        shape = BodyShape(body_shape) if body_shape else None
        result = scorer.score(garments, body_shape=shape)
        
        return {
            "score": result.score,
            "ratio": result.ratio.value if hasattr(result.ratio, 'value') else result.ratio,
            "is_balanced": result.is_balanced,
            "visual_weight_distribution": result.weight_distribution,
            "suggestions": result.suggestions
        }
        
    except Exception as e:
        logger.error(f"Proportion analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/volume-balance")
async def check_volume_balance(garments: List[Garment], body_shape: Optional[str] = None):
    """
    Check volume balance between top and bottom.
    
    Ensures silhouette harmony for different body shapes.
    """
    try:
        scorer = get_volume_balance_scorer()
        
        shape = BodyShape(body_shape) if body_shape else None
        result = scorer.score(garments, body_shape=shape)
        
        return {
            "score": result.score,
            "top_volume": result.top_volume.value if hasattr(result.top_volume, 'value') else result.top_volume,
            "bottom_volume": result.bottom_volume.value if hasattr(result.bottom_volume, 'value') else result.bottom_volume,
            "is_balanced": result.is_balanced,
            "recommended_for_body_shape": result.recommendations
        }
        
    except Exception as e:
        logger.error(f"Volume balance check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pattern-mixing")
async def analyze_pattern_mixing(request: PatternMixingRequest):
    """
    Analyze pattern mixing compatibility.
    
    Evaluates if patterns in outfit work well together
    based on scale, family, and density rules.
    """
    try:
        scorer = get_pattern_mixing_scorer()
        result = scorer.score(request.garments)
        
        return {
            "score": result.score,
            "pattern_count": result.pattern_count,
            "is_compatible": result.is_compatible,
            "patterns_found": [p.model_dump() if hasattr(p, 'model_dump') else p for p in result.patterns],
            "mixing_rules_followed": result.rules_followed,
            "suggestions": result.suggestions
        }
        
    except Exception as e:
        logger.error(f"Pattern mixing analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/design-principles")
async def analyze_design_principles(request: DesignPrinciplesRequest):
    """
    Analyze outfit against fashion design principles.
    
    Evaluates:
    - Balance (symmetrical/asymmetrical)
    - Proportion (golden ratio)
    - Rhythm (repetition, gradation)
    - Emphasis (focal point)
    - Harmony (unity)
    - Line direction
    """
    try:
        scorer = get_design_principles_scorer()
        
        shape = BodyShape(request.body_shape) if request.body_shape else None
        result = scorer.score(request.garments, body_shape=shape)
        
        return {
            "score": result.score,
            "grade": result.grade,
            "balance": result.balance.model_dump() if hasattr(result.balance, 'model_dump') else result.balance,
            "proportion": result.proportion.model_dump() if hasattr(result.proportion, 'model_dump') else result.proportion,
            "rhythm": result.rhythm.model_dump() if hasattr(result.rhythm, 'model_dump') else result.rhythm,
            "emphasis": result.emphasis.model_dump() if hasattr(result.emphasis, 'model_dump') else result.emphasis,
            "harmony": result.harmony.model_dump() if hasattr(result.harmony, 'model_dump') else result.harmony,
            "recommendations": result.recommendations
        }
        
    except Exception as e:
        logger.error(f"Design principles analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/creativity")
async def score_creativity(request: CreativityRequest):
    """
    Score outfit creativity and rule-breaking intelligence.
    
    Identifies intentional style risks that work.
    """
    try:
        scorer = get_creativity_scorer()
        result = scorer.score(request.garments, style_aesthetic=request.style_aesthetic)
        
        return {
            "score": result.score,
            "creativity_level": result.level.value if hasattr(result.level, 'value') else result.level,
            "rule_breaks": [rb.model_dump() if hasattr(rb, 'model_dump') else rb for rb in result.rule_breaks],
            "is_intentional": result.is_intentional,
            "fashion_forward_elements": result.fashion_forward_elements,
            "summary": result.summary
        }
        
    except Exception as e:
        logger.error(f"Creativity scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/three-color-rule")
async def check_three_color_rule(garments: List[Garment]):
    """
    Check adherence to the three-color rule.
    
    Classic rule: limit outfit to 3 main colors for cohesion.
    """
    try:
        scorer = get_three_color_scorer()
        result = scorer.score(garments)
        
        return {
            "score": result.score,
            "color_count": result.color_count,
            "colors_used": result.colors,
            "follows_rule": result.follows_rule,
            "suggestions": result.suggestions
        }
        
    except Exception as e:
        logger.error(f"Three color rule check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sandwich-rule")
async def check_sandwich_rule(garments: List[Garment]):
    """
    Check the color sandwich rule.
    
    Colors should "sandwich" for visual cohesion
    (e.g., same color on top and shoes).
    """
    try:
        scorer = get_sandwich_rule_scorer()
        result = scorer.score(garments)
        
        return {
            "score": result.score,
            "has_sandwich": result.has_sandwich,
            "sandwich_color": result.sandwich_color,
            "connecting_elements": result.connecting_elements,
            "suggestions": result.suggestions
        }
        
    except Exception as e:
        logger.error(f"Sandwich rule check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/occasion-fit")
async def check_occasion_fit(garments: List[Garment], occasion: str):
    """
    Check if outfit is appropriate for occasion.
    
    Evaluates formality level and dress code compliance.
    """
    try:
        scorer = get_occasion_scorer()
        occ = Occasion(occasion)
        result = scorer.score(garments, occasion=occ)
        
        return {
            "score": result.score,
            "is_appropriate": result.is_appropriate,
            "outfit_formality": result.outfit_formality,
            "occasion_formality": result.occasion_formality,
            "formality_gap": result.formality_gap,
            "suggestions": result.suggestions
        }
        
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid occasion. Valid values: {[o.value for o in Occasion]}"
        )
    except Exception as e:
        logger.error(f"Occasion fit check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
