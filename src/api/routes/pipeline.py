"""
Full Pipeline API Route
========================

Single endpoint that integrates ALL layers of the fashion recommendation system:

Layer 0 - Segmentation:  Extract garments from wardrobe images
Layer 1 - Vision:        Analyze attributes of each garment
Layer 2 - Style:         Score outfit combinations (OutfitBuilder)
Layer 3 - Context:       Apply user profile + context (ContextEngine + StyleProfilePipeline)
Layer 4 - LLM:           Generate natural-language explanations
Layer 5 - Visualization: Create catalogue images of top outfits
Layer 6 - Try-On:        (Optional) Virtual try-on on user photo

Request flow:
    1. (Optional) Analyze user photo → StyleProfile (height, weight, body shape, skin tone…)
    2. For each wardrobe image, extract garments (specific category or all)
    3. Build outfit combinations and score them (Layer 2)
    4. Apply context scoring (Layer 3)
    5. Select top-N combinations
    6. Generate LLM explanations for each (Layer 4)
    7. Generate catalogue visualisations (Layer 5)
    8. (Optional) Virtual try-on for the top outfit (Layer 6)
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path
from uuid import uuid4
import base64
import io
import time
import tempfile

from src.core.models import (
    Garment, GarmentAttributes, UserContext, Outfit, OutfitItem,
    GarmentCategory, ColorProfile, Occasion,
    FormalityLevel, PatternInfo, MaterialProfile,
)
from src.core import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class WardrobeImageItem(BaseModel):
    """Describes a single wardrobe image to process."""
    image_b64: str = Field(
        ..., description="Base-64 encoded image bytes"
    )
    extract_category: Optional[str] = Field(
        None,
        description=(
            "If set, extract only the garment of this category "
            "(e.g. 'top', 'bottom', 'dress', 'shoes'). "
            "If None, extract ALL garments visible in the image."
        ),
    )


class PreAnalyzedGarmentItem(BaseModel):
    """
    A garment whose vision attributes were already extracted (cached at save-time).
    Bypasses Stage 0+1 entirely — no ML inference needed.
    """
    id: str = Field(..., description="Garment ID from the caller's database")
    category: str = Field(..., description="Garment category: top, bottom, shoes, outerwear, accessory, dress")
    subcategory: Optional[str] = Field(None, description="e.g. 'oxford shirt', 'chino', 'sneakers'")
    color_primary: str = Field(..., description="Primary color name")
    color_secondary: Optional[str] = Field(None)
    color_hex: Optional[str] = Field(None, description="Hex code e.g. '#1B2A4A'")
    pattern: Optional[str] = Field("solid", description="Pattern type")
    material: Optional[str] = Field(None, description="Primary material")
    formality: Optional[str] = Field("casual", description="Formality level")
    seasons: Optional[List[str]] = Field(default_factory=list, description="Suitable seasons")
    confidence: float = Field(default=0.85, ge=0, le=1)


class UserProfileInput(BaseModel):
    """Optional user physical profile (photo + measurements)."""
    image_b64: Optional[str] = Field(
        None, description="Base-64 encoded user photo for style-profile extraction"
    )
    height_cm: Optional[float] = Field(None, description="User height in centimeters")
    weight_kg: Optional[float] = Field(None, description="User weight in kilograms")


class PipelineRequest(BaseModel):
    """Full-pipeline recommendation request."""

    # --- Wardrobe (at least one of the two must be provided) ---
    wardrobe_images: List[WardrobeImageItem] = Field(
        default_factory=list,
        description="List of wardrobe images with optional extraction hints",
    )

    # --- Pre-analyzed garments (fast path — skips Stage 0+1) ---
    pre_analyzed_garments: List[PreAnalyzedGarmentItem] = Field(
        default_factory=list,
        description=(
            "Garments whose attributes were already extracted. "
            "These bypass vision ML entirely. Use when the caller cached "
            "vision_features at garment-add time."
        ),
    )

    # --- User profile (optional) ---
    user_profile: Optional[UserProfileInput] = Field(
        None,
        description="Optional user photo and body measurements for personalised scoring",
    )

    # --- Context ---
    context: UserContext = Field(
        default_factory=UserContext,
        description="User context (occasion, weather, preferences, etc.)",
    )

    # --- Selection ---
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of best outfit combinations to return",
    )

    # --- Scoring ---
    scoring_profile: Optional[str] = Field(
        None,
        description="Scoring profile: default, minimalist, creative, business, casual",
    )

    # --- Options ---
    enable_explanation: bool = Field(
        True, description="Generate LLM explanations (Layer 4)"
    )
    explanation_detail: str = Field(
        "standard",
        description="LLM explanation detail level: brief, standard, detailed",
    )
    enable_visualization: bool = Field(
        True, description="Generate catalogue images (Layer 5)"
    )
    enable_tryon: bool = Field(
        False, description="Run virtual try-on on user photo (Layer 6). Requires user_profile.image_b64"
    )
    tryon_backend: str = Field(
        "catvton",
        description="Try-on backend: catvton, replicate",
    )
    enable_cf_boost: bool = Field(
        False,
        description="Apply collaborative-filtering personalisation (Layer 7). Requires user_id.",
    )
    user_id: Optional[str] = Field(
        None,
        description="User ID for collaborative-filtering boost. Required when enable_cf_boost=True.",
    )


# --- Nested response models ---

class GarmentOut(BaseModel):
    """Garment information returned to the client."""
    id: str
    category: str
    subcategory: Optional[str] = None
    color_primary: str
    formality: Optional[str] = None
    confidence: float = 0.0


class OutfitRecommendation(BaseModel):
    """A single scored outfit recommendation."""
    rank: int
    name: str
    overall_score: float
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    garments: List[GarmentOut]
    explanation: Optional[str] = None
    catalogue_image_b64: Optional[str] = None
    tryon_image_b64: Optional[str] = None


class UserProfileOut(BaseModel):
    """Extracted user profile summary."""
    body_shape: Optional[str] = None
    skin_tone: Optional[str] = None
    undertone: Optional[str] = None
    hair_color: Optional[str] = None
    contrast_level: Optional[str] = None
    visual_weight: Optional[str] = None
    face_shape: Optional[str] = None
    estimated_top_size: Optional[str] = None
    estimated_bottom_size: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None

    # 12-season colour analysis
    season_sub: Optional[str] = None
    chroma: Optional[str] = None
    season_confidence: Optional[float] = None

    # Enhanced morphology
    body_shape_secondary: Optional[str] = None
    body_shape_scores: Optional[Dict[str, float]] = None
    waist_hip_ratio: Optional[float] = None


class PipelineResponse(BaseModel):
    """Full pipeline response."""
    recommendations: List[OutfitRecommendation]
    total_garments_extracted: int
    total_combinations_scored: int
    user_profile: Optional[UserProfileOut] = None
    processing_time_ms: float
    stages_completed: List[str]
    errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Lazy-loaded service singletons
# ---------------------------------------------------------------------------

_outfit_builder = None
_context_engine = None
_outfit_explainer = None
_visualizer = None
_tryon_service = None
_style_profile_pipeline = None
_garment_extraction_pipeline = None
_vision_service = None


def get_outfit_builder():
    """Get or create OutfitBuilder."""
    global _outfit_builder
    if _outfit_builder is None:
        from src.layer2_style import OutfitBuilder
        _outfit_builder = OutfitBuilder()
    return _outfit_builder


def get_context_engine():
    """Get or create ContextEngine."""
    global _context_engine
    if _context_engine is None:
        from src.layer3_context import ContextEngine
        _context_engine = ContextEngine()
    return _context_engine


def get_outfit_explainer():
    """Get or create OutfitExplainer."""
    global _outfit_explainer
    if _outfit_explainer is None:
        from src.layer4_llm import OutfitExplainer
        _outfit_explainer = OutfitExplainer()
    return _outfit_explainer


def get_visualizer():
    """Get or create OutfitVisualizer."""
    global _visualizer
    if _visualizer is None:
        from src.layer5_visualization import OutfitVisualizer
        _visualizer = OutfitVisualizer()
    return _visualizer


def get_tryon_service(backend: str = "catvton"):
    """Get or create VirtualTryOnService."""
    global _tryon_service
    if _tryon_service is None:
        from src.layer6_tryon import VirtualTryOnService, TryOnBackend
        backend_enum = TryOnBackend(backend)
        _tryon_service = VirtualTryOnService(backend=backend_enum)
    return _tryon_service


def get_style_profile_pipeline():
    """Get or create StyleProfilePipeline."""
    global _style_profile_pipeline
    if _style_profile_pipeline is None:
        from src.layer3_context.user_profile.pipeline import StyleProfilePipeline
        _style_profile_pipeline = StyleProfilePipeline()
    return _style_profile_pipeline


def get_garment_extraction_pipeline():
    """Get or create GarmentExtractionPipeline (Layer 0)."""
    global _garment_extraction_pipeline
    if _garment_extraction_pipeline is None:
        from src.layer0_segmentation.pipeline import GarmentExtractionPipeline
        _garment_extraction_pipeline = GarmentExtractionPipeline()
    return _garment_extraction_pipeline


def get_vision_service():
    """Get or create VisionService (Layer 1)."""
    global _vision_service
    if _vision_service is None:
        from src.layer1_vision import VisionService
        _vision_service = VisionService()
    return _vision_service


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _b64_to_pil(b64_string: str):
    """Decode a base-64 string to a PIL Image."""
    from PIL import Image
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes))


def _pil_to_b64(img) -> str:
    """Encode a PIL Image to a base-64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _garment_to_out(g: Garment) -> GarmentOut:
    """Convert internal Garment to API-facing GarmentOut."""
    return GarmentOut(
        id=g.id,
        category=g.attributes.category.value,
        subcategory=g.attributes.subcategory,
        color_primary=g.attributes.color.primary,
        formality=g.attributes.formality_level.value if g.attributes.formality_level else None,
        confidence=g.attributes.confidence_score,
    )


def _pre_analyzed_to_garment(item: PreAnalyzedGarmentItem) -> Garment:
    """
    Convert a PreAnalyzedGarmentItem to a full Garment model.
    This skips ALL ML inference — attributes are taken as-is from the caller.
    """
    cat_str = item.category.lower()
    try:
        cat = GarmentCategory(cat_str)
    except ValueError:
        cat = GarmentCategory.TOP  # safe fallback

    formality_str = (item.formality or "casual").lower()
    try:
        formality = FormalityLevel(formality_str)
    except ValueError:
        formality = FormalityLevel.CASUAL

    from src.core.models import Season
    season_map = {"spring": Season.SPRING, "summer": Season.SUMMER, "fall": Season.FALL, "winter": Season.WINTER}
    seasons = [season_map[s.lower()] for s in (item.seasons or []) if s.lower() in season_map]

    return Garment(
        id=item.id,
        attributes=GarmentAttributes(
            category=cat,
            subcategory=item.subcategory,
            color=ColorProfile(
                primary=item.color_primary,
                secondary=item.color_secondary,
                hex_codes=[item.color_hex] if item.color_hex else [],
            ),
            pattern=PatternInfo(type=item.pattern or "solid"),
            material=MaterialProfile(primary=item.material) if item.material else None,
            formality_level=formality,
            season_suitable=seasons,
            confidence_score=item.confidence,
        ),
    )


# ---------------------------------------------------------------------------
# Main pipeline endpoint
# ---------------------------------------------------------------------------

@router.post("/recommend", response_model=PipelineResponse)
async def full_pipeline_recommend(request: PipelineRequest):
    """
    Full-pipeline outfit recommendation.

    Integrates every layer of the system:
    0. Garment extraction from wardrobe images
    1. Vision attribute analysis
    2. Outfit combination building + scoring
    3. Context-aware filtering + user-profile scoring
    4. LLM explanations
    5. Catalogue visualisations
    6. (optional) Virtual try-on
    """
    start = time.time()
    stages: List[str] = []
    errors: List[str] = []

    # ------------------------------------------------------------------
    # STAGE 3a — User profile extraction (run FIRST so that profile-only
    #            requests still return data even without a full wardrobe)
    # ------------------------------------------------------------------
    user_profile_out: Optional[UserProfileOut] = None
    style_profile = None
    if request.user_profile:
        t_stage = time.time()
        try:
            user_profile_out, style_profile = _extract_user_profile(
                request.user_profile
            )
            # Merge profile info into context
            _enrich_context_from_profile(request.context, style_profile)
            stages.append("user_profile")
        except Exception as exc:
            logger.error(f"User profile extraction failed: {exc}")
            errors.append(f"user_profile: {exc}")
        logger.info(f"⏱ STAGE 3a — User profile: {(time.time() - t_stage) * 1000:.1f}ms")

    # ------------------------------------------------------------------
    # STAGE 0+1 — Extract garments from wardrobe images
    #             Fast path: if pre_analyzed_garments are provided, convert
    #             them directly to Garment objects (skips all ML inference).
    # ------------------------------------------------------------------
    all_garments: List[Garment] = []
    t_stage = time.time()

    # Fast path — pre-analyzed garments (no ML, ~0ms)
    if request.pre_analyzed_garments:
        for pa in request.pre_analyzed_garments:
            try:
                all_garments.append(_pre_analyzed_to_garment(pa))
            except Exception as exc:
                logger.warning(f"Could not convert pre-analyzed garment {pa.id}: {exc}")
        if all_garments:
            stages.append("pre_analyzed_garment_conversion")
        logger.info(f"⏱ STAGE 0+1 — Pre-analyzed fast path: {(time.time() - t_stage) * 1000:.1f}ms ({len(request.pre_analyzed_garments)} → {len(all_garments)} garments)")

    # Slow path — raw images requiring ML extraction
    if request.wardrobe_images:
        t_img = time.time()
        try:
            extracted = await _extract_garments_from_images(
                request.wardrobe_images
            )
            all_garments.extend(extracted)
            stages.append("garment_extraction")
        except Exception as exc:
            logger.error(f"Garment extraction failed: {exc}")
            errors.append(f"garment_extraction: {exc}")
        logger.info(f"⏱ STAGE 0+1 — Image extraction: {(time.time() - t_img) * 1000:.1f}ms ({len(request.wardrobe_images)} images → {len(all_garments)} garments)")

    if not request.pre_analyzed_garments and not request.wardrobe_images:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of 'wardrobe_images' or 'pre_analyzed_garments'.",
        )

    if len(all_garments) < 2:
        # If we have a user profile, return it even without enough garments
        if user_profile_out is not None:
            processing_time = (time.time() - start) * 1000
            return PipelineResponse(
                recommendations=[],
                total_garments_extracted=len(all_garments),
                total_combinations_scored=0,
                user_profile=user_profile_out,
                processing_time_ms=round(processing_time, 1),
                stages_completed=stages,
                errors=[
                    f"garment_extraction: only {len(all_garments)} garments found "
                    "(need ≥ 2 for outfit building)"
                ],
            )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least 2 garments to build outfits, "
                f"but only {len(all_garments)} were extracted. "
                "Check your images or extraction settings."
            ),
        )

    # ------------------------------------------------------------------
    # STAGE 2 — Build outfit combinations & score
    # ------------------------------------------------------------------
    t_stage = time.time()
    builder = get_outfit_builder()
    builder.context = request.context

    # Organise garments into wardrobe dict by category
    wardrobe_dict = _organise_wardrobe(all_garments)

    candidates = []
    try:
        candidates = builder.search_best_outfits(
            wardrobe=wardrobe_dict,
            top_k=request.top_k,
            profile=request.scoring_profile,
        )
        stages.append("outfit_scoring")
    except Exception as exc:
        import traceback as _tb
        logger.error(f"Outfit scoring failed: {exc}\n{''.join(_tb.format_exception(type(exc), exc, exc.__traceback__))}")
        errors.append(f"outfit_scoring: {exc}")
    logger.info(f"⏱ STAGE 2 — Outfit scoring: {(time.time() - t_stage) * 1000:.1f}ms ({len(candidates)} candidates)")

    if not candidates:
        raise HTTPException(
            status_code=400,
            detail="No valid outfit combinations could be generated from the extracted garments.",
        )

    total_combinations = len(candidates)

    # Keep only top-k
    top_candidates = candidates[: request.top_k]

    # ------------------------------------------------------------------
    # STAGE 3b — Context scoring
    # ------------------------------------------------------------------
    t_stage = time.time()
    try:
        ctx_engine = get_context_engine()
        if style_profile:
            ctx_engine.set_style_profile(style_profile)

        # Build Outfit objects for context scoring
        outfits_for_context = _candidates_to_outfits(top_candidates)
        scored_outfits = await ctx_engine.apply_context(
            outfits_for_context, request.context
        )
        # Re-rank top_candidates based on context scoring
        _update_candidate_scores(top_candidates, scored_outfits)
        top_candidates.sort(reverse=True)
        stages.append("context_scoring")
    except Exception as exc:
        logger.error(f"Context scoring failed: {exc}")
        errors.append(f"context_scoring: {exc}")
    logger.info(f"⏱ STAGE 3b — Context scoring: {(time.time() - t_stage) * 1000:.1f}ms")

    # ------------------------------------------------------------------
    # STAGE 3c — Collaborative-filtering personalisation (optional)
    # ------------------------------------------------------------------
    if request.enable_cf_boost and request.user_id:
        t_stage = time.time()
        try:
            from src.layer7_cf.cf_engine import get_cf_engine
            cf_engine = get_cf_engine()
            if cf_engine.cf.is_trained:
                # Convert candidates to dicts for the hybrid recommender
                cand_dicts = []
                for cand in top_candidates:
                    cand_dicts.append({
                        "id": getattr(cand, "name", str(id(cand))),
                        "overall_score": cand.overall_score,
                        "_candidate": cand,
                    })
                reranked = cf_engine.recommender.rerank(
                    cand_dicts, request.user_id, cf_engine.cf
                )
                # Re-sort top_candidates based on combined_score order
                id_order = {d["id"]: i for i, d in enumerate(reranked)}
                top_candidates.sort(
                    key=lambda c: id_order.get(
                        getattr(c, "name", str(id(c))), 999
                    )
                )
                stages.append("cf_boost")
        except Exception as exc:
            logger.error(f"CF boost failed (non-blocking): {exc}")
            errors.append(f"cf_boost: {exc}")
        logger.info(f"⏱ STAGE 3c — CF boost: {(time.time() - t_stage) * 1000:.1f}ms")
    else:
        logger.info(f"⏱ STAGE 3c — CF boost: SKIPPED (enable_cf_boost={request.enable_cf_boost}, user_id={request.user_id})")

    # ------------------------------------------------------------------
    # STAGE 4 — LLM explanations (parallelized with asyncio.gather)
    # ------------------------------------------------------------------
    explanations: Dict[int, str] = {}
    if request.enable_explanation:
        t_stage = time.time()
        try:
            import asyncio
            explainer = get_outfit_explainer()

            async def _explain_one(idx: int, cand) -> tuple:
                t_exp = time.time()
                outfit_obj = _candidate_to_outfit(cand, idx)
                explanation = await explainer.explain(
                    outfit_obj,
                    request.context,
                    detail_level=request.explanation_detail,
                )
                elapsed = (time.time() - t_exp) * 1000
                logger.info(f"⏱ STAGE 4 — LLM explanation #{idx + 1}: {elapsed:.1f}ms")
                return idx, explanation.summary

            results = await asyncio.gather(
                *[_explain_one(i, c) for i, c in enumerate(top_candidates)],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"LLM explanation failed for one outfit: {result}")
                else:
                    explanations[result[0]] = result[1]
            stages.append("llm_explanation")
        except Exception as exc:
            logger.error(f"LLM explanation failed: {exc}")
            errors.append(f"llm_explanation: {exc}")
        logger.info(f"⏱ STAGE 4 — LLM explanations TOTAL: {(time.time() - t_stage) * 1000:.1f}ms ({len(explanations)} outfits)")
    else:
        logger.info("⏱ STAGE 4 — LLM explanations: SKIPPED")

    # ------------------------------------------------------------------
    # STAGE 5 — Visualisation
    # ------------------------------------------------------------------
    catalogue_images: Dict[int, str] = {}
    if request.enable_visualization:
        t_stage = time.time()
        try:
            viz = get_visualizer()
            for idx, cand in enumerate(top_candidates):
                outfit_viz = viz.create_outfit_catalogue(
                    garments=cand.garments,
                    outfit_name=f"#{idx + 1}: {cand.name}",
                    score=cand.overall_score,
                )
                catalogue_images[idx] = _pil_to_b64(outfit_viz.image)
            stages.append("visualization")
        except Exception as exc:
            logger.error(f"Visualization failed: {exc}")
            errors.append(f"visualization: {exc}")
        logger.info(f"⏱ STAGE 5 — Visualization: {(time.time() - t_stage) * 1000:.1f}ms ({len(catalogue_images)} images)")
    else:
        logger.info("⏱ STAGE 5 — Visualization: SKIPPED")

    # ------------------------------------------------------------------
    # STAGE 6 — Virtual try-on (optional, top outfit only)
    # ------------------------------------------------------------------
    tryon_images: Dict[int, str] = {}
    if request.enable_tryon:
        t_stage = time.time()
        if not request.user_profile or not request.user_profile.image_b64:
            errors.append("tryon: user_profile.image_b64 is required for virtual try-on")
        else:
            try:
                tryon = get_tryon_service(request.tryon_backend)
                person_img = _b64_to_pil(request.user_profile.image_b64)
                # Try-on only for the #1 outfit
                best_cand = top_candidates[0]
                result = tryon.try_on_outfit(
                    person_image=person_img,
                    outfit_name=best_cand.name,
                    garments=best_cand.garments,
                    outfit_score=best_cand.overall_score,
                )
                if result.composite_image:
                    tryon_images[0] = _pil_to_b64(result.composite_image)
                stages.append("tryon")
            except Exception as exc:
                logger.error(f"Try-on failed: {exc}")
                errors.append(f"tryon: {exc}")
        logger.info(f"⏱ STAGE 6 — Try-on: {(time.time() - t_stage) * 1000:.1f}ms")
    else:
        logger.info("⏱ STAGE 6 — Try-on: SKIPPED")

    # ------------------------------------------------------------------
    # Build response
    # ------------------------------------------------------------------
    recommendations: List[OutfitRecommendation] = []
    for idx, cand in enumerate(top_candidates):
        rec = OutfitRecommendation(
            rank=idx + 1,
            name=cand.name,
            overall_score=round(cand.overall_score, 4),
            score_breakdown=cand.scorecard.scores if cand.scorecard else {},
            garments=[_garment_to_out(g) for g in cand.garments],
            explanation=explanations.get(idx),
            catalogue_image_b64=catalogue_images.get(idx),
            tryon_image_b64=tryon_images.get(idx),
        )
        recommendations.append(rec)

    processing_time = (time.time() - start) * 1000
    logger.info(
        f"⏱ PIPELINE TOTAL: {processing_time:.1f}ms | stages={stages} | garments={len(all_garments)} | outfits={len(recommendations)} | errors={errors or 'none'}",
    )

    return PipelineResponse(
        recommendations=recommendations,
        total_garments_extracted=len(all_garments),
        total_combinations_scored=total_combinations,
        user_profile=user_profile_out,
        processing_time_ms=round(processing_time, 1),
        stages_completed=stages,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _extract_garments_from_images(
    items: List[WardrobeImageItem],
) -> List[Garment]:
    """
    For each wardrobe image:
    - If extract_category is set → extract only that category (Layer 0)
    - Else → extract all garments from the image (Layer 0 + Layer 1)
    """
    builder = get_outfit_builder()
    garments: List[Garment] = []

    for item in items:
        img = _b64_to_pil(item.image_b64)
        # Save to a temp file so existing pipelines can consume a path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp, format="PNG")
            tmp_path = Path(tmp.name)

        if item.extract_category:
            # Extract only a specific garment type via Layer 0
            try:
                pipeline = get_garment_extraction_pipeline()
                result = pipeline.process(
                    tmp_path, categories=[item.extract_category]
                )
                for eg in result.garments:
                    # Build Garment from extracted result + vision attributes
                    garment = await builder.extract_garment_from_image(
                        eg.image_path or tmp_path
                    )
                    garments.append(garment)
            except Exception:
                # Fallback: treat the whole image as the specified garment
                garment = await builder.extract_garment_from_image(tmp_path)
                garments.append(garment)
        else:
            # Extract all garments from the image
            try:
                extracted = await builder.extract_full_outfit_from_image(tmp_path)
                garments.extend(extracted)
            except Exception:
                # Fallback: treat the whole image as a single garment
                garment = await builder.extract_garment_from_image(tmp_path)
                garments.append(garment)

        # Clean up temp file
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return garments


def _extract_user_profile(profile_input: UserProfileInput):
    """Run the StyleProfilePipeline on the user photo."""
    from src.layer3_context.user_profile.pipeline import PipelineResult

    pipeline = get_style_profile_pipeline()

    img = None
    if profile_input.image_b64:
        img = _b64_to_pil(profile_input.image_b64)

    result: PipelineResult = pipeline.analyze(
        image=img,
        height_cm=profile_input.height_cm,
        weight_kg=profile_input.weight_kg,
    )

    profile = result.profile

    # Helper to safely extract enum .value
    def _val(obj):
        return obj.value if hasattr(obj, "value") else str(obj) if obj else None

    body_shape_raw = getattr(profile, "body_shape", None) if profile else None

    out = UserProfileOut(
        body_shape=_val(body_shape_raw),
        skin_tone=(
            profile.color_profile.skin_tone.value
            if profile and hasattr(profile, "color_profile") and profile.color_profile and profile.color_profile.skin_tone
            else (profile.skin_tone.value if profile and profile.skin_tone else None)
        ),
        undertone=(
            profile.color_profile.undertone.value
            if profile and hasattr(profile, "color_profile") and profile.color_profile and profile.color_profile.undertone
            else (profile.undertone.value if profile and profile.undertone else None)
        ),
        hair_color=(
            profile.hair_color.value
            if profile and hasattr(profile, "hair_color") and profile.hair_color
            else None
        ),
        contrast_level=(
            profile.contrast_level.value
            if profile and hasattr(profile, "contrast_level") and profile.contrast_level
            else None
        ),
        visual_weight=(
            profile.visual_weight.value
            if profile and hasattr(profile, "visual_weight") and profile.visual_weight
            else None
        ),
        face_shape=(
            profile.face_shape.value
            if profile and hasattr(profile, "face_shape") and profile.face_shape
            else None
        ),
        estimated_top_size=(
            profile.body_metrics.estimated_top_size
            if profile and hasattr(profile, "body_metrics") and profile.body_metrics
            else None
        ),
        estimated_bottom_size=(
            profile.body_metrics.estimated_bottom_size
            if profile and hasattr(profile, "body_metrics") and profile.body_metrics
            else None
        ),
        height_cm=profile_input.height_cm,
        weight_kg=profile_input.weight_kg,
        # 12-season colour analysis (propagated from SkinAnalysis → StyleProfile)
        season_sub=getattr(profile, "season_sub", None) if profile else None,
        chroma=getattr(profile, "chroma", None) if profile else None,
        season_confidence=getattr(profile, "season_confidence", None) if profile else None,
        # Enhanced morphology (propagated from BodyMetrics → StyleProfile)
        body_shape_secondary=_val(getattr(profile, "body_shape_secondary", None)) if profile else None,
        body_shape_scores=getattr(profile, "body_shape_scores", None) if profile else None,
        waist_hip_ratio=getattr(profile, "waist_hip_ratio", None) if profile else None,
    )

    return out, profile


def _enrich_context_from_profile(context: UserContext, style_profile):
    """Merge extracted style profile data into the UserContext."""
    if style_profile is None:
        return

    # Body shape
    if hasattr(style_profile, "body_shape") and style_profile.body_shape:
        body_val = style_profile.body_shape
        if hasattr(body_val, "value"):
            body_val = body_val.value
        context.body_shape = body_val

    # Skin undertone
    if (
        hasattr(style_profile, "color_profile")
        and style_profile.color_profile
        and hasattr(style_profile.color_profile, "undertone")
        and style_profile.color_profile.undertone
    ):
        ut = style_profile.color_profile.undertone
        context.skin_undertone = ut.value if hasattr(ut, "value") else str(ut)

    # Hair color
    if hasattr(style_profile, "hair_color") and style_profile.hair_color:
        hc = style_profile.hair_color
        context.hair_color = hc.value if hasattr(hc, "value") else str(hc)

    # Contrast type
    if hasattr(style_profile, "contrast_level") and style_profile.contrast_level:
        cl = style_profile.contrast_level
        context.contrast_type = cl.value if hasattr(cl, "value") else str(cl)


def _organise_wardrobe(garments: List[Garment]) -> Dict[str, List[Garment]]:
    """Organise a flat garment list into {category: [garments]} dict."""
    wardrobe: Dict[str, List[Garment]] = {
        "tops": [],
        "bottoms": [],
        "shoes": [],
        "outerwear": [],
        "accessories": [],
        "full_body": [],
    }

    category_mapping = {
        "top": "tops",
        "bottom": "bottoms",
        "shoes": "shoes",
        "outerwear": "outerwear",
        "accessory": "accessories",
        "bag": "accessories",
        "dress": "full_body",
    }

    for g in garments:
        cat_value = g.attributes.category.value
        bucket = category_mapping.get(cat_value, "accessories")
        wardrobe[bucket].append(g)

    return wardrobe


def _candidates_to_outfits(candidates) -> List[Outfit]:
    """Convert OutfitCandidate list to Outfit list for the context engine."""
    outfits: List[Outfit] = []
    for i, cand in enumerate(candidates):
        outfits.append(_candidate_to_outfit(cand, i))
    return outfits


def _candidate_to_outfit(cand, idx: int = 0) -> Outfit:
    """Convert a single OutfitCandidate to an Outfit model."""
    items = []
    for g in cand.garments:
        items.append(
            OutfitItem(
                garment=g,
                role=g.attributes.category.value,
            )
        )
    return Outfit(
        id=f"outfit_{idx}_{uuid4().hex[:8]}",
        items=items,
        compatibility_score=cand.scorecard.scores.get("color_harmony", 0.5) if cand.scorecard else 0.5,
        style_coherence_score=cand.scorecard.scores.get("design_principles", 0.5) if cand.scorecard else 0.5,
        occasion_match_score=cand.scorecard.scores.get("occasion", 0.5) if cand.scorecard else 0.5,
        overall_score=cand.overall_score,
    )


def _update_candidate_scores(candidates, scored_outfits: List[Outfit]):
    """Update candidate overall_score from the context-scored Outfit list."""
    # Create a mapping id->score from scored outfits
    score_map: Dict[str, float] = {}
    for outfit in scored_outfits:
        score_map[outfit.id] = outfit.overall_score

    # If context scoring didn't change IDs we can match by position
    for i, cand in enumerate(candidates):
        if i < len(scored_outfits):
            cand.overall_score = scored_outfits[i].overall_score
