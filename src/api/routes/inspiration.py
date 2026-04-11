"""
Inspiration-to-Outfit API Route
=================================

Accepts an inspiration image + wardrobe → returns outfits that recreate the
inspiration's style using garments from the user's own closet.

Pipeline stages:
  1. Decode inspiration image → Style DNA (via Gemini vision)
  2. Build slot blueprint (with body-shape overlay)
  3. Score wardrobe garments against each slot
  4. Greedy CSP assembly of complete outfits
  5. Inspiration-fidelity re-ranking
  6. LLM explanation (parallel)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time, json, base64, asyncio

from src.core.models import (
    Garment, GarmentAttributes, UserContext,
    GarmentCategory, ColorProfile, FormalityLevel,
    PatternInfo, MaterialProfile, Season,
    Outfit, OutfitItem,
)
from src.core import get_logger

# Reuse existing route helpers from the main pipeline module
from src.api.routes.pipeline import (
    PreAnalyzedGarmentItem,
    GarmentOut,
    OutfitRecommendation,
    PipelineResponse,
    UserProfileOut,
    _pre_analyzed_to_garment,
    _garment_to_out,
    get_outfit_explainer,
)

from src.layer2_style.inspiration_scorer import (
    InspirationScorer, StyleDNA, AssembledOutfit,
)

logger = get_logger(__name__)
router = APIRouter()

# ── Lazy singleton ────────────────────────────────────────────
_scorer: InspirationScorer | None = None

def _get_scorer() -> InspirationScorer:
    global _scorer
    if _scorer is None:
        _scorer = InspirationScorer()
    return _scorer


# ── Request / Response models ─────────────────────────────────

class InspirationRequest(BaseModel):
    """Request to generate outfits from an inspiration image."""

    inspiration_image_b64: str = Field(
        ..., description="Base-64 encoded inspiration photo"
    )

    pre_analyzed_garments: List[PreAnalyzedGarmentItem] = Field(
        default_factory=list,
        description="Wardrobe garments with pre-extracted attributes",
    )

    body_shape: Optional[str] = Field(
        None, description="User body shape: pear, apple, hourglass, rectangle, inverted_triangle, athletic"
    )

    occasion: Optional[str] = Field(
        None, description="Target occasion (optional — auto-detected from inspiration if absent)"
    )

    top_k: int = Field(3, ge=1, le=10, description="Number of outfits to return")

    enable_explanation: bool = Field(
        True, description="Generate LLM explanations for each outfit"
    )
    explanation_detail: str = Field(
        "standard", description="LLM explanation detail: brief, standard, detailed"
    )


class StyleDNAOut(BaseModel):
    """Decoded style DNA for client consumption."""
    palette: List[str] = []
    palette_hex: List[str] = []
    formality: str = "casual"
    aesthetic: List[str] = []
    key_piece: Optional[str] = None
    mood: Optional[str] = None
    season_vibe: str = "all_season"
    color_relationship: str = "tonal"


class InspirationResponse(BaseModel):
    """Response from the inspiration outfit pipeline."""
    recommendations: List[OutfitRecommendation]
    style_dna: StyleDNAOut
    total_garments: int
    processing_time_ms: float
    stages_completed: List[str] = []
    errors: List[str] = []


# ── Style DNA extraction via Gemini Vision ────────────────────

_STYLE_DNA_SYSTEM = """\
You are a fashion style analyst. Analyze the provided inspiration outfit image 
and extract its Style DNA as a structured JSON object.

Return ONLY valid JSON with these fields:
{
  "palette": ["color_name_1", "color_name_2", ...],
  "palette_hex": ["#RRGGBB", "#RRGGBB", ...],
  "formality": "casual" | "smart_casual" | "business" | "formal",
  "aesthetic": ["tag1", "tag2", ...],
  "silhouette_top": "slim" | "regular" | "oversized",
  "silhouette_bottom": "slim" | "regular" | "wide",
  "key_piece_category": "top" | "bottom" | "outerwear" | "dress" | "shoes" | null,
  "key_piece_description": "short description of the statement piece" | null,
  "layering_depth": 1 | 2 | 3,
  "season_vibe": "spring" | "summer" | "fall" | "winter" | "all_season",
  "mood": "editorial" | "street" | "bohemian" | "classic" | "sporty" | "romantic" | null,
  "color_relationship": "tonal" | "complementary" | "monochrome" | "contrast"
}

Rules:
- palette: list the 2-5 dominant colors by name (e.g. ["ivory", "camel", "cognac"])
- palette_hex: corresponding hex codes
- aesthetic: tags like "quiet luxury", "minimalist", "streetwear", "athleisure", etc.
- Be precise with silhouette (the fit/volume of top vs bottom)
- key_piece is the garment that defines the outfit's character
- Return ONLY the JSON object, no markdown fences or extra text.
"""


async def _decode_style_dna(image_b64: str) -> StyleDNA:
    """Send the inspiration image to Gemini and parse Style DNA."""
    from openai import AsyncOpenAI
    from config import get_settings

    settings = get_settings()

    # Build client — same pattern as LLMService
    if settings.openai_api_key:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        model = settings.llm_model
    else:
        client = AsyncOpenAI(
            api_key=settings.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model = settings.llm_model

    # Build the vision message
    messages = [
        {"role": "system", "content": _STYLE_DNA_SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": "Analyze this outfit image and extract its Style DNA as JSON.",
                },
            ],
        },
    ]

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=600,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

        data = json.loads(raw)

        return StyleDNA(
            palette=data.get("palette", []),
            palette_hex=data.get("palette_hex", []),
            formality=data.get("formality", "casual"),
            aesthetic=data.get("aesthetic", []),
            silhouette_top=data.get("silhouette_top", "regular"),
            silhouette_bottom=data.get("silhouette_bottom", "regular"),
            key_piece_category=data.get("key_piece_category"),
            key_piece_description=data.get("key_piece_description"),
            layering_depth=data.get("layering_depth", 1),
            season_vibe=data.get("season_vibe", "all_season"),
            mood=data.get("mood"),
            color_relationship=data.get("color_relationship", "tonal"),
        )
    except json.JSONDecodeError as e:
        logger.error(f"Style DNA JSON parse failed: {e}\nRaw: {raw[:500]}")
        # Return a neutral fallback DNA
        return StyleDNA()
    except Exception as e:
        logger.error(f"Style DNA extraction failed: {e}")
        return StyleDNA()


# ── LLM explanation for inspiration outfits ───────────────────

async def _explain_inspiration_outfit(
    outfit: AssembledOutfit,
    dna: StyleDNA,
    body_shape: Optional[str],
    idx: int,
    detail: str = "standard",
) -> str:
    """Generate a contextual explanation for an inspiration-matched outfit."""
    from openai import AsyncOpenAI
    from config import get_settings

    settings = get_settings()

    if settings.openai_api_key:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        model = settings.llm_model
    else:
        client = AsyncOpenAI(
            api_key=settings.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model = settings.llm_model

    garment_desc = "\n".join(
        f"- {g.attributes.category.value}: "
        f"{g.attributes.color.primary} "
        f"{g.attributes.subcategory or ''} "
        f"({g.attributes.formality_level.value if g.attributes.formality_level else 'casual'})"
        for g in outfit.garments
    )

    body_note = ""
    if body_shape:
        body_note = (
            f"\n\nThe user has a {body_shape} body shape. "
            f"Comment briefly on why these pieces work for their silhouette."
        )

    prompt = f"""\
You are a personal stylist. Explain (in 2-3 sentences) why this outfit 
recreates the inspiration look using the user's own wardrobe.

Inspiration Style DNA:
- Palette: {', '.join(dna.palette)}
- Aesthetic: {', '.join(dna.aesthetic)}
- Formality: {dna.formality}
- Mood: {dna.mood or 'N/A'}

Outfit #{idx + 1} (match score: {outfit.total_score:.0%}):
{garment_desc}

Score breakdown:
- Style slot average: {outfit.style_score:.0%}
- Inspiration fidelity: {outfit.inspiration_fidelity:.0%}
- Body harmony: {outfit.body_harmony:.0%}{body_note}

Keep it concise, warm, and actionable. Focus on WHY these pieces capture 
the inspiration's essence.
"""

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Inspiration explanation failed: {e}")
        return f"This outfit captures {dna.formality} {', '.join(dna.aesthetic[:2])} energy from the inspiration."


# ── Name generator ────────────────────────────────────────────

def _generate_outfit_name(dna: StyleDNA, idx: int) -> str:
    """Generate a catchy outfit name from the DNA."""
    mood = dna.mood or "chic"
    aesthetic = dna.aesthetic[0] if dna.aesthetic else "style"
    names = [
        f"The {mood.title()} Edit",
        f"{aesthetic.title()} Remix",
        f"Inspired {mood.title()}",
        f"Your {aesthetic.title()} Take",
        f"The {dna.formality.replace('_', ' ').title()} Muse",
    ]
    return names[idx % len(names)]


# ═══════════════════════════════════════════════════════════════
#  Main endpoint
# ═══════════════════════════════════════════════════════════════

@router.post("/from-inspiration", response_model=InspirationResponse)
async def outfit_from_inspiration(request: InspirationRequest):
    """
    Generate outfits from an inspiration image using the user's wardrobe.

    6-stage pipeline:
      1. Decode inspiration → Style DNA (Gemini vision)
      2. Build slot blueprint (body-shape aware)
      3. Score wardrobe garments against slots
      4. Greedy CSP assembly
      5. Re-rank by inspiration fidelity + body harmony
      6. Parallel LLM explanations
    """
    start = time.time()
    stages: List[str] = []
    errors: List[str] = []

    # ── STAGE 1: Decode Style DNA ──
    t = time.time()
    dna = await _decode_style_dna(request.inspiration_image_b64)
    stages.append("style_dna_extraction")
    logger.info(f"⏱ STAGE 1 — Style DNA: {(time.time() - t) * 1000:.1f}ms | "
                f"palette={dna.palette} formality={dna.formality} aesthetic={dna.aesthetic}")

    # ── Convert pre-analyzed garments ──
    t = time.time()
    garments: List[Garment] = []
    for pa in request.pre_analyzed_garments:
        try:
            garments.append(_pre_analyzed_to_garment(pa))
        except Exception as e:
            logger.warning(f"Garment conversion failed for {pa.id}: {e}")
    stages.append("garment_conversion")
    logger.info(f"⏱ STAGE 1b — Garment conversion: {(time.time() - t) * 1000:.1f}ms "
                f"({len(request.pre_analyzed_garments)} → {len(garments)})")

    if len(garments) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 garments, got {len(garments)}.",
        )

    # ── STAGE 2: Build slot blueprint ──
    t = time.time()
    scorer = _get_scorer()
    slots = scorer.build_slot_blueprint(dna, body_shape=request.body_shape)
    stages.append("slot_blueprint")
    logger.info(f"⏱ STAGE 2 — Slot blueprint: {(time.time() - t) * 1000:.1f}ms "
                f"({len(slots)} slots)")

    # ── STAGE 3+4: Assemble outfits ──
    t = time.time()
    assembled = scorer.assemble_outfits(
        slots=slots,
        garments=garments,
        dna=dna,
        body_shape=request.body_shape,
        top_k=request.top_k,
    )
    stages.append("outfit_assembly")
    logger.info(f"⏱ STAGE 3+4 — Assembly: {(time.time() - t) * 1000:.1f}ms "
                f"({len(assembled)} outfits)")

    if not assembled:
        raise HTTPException(
            status_code=400,
            detail="Could not assemble any outfits from your wardrobe "
                   "that match this inspiration. Try with more garments.",
        )

    # ── STAGE 5: LLM explanations (parallel) ──
    explanations: Dict[int, str] = {}
    if request.enable_explanation:
        t = time.time()
        try:
            async def _explain_one(idx: int, outfit: AssembledOutfit):
                text = await _explain_inspiration_outfit(
                    outfit, dna, request.body_shape, idx, request.explanation_detail
                )
                return idx, text

            results = await asyncio.gather(
                *[_explain_one(i, o) for i, o in enumerate(assembled)],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"Explanation failed: {r}")
                else:
                    explanations[r[0]] = r[1]
            stages.append("llm_explanation")
        except Exception as e:
            errors.append(f"llm_explanation: {e}")
        logger.info(f"⏱ STAGE 5 — Explanations: {(time.time() - t) * 1000:.1f}ms")

    # ── Build response ──
    recommendations: List[OutfitRecommendation] = []
    for idx, outfit in enumerate(assembled):
        rec = OutfitRecommendation(
            rank=idx + 1,
            name=_generate_outfit_name(dna, idx),
            overall_score=round(outfit.total_score, 4),
            score_breakdown={
                "style_slot_avg": outfit.style_score,
                "inspiration_fidelity": outfit.inspiration_fidelity,
                "body_harmony": outfit.body_harmony,
            },
            garments=[_garment_to_out(g) for g in outfit.garments],
            explanation=explanations.get(idx),
        )
        recommendations.append(rec)

    processing_time = (time.time() - start) * 1000
    logger.info(
        f"⏱ INSPIRATION PIPELINE TOTAL: {processing_time:.1f}ms | "
        f"stages={stages} | outfits={len(recommendations)}"
    )

    return InspirationResponse(
        recommendations=recommendations,
        style_dna=StyleDNAOut(
            palette=dna.palette,
            palette_hex=dna.palette_hex,
            formality=dna.formality,
            aesthetic=dna.aesthetic,
            key_piece=dna.key_piece_description,
            mood=dna.mood,
            season_vibe=dna.season_vibe,
            color_relationship=dna.color_relationship,
        ),
        total_garments=len(garments),
        processing_time_ms=round(processing_time, 1),
        stages_completed=stages,
        errors=errors,
    )
