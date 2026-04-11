"""
Capsule Wardrobe Routes
=======================
POST /capsule/travel     — build a travel/contextual capsule from a wardrobe.
POST /capsule/optimize   — select the best N-piece capsule (Greedy + 2-opt).

The full algorithm lives in src/api/services/capsule_service.py (travel)
and src/api/services/capsule_optimizer_service.py (optimize).
All tuneable constants are in config/capsule/capsule_config.json.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List

from src.api.services.capsule_service import (
    TravelCapsuleRequest,
    TravelCapsuleResponse,
    build_travel_capsule,
)
from src.api.services.capsule_optimizer_service import (
    OptimizeCapsuleRequest,
    OptimizedCapsuleResponse,
    optimize_capsule,
)
from src.core import get_logger

logger = get_logger(__name__)
router = APIRouter()


class TravelCapsulePayload(BaseModel):
    """
    Wrapper model so FastAPI can parse both `request` and `garments`
    from a single JSON body.

    The AlgoStyle backend sends:
        { "request": { ...TravelCapsuleRequest fields... }, "garments": [ ... ] }
    """
    request: TravelCapsuleRequest
    garments: List[Dict[str, Any]]


@router.post("/travel", response_model=TravelCapsuleResponse, summary="Build a travel capsule")
async def travel_capsule(payload: TravelCapsulePayload) -> TravelCapsuleResponse:
    """
    Build a rule-based travel capsule wardrobe.

    **Request body** (`TravelCapsulePayload`)
    - `request`: capsule parameters (occasion, climate, duration, max_pieces …)
    - `garments`: serialised wardrobe items already fetched by the caller

    The garment list is supplied by the **AlgoStyle backend**, which owns the
    database.  This service is intentionally stateless — it only contains the
    selection algorithm.

    **Returns** a `TravelCapsuleResponse` with:
    - `best_group` — highest-scoring capsule
    - `alternative_groups` — up to 3 alternatives
    - `missing_pieces` — gaps vs the ideal template
    - `pieces_selected`, `total_wardrobe_items`, `source`
    """
    if not payload.garments:
        raise HTTPException(status_code=422, detail="garments list must not be empty")

    try:
        result = build_travel_capsule(request=payload.request, garments=payload.garments)
        return result
    except Exception as exc:
        logger.error(f"capsule build failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# POST /capsule/optimize — N-piece greedy + 2-opt
# ─────────────────────────────────────────────────────────────

class OptimizeCapsulePayload(BaseModel):
    """
    Wrapper model: AlgoStyle backend sends
        { "request": { ... }, "garments": [ ... ] }
    """
    request: OptimizeCapsuleRequest
    garments: List[Dict[str, Any]]


@router.post(
    "/optimize",
    response_model=OptimizedCapsuleResponse,
    summary="Select the best N-piece capsule wardrobe",
)
async def optimize_capsule_endpoint(
    payload: OptimizeCapsulePayload,
) -> OptimizedCapsuleResponse:
    """
    Pick the **optimal N garments** from a wardrobe using Greedy + 2-opt
    local search.

    **Request body** (`OptimizeCapsulePayload`)
    - `request`: optimisation parameters (n_pieces, occasions, climate,
      anchor_ids, excluded_ids)
    - `garments`: serialised wardrobe items from the AlgoStyle backend

    **Returns** an `OptimizedCapsuleResponse` with:
    - `selected_garments` — chosen pieces with per-piece contribution
    - `total_score` — composite score (0-100)
    - `valid_combinations` — number of outfit permutations
    - `occasion_coverage`, `color_palette`, `missing_pieces`
    - `alternatives` — up to 2 alternative capsule sets
    - `summary` — human-readable capsule description
    """
    if not payload.garments:
        raise HTTPException(status_code=422, detail="garments list must not be empty")

    if payload.request.n_pieces < 1:
        raise HTTPException(status_code=422, detail="n_pieces must be ≥ 1")

    try:
        result = optimize_capsule(
            request=payload.request,
            garments=payload.garments,
        )
        return result
    except Exception as exc:
        logger.error(f"capsule optimize failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

