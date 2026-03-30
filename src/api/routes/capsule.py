"""
Capsule Wardrobe Routes
=======================
POST /capsule/travel  — build a travel/contextual capsule from a wardrobe.

The full algorithm lives in src/api/services/capsule_service.py.
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
