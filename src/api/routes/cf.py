"""
API routes for Collaborative Filtering (Layer 7).
===================================================

All endpoints return ``200`` even when the CF model is not trained.
The ``trained`` flag in the response tells the caller whether
personalisation is active.

Endpoints
---------
GET  /cf/recommendations/{user_id}
GET  /cf/similar-garments/{garment_id}
GET  /cf/boost-score/{user_id}/{garment_id}
POST /cf/retrain
GET  /cf/status
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from src.core import get_logger
from src.layer7_cf.cf_engine import get_cf_engine

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class CFRecommendationItem(BaseModel):
    garment_id: str
    score: float
    confidence: float


class CFRecommendationsResponse(BaseModel):
    trained: bool
    user_id: str
    recommendations: List[CFRecommendationItem] = Field(default_factory=list)


class SimilarGarmentsResponse(BaseModel):
    trained: bool
    garment_id: str
    similar: List[CFRecommendationItem] = Field(default_factory=list)


class BoostScoreResponse(BaseModel):
    trained: bool
    user_id: str
    garment_id: str
    score: float
    confidence: float


class RetrainRequest(BaseModel):
    interactions: List[Dict[str, Any]] = Field(
        ..., description="List of interaction dicts with user_id, garment_id, and signal keys"
    )


class RetrainResponse(BaseModel):
    trained: bool
    matrix: Dict[str, Any] = Field(default_factory=dict)
    model: Dict[str, Any] = Field(default_factory=dict)


class CFStatusResponse(BaseModel):
    trained: bool
    factors: int
    iterations: int
    regularization: float
    n_users: int
    n_garments: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/cf/recommendations/{user_id}",
    response_model=CFRecommendationsResponse,
    summary="Get CF recommendations for a user",
    tags=["Collaborative Filtering"],
)
async def get_recommendations(
    user_id: str,
    n: int = Query(default=10, ge=1, le=100),
    filter_owned: bool = Query(default=True),
) -> CFRecommendationsResponse:
    """Return top-N personalised recommendations for *user_id*.

    If the model is not trained, returns an empty list with ``trained=false``.
    """
    engine = get_cf_engine()
    recs = engine.cf.recommend(user_id, n=n, filter_owned=filter_owned)
    return CFRecommendationsResponse(
        trained=engine.cf.is_trained,
        user_id=user_id,
        recommendations=[
            CFRecommendationItem(
                garment_id=r.garment_id, score=r.score, confidence=r.confidence
            )
            for r in recs
        ],
    )


@router.get(
    "/cf/similar-garments/{garment_id}",
    response_model=SimilarGarmentsResponse,
    summary="Find similar garments",
    tags=["Collaborative Filtering"],
)
async def get_similar_garments(
    garment_id: str,
    n: int = Query(default=5, ge=1, le=50),
) -> SimilarGarmentsResponse:
    """Return the *n* most similar garments in latent space."""
    engine = get_cf_engine()
    similar = engine.cf.find_similar_garments(garment_id, n=n)
    return SimilarGarmentsResponse(
        trained=engine.cf.is_trained,
        garment_id=garment_id,
        similar=[
            CFRecommendationItem(
                garment_id=s.garment_id, score=s.score, confidence=s.confidence
            )
            for s in similar
        ],
    )


@router.get(
    "/cf/boost-score/{user_id}/{garment_id}",
    response_model=BoostScoreResponse,
    summary="Get CF affinity score for user × garment",
    tags=["Collaborative Filtering"],
)
async def get_boost_score(user_id: str, garment_id: str) -> BoostScoreResponse:
    """Return the cosine-similarity based CF affinity score."""
    engine = get_cf_engine()
    result = engine.cf.get_boost_score(user_id, garment_id)
    return BoostScoreResponse(
        trained=engine.cf.is_trained,
        user_id=user_id,
        garment_id=garment_id,
        score=result.score,
        confidence=result.confidence,
    )


@router.post(
    "/cf/retrain",
    response_model=RetrainResponse,
    summary="Retrain the CF model with new interaction data",
    tags=["Collaborative Filtering"],
)
async def retrain_model(body: RetrainRequest) -> RetrainResponse:
    """Submit interaction data and trigger a full retrain."""
    engine = get_cf_engine()
    result = engine.retrain(body.interactions)
    return RetrainResponse(**result)


@router.get(
    "/cf/status",
    response_model=CFStatusResponse,
    summary="Get CF model status",
    tags=["Collaborative Filtering"],
)
async def get_status() -> CFStatusResponse:
    """Return the current CF model status."""
    engine = get_cf_engine()
    return CFStatusResponse(**engine.status())
