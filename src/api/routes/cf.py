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
GET  /cf/similar-users/{user_id}
GET  /cf/item-pairs/{garment_id}
GET  /cf/boost-score/{user_id}/{garment_id}
POST /cf/hybrid-score
POST /cf/retrain
GET  /cf/status
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

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


class UserNeighbourItem(BaseModel):
    user_id: str
    similarity: float
    shared_items: int


class SimilarUsersResponse(BaseModel):
    trained: bool
    user_id: str
    similar_users: List[UserNeighbourItem] = Field(default_factory=list)


class ItemPairItem(BaseModel):
    source_id: str
    paired_id: str
    score: float
    co_users: int


class ItemPairsResponse(BaseModel):
    trained: bool
    garment_id: str
    pairs: List[ItemPairItem] = Field(default_factory=list)


class BoostScoreResponse(BaseModel):
    trained: bool
    user_id: str
    garment_id: str
    score: float
    confidence: float


class HybridScoreCandidate(BaseModel):
    id: str = Field(..., description="Outfit or garment ID")
    overall_score: float = Field(0.5, description="Style pipeline score in [0, 1]")


class HybridScoreRequest(BaseModel):
    user_id: str
    candidates: List[HybridScoreCandidate]
    context_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Mapping of outfit_id → context score in [0, 1]",
    )


class HybridScoreItem(BaseModel):
    outfit_id: str
    style_score: float
    cf_score: float
    context_score: float
    combined_score: float
    personalization_active: bool


class HybridScoreResponse(BaseModel):
    trained: bool
    user_id: str
    results: List[HybridScoreItem] = Field(default_factory=list)


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
    model_type: str = "als"
    factors: int
    iterations: int
    regularization: float
    n_users: int
    n_garments: int
    redis_connected: bool = False


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
    """Return top-N personalised recommendations for *user_id*."""
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
    summary="Find similar garments in latent space",
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
    "/cf/similar-users/{user_id}",
    response_model=SimilarUsersResponse,
    summary="Find users with similar style (user-based CF)",
    tags=["Collaborative Filtering"],
)
async def get_similar_users(
    user_id: str,
    n: int = Query(default=5, ge=1, le=50),
) -> SimilarUsersResponse:
    """Users with a similar style profile to *user_id*.

    Each neighbour includes the cosine similarity and number of
    shared items in the interaction matrix.
    """
    engine = get_cf_engine()
    neighbours = engine.similar_users(user_id, n=n)
    return SimilarUsersResponse(
        trained=engine.cf.is_trained,
        user_id=user_id,
        similar_users=[
            UserNeighbourItem(
                user_id=nb.user_id,
                similarity=round(nb.similarity, 4),
                shared_items=nb.shared_items,
            )
            for nb in neighbours
        ],
    )


@router.get(
    "/cf/item-pairs/{garment_id}",
    response_model=ItemPairsResponse,
    summary="Find co-used item pairs (item-based CF)",
    tags=["Collaborative Filtering"],
)
async def get_item_pairs(
    garment_id: str,
    n: int = Query(default=5, ge=1, le=50),
) -> ItemPairsResponse:
    """Users who wore this garment also paired it with these items.

    Based on co-occurrence in the binary interaction matrix.
    """
    engine = get_cf_engine()
    pairs = engine.item_pairs(garment_id, n=n)
    return ItemPairsResponse(
        trained=engine.cf.is_trained,
        garment_id=garment_id,
        pairs=[
            ItemPairItem(
                source_id=p.source_id,
                paired_id=p.paired_id,
                score=round(p.score, 4),
                co_users=p.co_users,
            )
            for p in pairs
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
    "/cf/hybrid-score",
    response_model=HybridScoreResponse,
    summary="Compute hybrid scores (style 40% + CF 40% + context 20%)",
    tags=["Collaborative Filtering"],
)
async def compute_hybrid_score(body: HybridScoreRequest) -> HybridScoreResponse:
    """Blend style, CF, and context scores for a list of candidates.

    Accepts pre-computed style scores (``overall_score``) and optional
    per-outfit context scores.  Returns the re-ranked list with full
    score breakdown.
    """
    engine = get_cf_engine()
    candidates = [
        {"id": c.id, "overall_score": c.overall_score}
        for c in body.candidates
    ]
    ranked = engine.hybrid_score(
        candidates, body.user_id, context_scores=body.context_scores,
    )
    return HybridScoreResponse(
        trained=engine.cf.is_trained,
        user_id=body.user_id,
        results=[
            HybridScoreItem(
                outfit_id=r.get("id", ""),
                style_score=r["style_score"],
                cf_score=r["cf_score"],
                context_score=r["context_score"],
                combined_score=r["combined_score"],
                personalization_active=r["personalization_active"],
            )
            for r in ranked
        ],
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
