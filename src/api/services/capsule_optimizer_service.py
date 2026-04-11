"""
Capsule Optimizer Service
=========================
Thin service layer that wraps :class:`CapsuleOptimizer` and shapes its
output into Pydantic response models consumed by the API route.

This is the same pattern used by ``capsule_service.py`` for the travel
capsule — stateless, rule-based, zero LLM calls.

Public API
----------
optimize_capsule(request, garments) -> OptimizedCapsuleResponse
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.layer2_style.capsule.capsule_optimizer import (
    CapsuleOptimizer,
    CapsuleOptimizerResult,
    score_capsule,
    _garment_cat,
    _CATEGORY_ROLES,
)


# ─────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────

class OptimizeCapsuleRequest(BaseModel):
    """Parameters for N-piece capsule generation."""
    user_id: str = Field(..., description="Owner of the wardrobe")
    n_pieces: int = Field(
        ..., ge=1, le=50,
        description="Exact number of garments to select",
    )
    occasion_types: List[str] = Field(
        default=["casual", "smart_casual"],
        description="Target occasions (casual, smart_casual, business, formal, evening …)",
    )
    climate: str = Field(
        default="mixed",
        description="Destination climate: warm | cold | mixed",
    )
    anchor_ids: List[str] = Field(
        default_factory=list,
        description="Garment IDs to forcibly include",
    )
    excluded_ids: List[str] = Field(
        default_factory=list,
        description="Garment IDs to exclude from selection",
    )


class PieceContribution(BaseModel):
    garment_id: str
    garment: Dict[str, Any] = Field(default_factory=dict)
    category: str
    role: str
    outfit_contribution: int
    versatility_score: float


class AlternativeCapsule(BaseModel):
    garment_ids: List[str]
    total_score: float
    valid_combinations: int
    label: str


class OptimizedCapsuleResponse(BaseModel):
    """Full response for a capsule optimization request."""
    # Selection summary
    n_pieces: int
    total_wardrobe: int
    total_score: float
    valid_combinations: int

    # Palette
    color_palette: List[str] = Field(default_factory=list)
    color_names: List[str] = Field(default_factory=list)

    # Coverage
    occasion_coverage: Dict[str, float] = Field(default_factory=dict)
    practicality_score: float = 0.0

    # Selected pieces with contribution breakdown
    selected_garments: List[PieceContribution] = Field(default_factory=list)

    # Gaps
    missing_pieces: List[str] = Field(default_factory=list)

    # Alternatives
    alternatives: List[AlternativeCapsule] = Field(default_factory=list)

    # Summary text
    summary: str = ""

    source: str = "greedy_2opt"


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

_optimizer = CapsuleOptimizer()


def optimize_capsule(
    request: OptimizeCapsuleRequest,
    garments: List[Dict[str, Any]],
) -> OptimizedCapsuleResponse:
    """
    Select the best *n_pieces* from *garments* for the given context.

    Parameters
    ----------
    request : OptimizeCapsuleRequest
        Capsule parameters.
    garments : list[dict]
        Serialised garment items from the AlgoStyle backend.

    Returns
    -------
    OptimizedCapsuleResponse
    """
    result: CapsuleOptimizerResult = _optimizer.optimize(
        garments=garments,
        n_pieces=request.n_pieces,
        occasion_types=request.occasion_types,
        climate=request.climate,
        anchor_ids=request.anchor_ids,
        excluded_ids=request.excluded_ids,
    )

    # Build per-piece responses with original garment data attached
    garment_index = {g.get("id", ""): g for g in garments}
    selected_pieces: List[PieceContribution] = []
    for pc in result.piece_contributions:
        gid = pc["garment_id"]
        selected_pieces.append(PieceContribution(
            garment_id=gid,
            garment=garment_index.get(gid, {}),
            category=pc["category"],
            role=pc["role"],
            outfit_contribution=pc["outfit_contribution"],
            versatility_score=pc["versatility_score"],
        ))

    # Build alternatives
    alternatives: List[AlternativeCapsule] = []
    for alt in result.alternatives:
        alt_ids = [g.get("id", "") for g in alt.get("garments", [])]
        alternatives.append(AlternativeCapsule(
            garment_ids=alt_ids,
            total_score=alt.get("total_score", 0.0),
            valid_combinations=alt.get("valid_combinations", 0),
            label=alt.get("label", ""),
        ))

    # Summary text
    occ_str = " & ".join(request.occasion_types) if request.occasion_types else "general"
    summary = (
        f"{result.n_pieces} pieces selected from {result.total_wardrobe} "
        f"garments — {result.valid_combinations} outfit combinations "
        f"across {occ_str} (score: {result.total_score:.1f}/100)."
    )

    return OptimizedCapsuleResponse(
        n_pieces=result.n_pieces,
        total_wardrobe=result.total_wardrobe,
        total_score=result.total_score,
        valid_combinations=result.valid_combinations,
        color_palette=result.color_palette,
        color_names=result.color_names,
        occasion_coverage=result.occasion_coverage,
        practicality_score=result.practicality_score,
        selected_garments=selected_pieces,
        missing_pieces=result.missing_pieces,
        alternatives=alternatives,
        summary=summary,
    )
