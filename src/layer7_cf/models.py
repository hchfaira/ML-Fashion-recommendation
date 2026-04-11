"""
Data models for the Collaborative Filtering layer.
=====================================================

All data structures used across the CF pipeline live here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class InteractionRecord:
    """A single user-garment interaction with weighted signal.

    Attributes:
        user_id:      Unique identifier of the user.
        garment_id:   Unique identifier of the garment / outfit.
        signal_type:  Human-readable label for the signal kind
                      (e.g. ``"times_worn"``, ``"outfit_liked"``).
        raw_value:    The original numeric value before weighting.
        weight:       Multiplier applied to *raw_value* to produce
                      the final contribution to the interaction matrix.
    """
    user_id: str
    garment_id: str
    signal_type: str
    raw_value: float
    weight: float


@dataclass
class InteractionMatrix:
    """Aggregated user × garment strength matrix (dense representation).

    Attributes:
        user_ids:      Ordered list of user identifiers (row index).
        garment_ids:   Ordered list of garment identifiers (col index).
        data:          2-D list ``[n_users][n_garments]`` of aggregated
                       interaction strengths (≥ 0).
        user_to_idx:   Fast look-up ``user_id → row index``.
        garment_to_idx: Fast look-up ``garment_id → col index``.
    """
    user_ids: List[str]
    garment_ids: List[str]
    data: List[List[float]]
    user_to_idx: Dict[str, int] = field(default_factory=dict)
    garment_to_idx: Dict[str, int] = field(default_factory=dict)


@dataclass
class CFScore:
    """Score produced by the collaborative filter for one item.

    Attributes:
        garment_id:  Identifier of the scored garment / outfit.
        score:       Normalised score in ``[0, 1]``.
        confidence:  How reliable the score is (0 = cold-start fallback,
                     1 = well-supported by data).
    """
    garment_id: str
    score: float
    confidence: float


@dataclass
class HybridScore:
    """Combined style + CF + context score for one outfit candidate.

    Attributes:
        outfit_id:             Identifier of the outfit.
        style_score:           Score from the style pipeline (Layer 2/3).
        cf_score:              Score from collaborative filtering.
        context_score:         Score from context engine (weather, occasion, …).
        combined_score:        Blended final score.
        personalization_active: ``True`` when the CF model contributed
                                a meaningful signal.
    """
    outfit_id: str
    style_score: float
    cf_score: float
    context_score: float
    combined_score: float
    personalization_active: bool


@dataclass
class UserCFNeighbour:
    """A similar user discovered via user-based CF.

    Attributes:
        user_id:    Identifier of the similar user.
        similarity: Cosine similarity in ``[0, 1]``.
        shared_items: Number of items both users interacted with.
    """
    user_id: str
    similarity: float
    shared_items: int = 0


@dataclass
class ItemPair:
    """An item-based CF pairing: users who wore *source_id* also
    paired it with *paired_id*.

    Attributes:
        source_id:  The query garment.
        paired_id:  The co-occurring garment.
        score:      Affinity / co-occurrence score in ``[0, 1]``.
        co_users:   Number of users who co-used both items.
    """
    source_id: str
    paired_id: str
    score: float
    co_users: int = 0
