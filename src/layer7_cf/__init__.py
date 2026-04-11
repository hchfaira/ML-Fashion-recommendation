"""
Layer 7 — Collaborative Filtering
===================================

Personalised outfit recommendations powered by implicit-feedback
collaborative filtering (ALS / BPR).

Public API:

    from src.layer7_cf import (
        InteractionRecord,
        InteractionMatrix,
        CFScore,
        HybridScore,
        UserCFNeighbour,
        ItemPair,
        InteractionBuilder,
        CollaborativeFilter,
        CFHybridRecommender,
        get_cf_engine,
    )
"""
from .models import (
    InteractionRecord, InteractionMatrix, CFScore, HybridScore,
    UserCFNeighbour, ItemPair,
)
from .interaction_builder import InteractionBuilder
from .collaborative_filter import CollaborativeFilter
from .hybrid_recommender import CFHybridRecommender
from .cf_engine import get_cf_engine

__all__ = [
    "InteractionRecord",
    "InteractionMatrix",
    "CFScore",
    "HybridScore",
    "UserCFNeighbour",
    "ItemPair",
    "InteractionBuilder",
    "CollaborativeFilter",
    "CFHybridRecommender",
    "get_cf_engine",
]
