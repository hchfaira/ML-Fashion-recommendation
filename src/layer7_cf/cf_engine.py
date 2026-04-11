"""
CF Engine — application-level singleton for the CF subsystem.
==============================================================

Provides :func:`get_cf_engine` which returns a lazily-initialised
:class:`CFEngine` singleton.  The engine wraps:

* :class:`InteractionBuilder`
* :class:`CollaborativeFilter`
* :class:`CFHybridRecommender`

and manages persistence in ``models/cf/``.

Supports ALS (default) and BPR model types, optional Redis caching
for user latent vectors, user-based / item-based CF queries, and
hybrid scoring with 40/40/20 (style / CF / context) blend.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core import get_logger
from .collaborative_filter import CollaborativeFilter
from .hybrid_recommender import CFHybridRecommender
from .interaction_builder import InteractionBuilder
from .models import UserCFNeighbour, ItemPair

logger = get_logger(__name__)

_DEFAULT_MODEL_DIR = Path("models/cf")


class CFEngine:
    """High-level façade around the CF pipeline.

    Parameters
    ----------
    model_dir : Path
        Directory for persisted model files.
    model_type : str
        ``"als"`` or ``"bpr"`` — forwarded to :class:`CollaborativeFilter`.
    redis_url : str or None
        Redis URL for vector caching.  Defaults to the ``REDIS_URL``
        environment variable.  ``None`` or empty disables caching.

    Attributes
    ----------
    builder : InteractionBuilder
    cf : CollaborativeFilter
    recommender : CFHybridRecommender
    """

    def __init__(
        self,
        model_dir: Path | str = _DEFAULT_MODEL_DIR,
        model_type: str = "als",
        redis_url: Optional[str] = None,
    ) -> None:
        self.model_dir = Path(model_dir)

        # Resolve Redis URL from arg → env → None
        _redis_url = redis_url or os.getenv("REDIS_URL") or None

        self.builder = InteractionBuilder()
        self.cf = CollaborativeFilter(
            model_type=model_type,
            redis_url=_redis_url,
        )
        self.recommender = CFHybridRecommender()

        # Attempt to load a persisted model on startup
        if self.model_dir.exists():
            self.cf.load(self.model_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrain(self, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the interaction matrix from *raw_data* and retrain.

        Returns a status dict suitable for JSON serialisation.
        """
        matrix = self.builder.build(raw_data)
        stats = InteractionBuilder.stats(matrix)
        success = self.cf.train(matrix)

        if success:
            try:
                self.cf.save(self.model_dir)
            except Exception as exc:
                logger.error("Failed to persist CF model: %s", exc)

        return {
            "trained": success,
            "matrix": stats,
            "model": self.cf.status(),
        }

    def similar_users(
        self, user_id: str, n: int = 5
    ) -> List[UserCFNeighbour]:
        """Return the *n* most similar users (user-based CF)."""
        return self.cf.find_similar_users(user_id, n=n)

    def item_pairs(
        self, garment_id: str, n: int = 5
    ) -> List[ItemPair]:
        """Return the *n* most co-used items (item-based CF)."""
        return self.cf.find_item_pairs(garment_id, n=n)

    def hybrid_score(
        self,
        candidates: List[Dict[str, Any]],
        user_id: str,
        context_scores: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Run the full hybrid rerank (style 40% + CF 40% + context 20%)."""
        return self.recommender.rerank(
            candidates, user_id, self.cf, context_scores=context_scores,
        )

    def status(self) -> Dict[str, Any]:
        """Return the current engine status."""
        return self.cf.status()


# ------------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------------

_engine: Optional[CFEngine] = None


def get_cf_engine(
    model_dir: Path | str = _DEFAULT_MODEL_DIR,
    model_type: str = "als",
    redis_url: Optional[str] = None,
) -> CFEngine:
    """Return the global :class:`CFEngine` singleton.

    The engine is created lazily on first call and reused afterwards.
    """
    global _engine
    if _engine is None:
        _engine = CFEngine(
            model_dir=model_dir,
            model_type=model_type,
            redis_url=redis_url,
        )
    return _engine


def reset_cf_engine() -> None:
    """Reset the global singleton (useful for tests)."""
    global _engine
    _engine = None
