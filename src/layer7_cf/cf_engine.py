"""
CF Engine — application-level singleton for the CF subsystem.
==============================================================

Provides :func:`get_cf_engine` which returns a lazily-initialised
:class:`CFEngine` singleton.  The engine wraps:

* :class:`InteractionBuilder`
* :class:`CollaborativeFilter`
* :class:`CFHybridRecommender`

and manages persistence in ``models/cf/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core import get_logger
from .collaborative_filter import CollaborativeFilter
from .hybrid_recommender import CFHybridRecommender
from .interaction_builder import InteractionBuilder

logger = get_logger(__name__)

_DEFAULT_MODEL_DIR = Path("models/cf")


class CFEngine:
    """High-level façade around the CF pipeline.

    Attributes
    ----------
    builder : InteractionBuilder
    cf : CollaborativeFilter
    recommender : CFHybridRecommender
    """

    def __init__(self, model_dir: Path | str = _DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.builder = InteractionBuilder()
        self.cf = CollaborativeFilter()
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

    def status(self) -> Dict[str, Any]:
        """Return the current engine status."""
        return self.cf.status()


# ------------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------------

_engine: Optional[CFEngine] = None


def get_cf_engine(model_dir: Path | str = _DEFAULT_MODEL_DIR) -> CFEngine:
    """Return the global :class:`CFEngine` singleton.

    The engine is created lazily on first call and reused afterwards.
    """
    global _engine
    if _engine is None:
        _engine = CFEngine(model_dir=model_dir)
    return _engine


def reset_cf_engine() -> None:
    """Reset the global singleton (useful for tests)."""
    global _engine
    _engine = None
