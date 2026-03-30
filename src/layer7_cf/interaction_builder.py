"""
Interaction Builder — constructs the user × garment matrix.
=============================================================

Converts raw interaction dictionaries (as they come from the DB / API)
into an :class:`InteractionMatrix` ready for ALS training.

Signal weights
--------------
Each interaction type has a predefined importance weight:

  +-----------------+--------+-------------------------------------------+
  | Signal          | Weight | Rationale                                 |
  +-----------------+--------+-------------------------------------------+
  | times_worn      |  5.0   | Strongest signal — actual wardrobe use    |
  | is_favorite     |  3.0   | Explicit preference                       |
  | worn_count      |  4.0   | Alias for historical wear frequency       |
  | outfit_created  |  2.0   | User actively assembled an outfit         |
  | outfit_liked    |  1.5   | Passive positive signal                   |
  | outfit_saved    |  1.0   | Weakest positive signal                   |
  +-----------------+--------+-------------------------------------------+

Counters (``times_worn``, ``worn_count``) are compressed via
``np.log1p`` before weighting to dampen extreme values.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.core import get_logger
from .models import InteractionMatrix, InteractionRecord

logger = get_logger(__name__)

# Default signal weights
SIGNAL_WEIGHTS: Dict[str, float] = {
    "times_worn": 5.0,
    "is_favorite": 3.0,
    "worn_count": 4.0,
    "outfit_created": 2.0,
    "outfit_liked": 1.5,
    "outfit_saved": 1.0,
}

# Signals that represent counters and should be log-compressed
_COUNTER_SIGNALS = {"times_worn", "worn_count"}


class InteractionBuilder:
    """Build an :class:`InteractionMatrix` from raw interaction dicts.

    Parameters
    ----------
    weights : dict, optional
        Override the default signal weights.  Keys that are not present
        in the override keep their default value.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self._weights = {**SIGNAL_WEIGHTS}
        if weights:
            self._weights.update(weights)

    @property
    def weights(self) -> Dict[str, float]:
        """Return the current signal weight map (read-only copy)."""
        return dict(self._weights)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        raw_interactions: List[Dict[str, Any]],
    ) -> InteractionMatrix:
        """Convert raw interaction dicts into an :class:`InteractionMatrix`.

        Each dict in *raw_interactions* **must** contain:

        * ``"user_id"`` — string
        * ``"garment_id"`` — string
        * at least one signal key matching a key in :data:`SIGNAL_WEIGHTS`

        Returns
        -------
        InteractionMatrix
            Dense 2-D matrix of aggregated weighted signals.
        """
        records = self._extract_records(raw_interactions)
        return self._aggregate(records)

    def extract_records(
        self,
        raw_interactions: List[Dict[str, Any]],
    ) -> List[InteractionRecord]:
        """Return the flat list of :class:`InteractionRecord` objects
        without aggregation (useful for debugging / inspection)."""
        return self._extract_records(raw_interactions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_records(
        self,
        raw_interactions: List[Dict[str, Any]],
    ) -> List[InteractionRecord]:
        records: List[InteractionRecord] = []
        for entry in raw_interactions:
            uid = entry.get("user_id")
            gid = entry.get("garment_id")
            if not uid or not gid:
                continue

            for signal, weight in self._weights.items():
                raw = entry.get(signal)
                if raw is None:
                    continue
                raw = float(raw)
                if raw <= 0:
                    continue

                # Compress counters
                if signal in _COUNTER_SIGNALS:
                    raw = math.log1p(raw)

                records.append(
                    InteractionRecord(
                        user_id=uid,
                        garment_id=gid,
                        signal_type=signal,
                        raw_value=raw,
                        weight=weight,
                    )
                )
        return records

    def _aggregate(
        self,
        records: List[InteractionRecord],
    ) -> InteractionMatrix:
        """Aggregate records into a dense matrix."""
        # Collect unique IDs preserving insertion order
        user_set: Dict[str, int] = {}
        garment_set: Dict[str, int] = {}

        for r in records:
            if r.user_id not in user_set:
                user_set[r.user_id] = len(user_set)
            if r.garment_id not in garment_set:
                garment_set[r.garment_id] = len(garment_set)

        n_users = len(user_set)
        n_garments = len(garment_set)

        data: List[List[float]] = [
            [0.0] * n_garments for _ in range(n_users)
        ]

        for r in records:
            u = user_set[r.user_id]
            g = garment_set[r.garment_id]
            data[u][g] += r.raw_value * r.weight

        user_ids = list(user_set.keys())
        garment_ids = list(garment_set.keys())

        return InteractionMatrix(
            user_ids=user_ids,
            garment_ids=garment_ids,
            data=data,
            user_to_idx=user_set,
            garment_to_idx=garment_set,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def stats(matrix: InteractionMatrix) -> Dict[str, Any]:
        """Return basic statistics about an :class:`InteractionMatrix`."""
        n_users = len(matrix.user_ids)
        n_garments = len(matrix.garment_ids)
        total_cells = n_users * n_garments
        non_zero = sum(
            1
            for row in matrix.data
            for v in row
            if v > 0
        )
        return {
            "n_users": n_users,
            "n_garments": n_garments,
            "total_cells": total_cells,
            "non_zero": non_zero,
            "sparsity": 1.0 - (non_zero / total_cells) if total_cells else 1.0,
        }
