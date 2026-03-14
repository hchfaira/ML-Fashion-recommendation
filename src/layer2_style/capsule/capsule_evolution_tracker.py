"""
Capsule Evolution Tracker — Fonctionnalité 5
=============================================

Suit l'évolution du score de capsule au fil du temps.

Mécanismes :
  • Snapshots JSON différentiels stockés dans
    tests/output/capsule_profiles/{user_id}/snapshots.json
  • Hash MD5 (via WardrobeCapsuleAnalyzer.wardrobe_hash) pour
    détecter les changements avant tout recalcul
  • Régression linéaire sur les 3 derniers points pour la prédiction
  • Ne recalcule RIEN si le hash est identique au dernier snapshot
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.core import get_logger
from src.core.models import (
    CapsuleAnalysisResult,
    CapsuleEvolutionResult,
    CapsuleSnapshot,
)

logger = get_logger(__name__)

_PROFILES_ROOT = (
    Path(__file__).parent.parent.parent.parent
    / "tests"
    / "output"
    / "capsule_profiles"
)


class CapsuleEvolutionTracker:
    """
    Persists capsule snapshots and computes trend analytics.

    Usage:
        tracker = CapsuleEvolutionTracker(user_id="user_42")
        result  = tracker.record(analysis, wardrobe_hash, action="removed_redundant_piece")
        # If hash unchanged, result is computed from existing history only.
    """

    def __init__(self, user_id: str = "default") -> None:
        self._user_id = user_id
        self._snapshot_path = _PROFILES_ROOT / user_id / "snapshots.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        analysis: CapsuleAnalysisResult,
        wardrobe_hash: str,
        action: str = "",
    ) -> CapsuleEvolutionResult:
        """
        Persist a new snapshot (only if wardrobe changed) and return evolution analytics.

        Parameters
        ----------
        analysis:
            Latest CapsuleAnalysisResult.
        wardrobe_hash:
            MD5 from WardrobeCapsuleAnalyzer.wardrobe_hash(garments).
        action:
            Human-readable description of what changed (e.g. "removed_redundant_piece").
        """
        snapshots = self._load_snapshots()

        # Hash-based deduplication — skip re-record if wardrobe unchanged
        if snapshots and snapshots[-1].get("hash") == wardrobe_hash:
            logger.debug("CapsuleEvolutionTracker: wardrobe unchanged — no new snapshot")
            return self._build_result(snapshots, analysis)

        # Build new snapshot
        prev = snapshots[-1] if snapshots else None
        delta_score = round(analysis.cohesion_score - (prev["cohesion_score"] if prev else analysis.cohesion_score), 2)
        delta_outfits = analysis.total_outfits - (prev["outfits_count"] if prev else analysis.total_outfits)

        new_snap: dict = {
            "hash": wardrobe_hash,
            "date": date.today().isoformat(),
            "cohesion_score": round(analysis.cohesion_score, 2),
            "garments_count": analysis.total_garments,
            "outfits_count": analysis.total_outfits,
            "key_pieces_count": len(analysis.key_pieces),
            "orphans_count": len(analysis.orphan_pieces),
            "action_taken": action,
            "delta_score": delta_score,
            "delta_outfits": delta_outfits,
        }
        snapshots.append(new_snap)
        self._save_snapshots(snapshots)

        return self._build_result(snapshots, analysis)

    def load_history(self) -> List[CapsuleSnapshot]:
        """Return all recorded snapshots as typed objects."""
        raw = self._load_snapshots()
        return [self._dict_to_snapshot(s) for s in raw]

    def clear(self) -> None:
        """Remove all snapshots (useful in tests)."""
        if self._snapshot_path.exists():
            self._snapshot_path.unlink()

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def _build_result(
        self, snapshots: List[dict], analysis: CapsuleAnalysisResult
    ) -> CapsuleEvolutionResult:
        typed = [self._dict_to_snapshot(s) for s in snapshots]

        latest = analysis.cohesion_score
        first = snapshots[0]["cohesion_score"] if snapshots else latest
        delta_score = round(latest - first, 2)

        first_outfits = snapshots[0]["outfits_count"] if snapshots else analysis.total_outfits
        delta_outfits = analysis.total_outfits - first_outfits

        first_garments = snapshots[0]["garments_count"] if snapshots else analysis.total_garments
        outfit_ratio = (
            round(analysis.total_outfits / max(analysis.total_garments, 1), 2)
        )
        pieces_ratio = (
            round(analysis.total_garments / max(first_garments, 1), 2)
        )

        weeks_to_90 = self._predict_weeks_to_90(snapshots, latest)
        trend = self._trend_direction(snapshots)

        return CapsuleEvolutionResult(
            snapshots=typed,
            delta_score=delta_score,
            delta_outfits=delta_outfits,
            outfit_ratio=outfit_ratio,
            pieces_ratio=pieces_ratio,
            predicted_weeks_to_90=weeks_to_90,
            trend_direction=trend,
            latest_cohesion=round(latest, 2),
        )

    @staticmethod
    def _predict_weeks_to_90(snapshots: List[dict], current_score: float) -> Optional[int]:
        """Linear regression on last 3 snapshots to predict weeks to 90/100."""
        if current_score >= 90:
            return 0

        pts = snapshots[-3:]
        if len(pts) < 2:
            return None

        # x = week index, y = score
        n = len(pts)
        xs = list(range(n))
        ys = [p["cohesion_score"] for p in pts]

        # Simple linear regression
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        numerator = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
        denominator = sum((xs[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return None  # Flat — no progression

        slope = numerator / denominator
        if slope <= 0:
            return None  # Declining or flat — can't predict

        weeks = math.ceil((90 - current_score) / slope)
        return max(weeks, 0)

    @staticmethod
    def _trend_direction(snapshots: List[dict]) -> str:
        if len(snapshots) < 2:
            return "stable"
        scores = [s["cohesion_score"] for s in snapshots[-5:]]
        delta = scores[-1] - scores[0]
        if delta > 3:
            return "improving"
        if delta < -3:
            return "declining"
        return "stable"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_snapshots(self) -> List[dict]:
        if not self._snapshot_path.exists():
            return []
        try:
            with open(self._snapshot_path) as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            logger.warning("CapsuleEvolutionTracker: could not load snapshots — starting fresh")
            return []

    def _save_snapshots(self, snapshots: List[dict]) -> None:
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._snapshot_path, "w") as f:
                json.dump(snapshots, f, indent=2)
        except Exception as exc:
            logger.error("CapsuleEvolutionTracker: failed to save snapshots: %s", exc)

    @staticmethod
    def _dict_to_snapshot(raw: dict) -> CapsuleSnapshot:
        return CapsuleSnapshot(
            date=raw.get("date", date.today().isoformat()),
            cohesion_score=raw.get("cohesion_score", 0.0),
            garments_count=raw.get("garments_count", 0),
            outfits_count=raw.get("outfits_count", 0),
            key_pieces_count=raw.get("key_pieces_count", 0),
            orphans_count=raw.get("orphans_count", 0),
            action_taken=raw.get("action_taken", ""),
            delta_score=raw.get("delta_score", 0.0),
            delta_outfits=raw.get("delta_outfits", 0),
        )
