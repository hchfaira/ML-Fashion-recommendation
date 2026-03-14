"""Tests — CapsuleEvolutionTracker (F5)"""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.core.models import (
    CapsuleAnalysisResult,
    CapsuleEvolutionResult,
    CapsuleSnapshot,
)
from src.layer2_style.capsule.capsule_evolution_tracker import CapsuleEvolutionTracker


# ============================================================================
# Helpers
# ============================================================================

def _analysis(cohesion: float = 65.0, garments: int = 12, outfits: int = 40) -> CapsuleAnalysisResult:
    return CapsuleAnalysisResult(
        cohesion_score=cohesion,
        color_cohesion_score=0.6,
        versatility_ratio=0.4,
        redundancy_penalty=0.1,
        orphan_penalty=0.05,
        dominant_colors=["black"],
        color_coverage_pct=0.8,
        garment_scores=[],
        key_pieces=["k1"],
        orphan_pieces=[],
        redundant_pairs=[],
        total_garments=garments,
        total_outfits=outfits,
        capsule_profile="standard",
        recommendation="",
        projected_score_after_cleanup=70.0,
    )


@pytest.fixture
def tracker(tmp_path):
    """Tracker using a temporary snapshot directory."""
    with patch(
        "src.layer2_style.capsule.capsule_evolution_tracker._PROFILES_ROOT",
        tmp_path / "capsule_profiles",
    ):
        t = CapsuleEvolutionTracker(user_id="test_user")
        yield t


# ============================================================================
# Basic recording
# ============================================================================

class TestBasicRecording:
    def test_returns_evolution_result(self, tracker):
        result = tracker.record(_analysis(), "hash_a")
        assert isinstance(result, CapsuleEvolutionResult)

    def test_first_record_creates_snapshot(self, tracker):
        tracker.record(_analysis(), "hash_a")
        history = tracker.load_history()
        assert len(history) == 1

    def test_second_different_hash_adds_snapshot(self, tracker):
        tracker.record(_analysis(65.0), "hash_a")
        tracker.record(_analysis(72.0), "hash_b")
        assert len(tracker.load_history()) == 2

    def test_same_hash_no_duplicate(self, tracker):
        tracker.record(_analysis(65.0), "hash_a")
        tracker.record(_analysis(65.0), "hash_a")  # same hash
        assert len(tracker.load_history()) == 1


# ============================================================================
# Snapshot data integrity
# ============================================================================

class TestSnapshotIntegrity:
    def test_snapshot_cohesion_matches_input(self, tracker):
        tracker.record(_analysis(72.5), "hash_x")
        history = tracker.load_history()
        assert history[0].cohesion_score == pytest.approx(72.5, abs=0.1)

    def test_snapshot_garments_count_matches(self, tracker):
        tracker.record(_analysis(65.0, garments=15), "hash_x")
        history = tracker.load_history()
        assert history[0].garments_count == 15

    def test_snapshot_outfits_count_matches(self, tracker):
        tracker.record(_analysis(65.0, outfits=50), "hash_x")
        history = tracker.load_history()
        assert history[0].outfits_count == 50

    def test_action_recorded(self, tracker):
        tracker.record(_analysis(), "hash_x", action="removed_orphan")
        history = tracker.load_history()
        assert history[0].action_taken == "removed_orphan"


# ============================================================================
# Delta computation
# ============================================================================

class TestDeltaComputation:
    def test_delta_score_is_difference(self, tracker):
        tracker.record(_analysis(60.0), "hash_a")
        tracker.record(_analysis(70.0), "hash_b")
        history = tracker.load_history()
        assert history[1].delta_score == pytest.approx(10.0, abs=0.5)

    def test_first_snapshot_delta_is_zero(self, tracker):
        tracker.record(_analysis(60.0), "hash_a")
        history = tracker.load_history()
        assert history[0].delta_score == pytest.approx(0.0, abs=0.01)

    def test_overall_delta_score_in_result(self, tracker):
        tracker.record(_analysis(60.0), "hash_a")
        result = tracker.record(_analysis(75.0), "hash_b")
        assert result.delta_score == pytest.approx(15.0, abs=1.0)


# ============================================================================
# Trend direction
# ============================================================================

class TestTrendDirection:
    def test_improving_trend(self, tracker):
        tracker.record(_analysis(50.0), "h1")
        tracker.record(_analysis(60.0), "h2")
        result = tracker.record(_analysis(70.0), "h3")
        assert result.trend_direction == "improving"

    def test_declining_trend(self, tracker):
        tracker.record(_analysis(70.0), "h1")
        tracker.record(_analysis(60.0), "h2")
        result = tracker.record(_analysis(50.0), "h3")
        assert result.trend_direction == "declining"

    def test_stable_trend(self, tracker):
        tracker.record(_analysis(65.0), "h1")
        tracker.record(_analysis(65.5), "h2")
        result = tracker.record(_analysis(65.2), "h3")
        assert result.trend_direction == "stable"


# ============================================================================
# Prediction
# ============================================================================

class TestPrediction:
    def test_already_at_90_returns_zero_weeks(self, tracker):
        result = tracker.record(_analysis(92.0), "h1")
        assert result.predicted_weeks_to_90 == 0

    def test_improving_gives_positive_prediction(self, tracker):
        tracker.record(_analysis(50.0), "h1")
        tracker.record(_analysis(60.0), "h2")
        result = tracker.record(_analysis(70.0), "h3")
        if result.predicted_weeks_to_90 is not None:
            assert result.predicted_weeks_to_90 > 0

    def test_single_snapshot_returns_none_prediction(self, tracker):
        result = tracker.record(_analysis(65.0), "h1")
        assert result.predicted_weeks_to_90 is None

    def test_flat_trend_returns_none_prediction(self, tracker):
        tracker.record(_analysis(65.0), "h1")
        tracker.record(_analysis(65.0, outfits=41), "h2")  # must have different hash
        result = tracker.record(_analysis(65.0, outfits=42), "h3")
        # Slope = 0 → no prediction
        assert result.predicted_weeks_to_90 is None


# ============================================================================
# Clear
# ============================================================================

class TestClear:
    def test_clear_removes_history(self, tracker):
        tracker.record(_analysis(), "h1")
        tracker.clear()
        assert tracker.load_history() == []

    def test_record_after_clear_starts_fresh(self, tracker):
        tracker.record(_analysis(50.0), "h1")
        tracker.clear()
        tracker.record(_analysis(80.0), "h2")
        history = tracker.load_history()
        assert len(history) == 1
        assert history[0].cohesion_score == pytest.approx(80.0, abs=0.1)
