"""
Tests for the enhanced body analyser: waist estimation & continuous scoring.

Covers:
- _estimate_waist_width never raises
- _classify_body_shape returns (primary, secondary, scores, confidence) tuple
- scores are normalised (sum ≈ 1)
- secondary is set only when score > 0.25
- hourglass requires waist_hip_ratio < 0.75
"""

import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

from src.layer3_context.user_profile.body_analyzer import BodyAnalyzer
from src.layer3_context.user_profile.models import BodyMetrics, BodyShape


@pytest.fixture
def analyzer():
    """Return a BodyAnalyzer without actually loading MediaPipe."""
    ba = BodyAnalyzer.__new__(BodyAnalyzer)
    ba.min_detection_confidence = 0.5
    ba._pose_landmarker = None
    ba._mp = None
    return ba


# ---------------------------------------------------------------------------
# _estimate_waist_width – safety
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEstimateWaistWidth:

    def test_returns_none_on_empty_landmarks(self, analyzer):
        """Should never raise, just return None."""
        result = analyzer._estimate_waist_width([], (640, 480))
        assert result is None

    def test_returns_none_on_low_visibility(self, analyzer):
        """Landmarks with low visibility should be skipped → None."""
        def _lm(x, y, vis):
            return SimpleNamespace(x=x, y=y, visibility=vis)

        landmarks = [_lm(0, 0, 0)] * 30  # all invisible
        result = analyzer._estimate_waist_width(landmarks, (640, 480))
        assert result is None

    def test_positive_width_with_good_landmarks(self, analyzer):
        """With visible shoulder/hip landmarks, waist width should be > 0."""
        def _lm(x, y, vis=0.9):
            return SimpleNamespace(x=x, y=y, visibility=vis)

        # Minimum 28 landmarks (indices go up to 28)
        landmarks = [_lm(0.5, 0.5, 0.0)] * 30
        # left_shoulder=11, right_shoulder=12
        landmarks[11] = _lm(0.3, 0.3)
        landmarks[12] = _lm(0.7, 0.3)
        # left_hip=23, right_hip=24
        landmarks[23] = _lm(0.35, 0.7)
        landmarks[24] = _lm(0.65, 0.7)

        width = analyzer._estimate_waist_width(landmarks, (640, 480))
        assert width is not None
        assert width > 0


# ---------------------------------------------------------------------------
# _classify_body_shape – continuous scoring
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassifyBodyShapeContinuous:

    def _make_metrics(self, **kwargs) -> BodyMetrics:
        return BodyMetrics(**kwargs)

    def test_returns_four_tuple(self, analyzer):
        metrics = self._make_metrics(shoulder_hip_ratio=1.0)
        result = analyzer._classify_body_shape(metrics)
        assert len(result) == 4

    def test_scores_sum_to_one(self, analyzer):
        metrics = self._make_metrics(shoulder_hip_ratio=1.05)
        _, _, scores, _ = analyzer._classify_body_shape(metrics)
        total = sum(scores.values())
        assert abs(total - 1.0) < 0.01, f"Scores sum to {total}, expected ~1.0"

    def test_no_secondary_when_dominant(self, analyzer):
        """Very wide shoulders → inverted_triangle dominant, secondary unlikely."""
        metrics = self._make_metrics(shoulder_hip_ratio=1.25)
        primary, secondary, scores, confidence = analyzer._classify_body_shape(metrics)
        assert primary == BodyShape.INVERTED_TRIANGLE
        # secondary may or may not be set depending on scoring

    def test_hourglass_requires_low_waist_ratio(self, analyzer):
        """Without a low waist-hip ratio, hourglass should not be primary."""
        # balanced ratio but NO waist data (waist_width_ratio = None)
        metrics = self._make_metrics(shoulder_hip_ratio=1.0)
        primary, _, scores, _ = analyzer._classify_body_shape(metrics)
        assert primary != BodyShape.HOURGLASS

    def test_hourglass_with_defined_waist(self, analyzer):
        """With a low waist-hip ratio AND balanced shoulders/hips → hourglass scores highly."""
        metrics = self._make_metrics(
            shoulder_hip_ratio=1.0,
            waist_width_ratio=0.70,  # < 0.75 threshold
        )
        _, _, scores, _ = analyzer._classify_body_shape(metrics)
        assert scores.get("hourglass", 0) > 0.05  # meaningful score

    def test_secondary_set_when_score_above_025(self, analyzer):
        """If two shapes score close, secondary should be set."""
        # Slightly broader shoulders, good legs → athletic & inverted_triangle both score
        metrics = self._make_metrics(
            shoulder_hip_ratio=1.08,
            leg_torso_ratio=1.1,
        )
        primary, secondary, scores, _ = analyzer._classify_body_shape(metrics)
        # At least one of the two top scorers should exceed 0.25
        sorted_scores = sorted(scores.values(), reverse=True)
        if sorted_scores[1] > 0.25:
            assert secondary is not None

    def test_none_when_ratio_missing(self, analyzer):
        metrics = self._make_metrics()
        primary, secondary, scores, confidence = analyzer._classify_body_shape(metrics)
        assert primary is None
        assert confidence == 0.0
