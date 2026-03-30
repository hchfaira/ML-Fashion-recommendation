"""
Tests for the enhanced 12-season color analysis in ColorAnalyzer.

Covers:
- chroma computation (clear / muted threshold at sqrt(a²+b²) == 20)
- depth mapping from SkinTone to light / medium / deep
- 12-season matrix classification
- FaceMesh fallback when MediaPipe is unavailable
"""

import math
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

from src.layer3_context.user_profile.color_analyzer import ColorAnalyzer
from src.layer3_context.user_profile.models import SkinTone, Undertone


# ---------------------------------------------------------------------------
# _compute_chroma
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeChroma:
    """Verify chroma classification boundary at C* = 20."""

    def test_clear_when_above_threshold(self):
        # a=15, b=15 → C* ≈ 21.2 → clear
        lab = (60.0, 15.0, 15.0)
        assert ColorAnalyzer._compute_chroma(lab) == "clear"

    def test_muted_when_below_threshold(self):
        # a=10, b=10 → C* ≈ 14.1 → muted
        lab = (60.0, 10.0, 10.0)
        assert ColorAnalyzer._compute_chroma(lab) == "muted"

    def test_boundary_exactly_20(self):
        # a=20, b=0 → C* = 20 → clear (>=)
        lab = (60.0, 20.0, 0.0)
        assert ColorAnalyzer._compute_chroma(lab) == "clear"

    def test_just_below_boundary(self):
        # a=19.9, b=0 → C* = 19.9 → muted
        lab = (60.0, 19.9, 0.0)
        assert ColorAnalyzer._compute_chroma(lab) == "muted"


# ---------------------------------------------------------------------------
# _skin_tone_to_depth
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSkinToneToDepth:
    """Verify 3-level depth bucketing."""

    @pytest.mark.parametrize("tone,expected", [
        (SkinTone.VERY_LIGHT, "light"),
        (SkinTone.LIGHT, "light"),
        (SkinTone.MEDIUM_LIGHT, "medium"),
        (SkinTone.MEDIUM, "medium"),
        (SkinTone.MEDIUM_DARK, "deep"),
        (SkinTone.DARK, "deep"),
        (SkinTone.VERY_DARK, "deep"),
    ])
    def test_depth_mapping(self, tone, expected):
        assert ColorAnalyzer._skin_tone_to_depth(tone) == expected


# ---------------------------------------------------------------------------
# _classify_season_12
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassifySeason12:
    """Verify the 12-season matrix look-up."""

    @pytest.fixture
    def analyzer(self):
        return ColorAnalyzer()

    def test_warm_light_clear_is_light_spring(self, analyzer):
        name, conf = analyzer._classify_season_12(Undertone.WARM, "light", "clear")
        assert name == "Light Spring"
        assert conf > 0

    def test_cool_deep_muted_is_cool_winter(self, analyzer):
        name, conf = analyzer._classify_season_12(Undertone.COOL, "deep", "muted")
        assert name == "Cool Winter"
        assert conf > 0

    def test_neutral_falls_back_to_cool_with_lower_confidence(self, analyzer):
        name, conf = analyzer._classify_season_12(Undertone.NEUTRAL, "medium", "clear")
        assert "Summer" in name or "Winter" in name  # cool bucket
        assert conf < 0.80  # lower than a definite warm/cool

    def test_all_12_combos_covered(self, analyzer):
        """Every valid (undertone_bucket, depth, chroma) combo should produce a result."""
        for tone_bucket in ("warm", "cool"):
            for depth in ("light", "medium", "deep"):
                for chroma in ("clear", "muted"):
                    undertone = Undertone.WARM if tone_bucket == "warm" else Undertone.COOL
                    name, conf = analyzer._classify_season_12(undertone, depth, chroma)
                    assert name != "Unknown", f"Missing entry for {tone_bucket}/{depth}/{chroma}"
                    assert 0 < conf <= 1


# ---------------------------------------------------------------------------
# _detect_face_zones fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDetectFaceZonesFallback:
    """When MediaPipe is not available, fixed regions must be returned."""

    def test_fallback_to_sample_regions_when_no_mediapipe(self):
        analyzer = ColorAnalyzer()
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        with patch("src.layer3_context.user_profile.color_analyzer._FACE_MESH_AVAILABLE", False):
            zones = analyzer._detect_face_zones(img)
        assert zones == analyzer.SAMPLE_REGIONS


# ---------------------------------------------------------------------------
# Full analyze() integration – new fields populated
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnalyzePopulatesNewFields:
    """analyze() must populate chroma, season_sub, season_confidence."""

    def test_new_fields_present(self):
        # Create a simple face-like image (uniform colour)
        img = Image.fromarray(np.full((100, 100, 3), 180, dtype=np.uint8))
        analyzer = ColorAnalyzer()
        result = analyzer.analyze(img)

        assert result.chroma in ("clear", "muted")
        assert isinstance(result.season_sub, str)
        assert result.season_confidence >= 0
