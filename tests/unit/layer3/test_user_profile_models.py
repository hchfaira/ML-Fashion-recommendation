"""
Unit tests for User Profile models.
"""

import pytest
from datetime import datetime

from src.layer3_context.user_profile.models import (
    BodyShape,
    FaceShape,
    SkinTone,
    Undertone,
    HairColor,
    ContrastLevel,
    VisualWeight,
    BodyMetrics,
    FacialLandmarks,
    SkinAnalysis,
    HairAnalysis,
    DetectedClothing,
    StyleProfile,
    ProfileInput,
)


class TestEnums:
    """Tests for enum types."""
    
    def test_body_shape_values(self):
        """Test BodyShape enum has expected values."""
        assert BodyShape.RECTANGLE.value == "rectangle"
        assert BodyShape.TRIANGLE.value == "triangle"
        assert BodyShape.INVERTED_TRIANGLE.value == "inverted_triangle"
        assert BodyShape.HOURGLASS.value == "hourglass"
        assert BodyShape.OVAL.value == "oval"
    
    def test_face_shape_values(self):
        """Test FaceShape enum has expected values."""
        shapes = [s.value for s in FaceShape]
        assert "oval" in shapes
        assert "round" in shapes
        assert "square" in shapes
        assert "heart" in shapes
    
    def test_skin_tone_values(self):
        """Test SkinTone enum has expected values."""
        tones = [t.value for t in SkinTone]
        assert "very_light" in tones
        assert "medium" in tones
        assert "dark" in tones
    
    def test_undertone_values(self):
        """Test Undertone enum has expected values."""
        assert Undertone.WARM.value == "warm"
        assert Undertone.COOL.value == "cool"
        assert Undertone.NEUTRAL.value == "neutral"
    
    def test_hair_color_values(self):
        """Test HairColor enum has expected values."""
        colors = [c.value for c in HairColor]
        assert "black" in colors
        assert "blonde" in colors
        assert "red" in colors
        assert "grey" in colors
    
    def test_contrast_level_values(self):
        """Test ContrastLevel enum has expected values."""
        levels = [l.value for l in ContrastLevel]
        assert "very_low" in levels
        assert "medium" in levels
        assert "very_high" in levels
    
    def test_visual_weight_values(self):
        """Test VisualWeight enum has expected values."""
        weights = [w.value for w in VisualWeight]
        assert "light" in weights
        assert "medium" in weights
        assert "heavy" in weights


class TestBodyMetrics:
    """Tests for BodyMetrics dataclass."""
    
    def test_creation_basic(self):
        """Test creating BodyMetrics with basic info."""
        metrics = BodyMetrics(
            body_shape=BodyShape.RECTANGLE,
            height_cm=175.0,
            weight_kg=70.0,
        )
        assert metrics.body_shape == BodyShape.RECTANGLE
        assert metrics.height_cm == 175.0
        assert metrics.weight_kg == 70.0
    
    def test_bmi_calculation(self):
        """Test BMI calculation."""
        metrics = BodyMetrics(
            body_shape=BodyShape.RECTANGLE,
            height_cm=180.0,
            weight_kg=80.0,
            bmi=24.7,  # 80 / (1.8 * 1.8) ≈ 24.7
        )
        assert metrics.bmi == pytest.approx(24.7, 0.1)
    
    def test_optional_fields(self):
        """Test optional fields default to None."""
        metrics = BodyMetrics(body_shape=BodyShape.OVAL)
        assert metrics.height_cm is None
        assert metrics.weight_kg is None
        assert metrics.shoulder_width_px is None
        assert metrics.hip_width_px is None


class TestFacialLandmarks:
    """Tests for FacialLandmarks dataclass."""
    
    def test_creation(self):
        """Test creating FacialLandmarks."""
        landmarks = FacialLandmarks(
            face_height=200.0,
            face_width=150.0,
            jaw_width=140.0,
            cheekbone_width=160.0,
        )
        assert landmarks.face_height == 200.0
        assert landmarks.face_width == 150.0
    
    def test_face_ratio(self):
        """Test face ratio property."""
        landmarks = FacialLandmarks(
            face_height=200.0,
            face_width=150.0,
        )
        # width / height = 150 / 200 = 0.75
        assert landmarks.face_ratio == pytest.approx(0.75, 0.01)
    
    def test_face_ratio_missing_values(self):
        """Test face ratio with missing values."""
        landmarks = FacialLandmarks(
            face_height=None,
            face_width=150.0,
        )
        assert landmarks.face_ratio is None


class TestSkinAnalysis:
    """Tests for SkinAnalysis dataclass."""
    
    def test_creation(self):
        """Test creating SkinAnalysis."""
        analysis = SkinAnalysis(
            skin_tone=SkinTone.MEDIUM,
            undertone=Undertone.WARM,
            dominant_skin_lab=(55.0, 15.0, 20.0),
            dominant_skin_rgb=(180, 150, 130),
        )
        assert analysis.skin_tone == SkinTone.MEDIUM
        assert analysis.undertone == Undertone.WARM
    
    def test_default_confidence(self):
        """Test default confidence value."""
        analysis = SkinAnalysis(
            skin_tone=SkinTone.LIGHT,
            undertone=Undertone.NEUTRAL,
        )
        assert analysis.skin_tone_confidence == 0.0


class TestHairAnalysis:
    """Tests for HairAnalysis dataclass."""
    
    def test_creation(self):
        """Test creating HairAnalysis."""
        analysis = HairAnalysis(
            hair_color=HairColor.DARK_BROWN,
            dominant_hair_rgb=(50, 30, 20),
            dominant_hair_lab=(25.0, 10.0, 15.0),
            hair_mask_coverage=0.15,
        )
        assert analysis.hair_color == HairColor.DARK_BROWN
        assert analysis.hair_mask_coverage == 0.15


class TestDetectedClothing:
    """Tests for DetectedClothing dataclass."""
    
    def test_creation_empty(self):
        """Test creating empty DetectedClothing."""
        clothing = DetectedClothing()
        assert clothing.top_type is None
        assert clothing.bottom_type is None
    
    def test_creation_with_top(self):
        """Test creating DetectedClothing with top."""
        clothing = DetectedClothing(
            top_type="t-shirt",
            top_color="blue",
            top_pattern="solid",
        )
        assert clothing.top_type == "t-shirt"
        assert clothing.top_color == "blue"


class TestStyleProfile:
    """Tests for StyleProfile dataclass."""
    
    def test_creation_minimal(self):
        """Test creating StyleProfile with minimal data."""
        profile = StyleProfile()
        assert profile.body_metrics is None
        assert profile.face_shape is None
        assert profile.skin_analysis is None
    
    def test_creation_full(self):
        """Test creating StyleProfile with full data."""
        body = BodyMetrics(body_shape=BodyShape.RECTANGLE, height_cm=175)
        skin = SkinAnalysis(skin_tone=SkinTone.MEDIUM, undertone=Undertone.WARM)
        hair = HairAnalysis(hair_color=HairColor.DARK_BROWN)
        
        profile = StyleProfile(
            body_metrics=body,
            skin_analysis=skin,
            hair_analysis=hair,
            contrast_level=ContrastLevel.HIGH,
            visual_weight=VisualWeight.MEDIUM,
        )
        
        assert profile.body_metrics.body_shape == BodyShape.RECTANGLE
        assert profile.skin_analysis.skin_tone == SkinTone.MEDIUM
        assert profile.contrast_level == ContrastLevel.HIGH
    
    def test_to_dict(self):
        """Test converting StyleProfile to dictionary."""
        profile = StyleProfile(
            body_shape=BodyShape.HOURGLASS,
            skin_tone=SkinTone.LIGHT,
            undertone=Undertone.COOL,
            contrast_level=ContrastLevel.MEDIUM,
        )
        
        result = profile.to_dict()
        
        assert isinstance(result, dict)
        assert "body_type" in result
        assert "skin_tone" in result
        assert "contrast_level" in result
    
    def test_to_json(self):
        """Test converting StyleProfile to JSON."""
        profile = StyleProfile(
            contrast_level=ContrastLevel.HIGH,
            visual_weight=VisualWeight.HEAVY,
        )
        
        json_str = profile.to_json()
        
        assert isinstance(json_str, str)
        assert "contrast_level" in json_str
        assert "visual_weight" in json_str


class TestProfileInput:
    """Tests for ProfileInput dataclass."""
    
    def test_creation_image_only(self):
        """Test creating ProfileInput with just image."""
        input_data = ProfileInput(image_path="/path/to/image.jpg")
        assert input_data.image_path == "/path/to/image.jpg"
        assert input_data.height_cm is None
        assert input_data.weight_kg is None
    
    def test_creation_with_metadata(self):
        """Test creating ProfileInput with metadata."""
        input_data = ProfileInput(
            image_path="/path/to/image.jpg",
            height_cm=180.0,
            weight_kg=75.0,
        )
        assert input_data.height_cm == 180.0
        assert input_data.weight_kg == 75.0
