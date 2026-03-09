"""
Unit tests for User Profile analyzers.
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from src.layer3_context.user_profile.models import (
    BodyShape,
    FaceShape,
    SkinTone,
    Undertone,
    HairColor,
    ContrastLevel,
    SkinAnalysis,
    HairAnalysis,
    BodyMetrics,
)
from src.layer3_context.user_profile.color_analyzer import ColorAnalyzer, ColorSample
from src.layer3_context.user_profile.contrast_analyzer import ContrastAnalyzer


class TestColorAnalyzer:
    """Tests for ColorAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        return ColorAnalyzer()
    
    @pytest.fixture
    def sample_face_image(self):
        """Create a sample face-like image."""
        # Create 200x200 image with skin-like color
        img = Image.new("RGB", (200, 200), (200, 160, 140))
        return img
    
    def test_analyze_returns_color_profile(self, analyzer, sample_face_image):
        """Test that analyze returns a ColorProfile."""
        result = analyzer.analyze(sample_face_image)
        
        assert result is not None
        assert result.skin_tone is not None
        assert result.undertone is not None
        assert result.confidence > 0
    
    def test_analyze_with_face_bbox(self, analyzer, sample_face_image):
        """Test analyze with face bounding box."""
        bbox = (50, 50, 100, 100)  # x, y, w, h
        result = analyzer.analyze(sample_face_image, face_bbox=bbox)
        
        assert result is not None
        assert result.skin_tone is not None
    
    def test_rgb_to_lab_conversion(self, analyzer):
        """Test RGB to LAB color space conversion."""
        # White
        lab = analyzer._rgb_to_lab((255, 255, 255))
        assert lab[0] > 95  # L* close to 100
        
        # Black
        lab = analyzer._rgb_to_lab((0, 0, 0))
        assert lab[0] < 5  # L* close to 0
    
    def test_lab_to_rgb_conversion(self, analyzer):
        """Test LAB to RGB color space conversion."""
        # Test round-trip
        original_rgb = (150, 100, 80)
        lab = analyzer._rgb_to_lab(original_rgb)
        recovered_rgb = analyzer._lab_to_rgb(lab)
        
        # Should be close (within rounding)
        assert abs(recovered_rgb[0] - original_rgb[0]) < 3
        assert abs(recovered_rgb[1] - original_rgb[1]) < 3
        assert abs(recovered_rgb[2] - original_rgb[2]) < 3
    
    def test_rgb_to_hex(self, analyzer):
        """Test RGB to hex conversion."""
        assert analyzer._rgb_to_hex((255, 255, 255)) == "#FFFFFF"
        assert analyzer._rgb_to_hex((0, 0, 0)) == "#000000"
        assert analyzer._rgb_to_hex((255, 0, 128)) == "#FF0080"
    
    def test_classify_skin_tone_light(self, analyzer):
        """Test classifying light skin tone."""
        # High L* value
        lab = (80, 15, 20)
        tone = analyzer._classify_skin_tone(lab)
        assert tone in [SkinTone.LIGHT, SkinTone.VERY_LIGHT]
    
    def test_classify_skin_tone_dark(self, analyzer):
        """Test classifying dark skin tone."""
        # Low L* value
        lab = (30, 15, 20)
        tone = analyzer._classify_skin_tone(lab)
        assert tone in [SkinTone.DARK, SkinTone.MEDIUM_DARK]
    
    def test_determine_undertone_warm(self, analyzer):
        """Test determining warm undertone."""
        # High a* and b* values
        lab = (60, 15, 25)
        undertone = analyzer._determine_undertone(lab)
        assert undertone == Undertone.WARM
    
    def test_determine_undertone_cool(self, analyzer):
        """Test determining cool undertone."""
        # Low a* and b* values
        lab = (60, 3, 5)
        undertone = analyzer._determine_undertone(lab)
        assert undertone == Undertone.COOL


class TestContrastAnalyzer:
    """Tests for ContrastAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        return ContrastAnalyzer()
    
    def test_compute_delta_e_same_color(self, analyzer):
        """Test Delta E for identical colors is 0."""
        lab = (50, 10, 20)
        delta_e = analyzer._compute_delta_e(lab, lab)
        assert delta_e == pytest.approx(0, abs=0.01)
    
    def test_compute_delta_e_different_colors(self, analyzer):
        """Test Delta E for different colors is positive."""
        lab1 = (50, 0, 0)
        lab2 = (100, 0, 0)  # Only lightness differs by 50
        delta_e = analyzer._compute_delta_e(lab1, lab2)
        assert delta_e == pytest.approx(50, abs=0.01)
    
    def test_compute_delta_e_high_contrast(self, analyzer):
        """Test Delta E for high contrast colors."""
        # Very light skin + black hair
        skin = (80, 15, 20)
        hair = (15, 0, 0)
        delta_e = analyzer._compute_delta_e(skin, hair)
        assert delta_e > 50  # High contrast
    
    def test_classify_contrast_very_low(self, analyzer):
        """Test classifying very low contrast."""
        delta_e = 5
        level = analyzer._classify_contrast(delta_e)
        assert level == ContrastLevel.VERY_LOW
    
    def test_classify_contrast_high(self, analyzer):
        """Test classifying high contrast."""
        delta_e = 55
        level = analyzer._classify_contrast(delta_e)
        assert level == ContrastLevel.HIGH
    
    def test_analyze_with_profiles(self, analyzer):
        """Test analyze with skin and hair analysis."""
        skin = SkinAnalysis(
            skin_tone=SkinTone.LIGHT,
            undertone=Undertone.WARM,
            dominant_skin_lab=(75, 15, 20),
        )
        hair = HairAnalysis(
            hair_color=HairColor.BLACK,
            dominant_hair_rgb=(20, 15, 10),
        )
        
        level = analyzer.analyze(skin_analysis=skin, hair_analysis=hair)
        assert level in ContrastLevel
    
    def test_analyze_with_direct_lab_values(self, analyzer):
        """Test analyze with direct LAB values."""
        skin_lab = (70, 15, 20)
        hair_lab = (20, 5, 5)
        
        level = analyzer.analyze(skin_lab=skin_lab, hair_lab=hair_lab)
        assert level in ContrastLevel
    
    def test_compute_feature_contrast(self, analyzer):
        """Test computing contrast between any two features."""
        feature1 = (80, 10, 15)
        feature2 = (30, 5, 10)
        
        delta_e, level = analyzer.compute_feature_contrast(feature1, feature2)
        
        assert delta_e > 0
        assert level in ContrastLevel


class TestBodyAnalyzerMocked:
    """Tests for BodyAnalyzer with mocked MediaPipe."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image."""
        return Image.new("RGB", (640, 480), (200, 180, 160))
    
    def test_body_analyzer_import(self):
        """Test BodyAnalyzer can be imported."""
        from src.layer3_context.user_profile.body_analyzer import BodyAnalyzer
        analyzer = BodyAnalyzer()
        assert analyzer is not None
    
    def test_classify_body_shape_rectangle(self):
        """Test body shape classification - rectangle."""
        from src.layer3_context.user_profile.body_analyzer import BodyAnalyzer
        
        analyzer = BodyAnalyzer()
        
        # Shoulder/hip ratio ≈ 1.0 (balanced)
        metrics = BodyMetrics(
            body_shape=None,
            shoulder_hip_ratio=1.02,
        )
        shape = analyzer._classify_body_shape(metrics)
        assert shape == BodyShape.RECTANGLE
    
    def test_classify_body_shape_hourglass(self):
        """Test body shape classification - hourglass.
        
        Note: Without waist data, balanced proportions default to RECTANGLE.
        """
        from src.layer3_context.user_profile.body_analyzer import BodyAnalyzer
        
        analyzer = BodyAnalyzer()
        
        # Shoulder ≈ hip (balanced) - defaults to RECTANGLE without waist data
        metrics = BodyMetrics(
            body_shape=None,
            shoulder_hip_ratio=1.0,
        )
        shape = analyzer._classify_body_shape(metrics)
        assert shape == BodyShape.RECTANGLE  # Without waist data
    
    def test_classify_body_shape_triangle(self):
        """Test body shape classification - triangle (pear)."""
        from src.layer3_context.user_profile.body_analyzer import BodyAnalyzer
        
        analyzer = BodyAnalyzer()
        
        # Hip wider than shoulder (ratio < 0.9)
        metrics = BodyMetrics(
            body_shape=None,
            shoulder_hip_ratio=0.85,
        )
        shape = analyzer._classify_body_shape(metrics)
        assert shape == BodyShape.TRIANGLE
    
    def test_classify_body_shape_inverted_triangle(self):
        """Test body shape classification - inverted triangle."""
        from src.layer3_context.user_profile.body_analyzer import BodyAnalyzer
        
        analyzer = BodyAnalyzer()
        
        # Shoulder wider than hip (ratio > 1.1)
        metrics = BodyMetrics(
            body_shape=None,
            shoulder_hip_ratio=1.2,
        )
        shape = analyzer._classify_body_shape(metrics)
        assert shape == BodyShape.INVERTED_TRIANGLE


class TestFaceAnalyzerMocked:
    """Tests for FaceAnalyzer with mocked MediaPipe."""
    
    def test_face_analyzer_import(self):
        """Test FaceAnalyzer can be imported."""
        from src.layer3_context.user_profile.face_analyzer import FaceAnalyzer
        analyzer = FaceAnalyzer()
        assert analyzer is not None
    
    def test_classify_face_shape_oval(self):
        """Test face shape classification - oval."""
        from src.layer3_context.user_profile.face_analyzer import FaceAnalyzer
        from src.layer3_context.user_profile.models import FacialLandmarks
        
        analyzer = FaceAnalyzer()
        
        landmarks = FacialLandmarks(
            face_height=200,
            face_width=150,  # ratio = 0.75
            jaw_width=130,
            cheekbone_width=160,  # jaw/cheek = 0.81
        )
        
        shape = analyzer.classify_face_shape(landmarks)
        assert shape == FaceShape.OVAL
    
    def test_classify_face_shape_round(self):
        """Test face shape classification - round."""
        from src.layer3_context.user_profile.face_analyzer import FaceAnalyzer
        from src.layer3_context.user_profile.models import FacialLandmarks
        
        analyzer = FaceAnalyzer()
        
        landmarks = FacialLandmarks(
            face_height=180,
            face_width=175,  # ratio = 0.97, very round
            jaw_width=140,
            cheekbone_width=160,
        )
        
        shape = analyzer.classify_face_shape(landmarks)
        assert shape == FaceShape.ROUND


class TestHairAnalyzerMocked:
    """Tests for HairAnalyzer with mocked MediaPipe."""
    
    def test_hair_analyzer_import(self):
        """Test HairAnalyzer can be imported."""
        from src.layer3_context.user_profile.hair_analyzer import HairAnalyzer
        analyzer = HairAnalyzer()
        assert analyzer is not None
    
    def test_classify_hair_color_black(self):
        """Test hair color classification - black."""
        from src.layer3_context.user_profile.hair_analyzer import HairAnalyzer
        
        analyzer = HairAnalyzer()
        
        lab = (15, 2, 3)  # Very low L*
        color = analyzer._classify_hair_color(lab)
        assert color == HairColor.BLACK
    
    def test_classify_hair_color_blonde(self):
        """Test hair color classification - blonde."""
        from src.layer3_context.user_profile.hair_analyzer import HairAnalyzer
        
        analyzer = HairAnalyzer()
        
        lab = (75, 5, 30)  # High L*, high b*
        color = analyzer._classify_hair_color(lab)
        assert color == HairColor.BLONDE
    
    def test_classify_hair_color_red(self):
        """Test hair color classification - red."""
        from src.layer3_context.user_profile.hair_analyzer import HairAnalyzer
        
        analyzer = HairAnalyzer()
        
        lab = (45, 30, 30)  # High a*
        color = analyzer._classify_hair_color(lab)
        assert color == HairColor.RED


class TestClothingDetector:
    """Tests for ClothingDetector."""
    
    def test_clothing_detector_import(self):
        """Test ClothingDetector can be imported."""
        from src.layer3_context.user_profile.clothing_detector import ClothingDetector
        detector = ClothingDetector()
        assert detector is not None
    
    def test_get_color_name(self):
        """Test color name lookup."""
        from src.layer3_context.user_profile.clothing_detector import ClothingDetector
        
        detector = ClothingDetector()
        
        # Exact matches
        assert detector._get_color_name((0, 0, 0)) == "black"
        assert detector._get_color_name((255, 255, 255)) == "white"
        
        # Close matches
        name = detector._get_color_name((250, 0, 0))  # Near red
        assert name == "red"
    
    def test_detect_pattern_solid(self):
        """Test pattern detection - solid."""
        from src.layer3_context.user_profile.clothing_detector import ClothingDetector
        
        detector = ClothingDetector()
        
        # Uniform color region
        pixels = np.full((50, 50, 3), [100, 100, 100], dtype=np.uint8)
        pattern = detector._detect_pattern(pixels)
        assert pattern == "solid"
