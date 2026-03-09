"""
Unit tests for User Profile Pipeline.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import tempfile
import os

from src.layer3_context.user_profile.pipeline import (
    StyleProfilePipeline,
    PipelineConfig,
    PipelineResult,
    AnalysisStage,
    extract_style_profile,
)
from src.layer3_context.user_profile.models import (
    BodyShape,
    FaceShape,
    SkinTone,
    Undertone,
    ContrastLevel,
    VisualWeight,
    BodyMetrics,
    SkinAnalysis,
    HairAnalysis,
    StyleProfile,
)
from src.layer3_context.user_profile.profile_builder import ProfileBuilder


class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig.default()
        
        assert config.enable_body_analysis is True
        assert config.enable_face_analysis is True
        assert config.enable_color_analysis is True
        assert config.enable_hair_analysis is True
        assert config.enable_contrast_analysis is True
        assert config.enable_clothing_detection is False  # Off by default
    
    def test_minimal_config(self):
        """Test minimal configuration."""
        config = PipelineConfig.minimal()
        
        assert config.enable_body_analysis is True
        assert config.enable_face_analysis is False
        assert config.enable_hair_analysis is False
        assert config.enable_clothing_detection is False
    
    def test_full_config(self):
        """Test full configuration."""
        config = PipelineConfig.full()
        
        assert config.enable_clothing_detection is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            enable_body_analysis=False,
            enable_face_analysis=True,
            height_cm=175.0,
            weight_kg=70.0,
        )
        
        assert config.enable_body_analysis is False
        assert config.height_cm == 175.0
        assert config.weight_kg == 70.0


class TestPipelineResult:
    """Tests for PipelineResult."""
    
    def test_result_creation(self):
        """Test creating a pipeline result."""
        profile = StyleProfile()
        
        result = PipelineResult(
            profile=profile,
            stages_completed=["body_detection", "face_detection"],
            stages_failed=["hair_analysis"],
            intermediate_results={},
            processing_time_ms=150.5,
        )
        
        assert result.profile is not None
        assert len(result.stages_completed) == 2
        assert len(result.stages_failed) == 1
        assert result.processing_time_ms == 150.5
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        profile = StyleProfile(contrast_level=ContrastLevel.HIGH)
        
        result = PipelineResult(
            profile=profile,
            stages_completed=["body_detection"],
            stages_failed=[],
            intermediate_results={},
            processing_time_ms=100.0,
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert "profile" in data
        assert "stages_completed" in data
        assert "processing_time_ms" in data
    
    def test_result_to_json(self):
        """Test converting result to JSON."""
        profile = StyleProfile()
        
        result = PipelineResult(
            profile=profile,
            stages_completed=[],
            stages_failed=[],
            intermediate_results={},
            processing_time_ms=50.0,
        )
        
        json_str = result.to_json()
        
        assert isinstance(json_str, str)
        assert "stages_completed" in json_str


class TestAnalysisStage:
    """Tests for AnalysisStage enum."""
    
    def test_stage_values(self):
        """Test that stages have expected values."""
        assert AnalysisStage.BODY_DETECTION.value == "body_detection"
        assert AnalysisStage.FACE_DETECTION.value == "face_detection"
        assert AnalysisStage.SKIN_TONE.value == "skin_tone"
        assert AnalysisStage.HAIR_ANALYSIS.value == "hair_analysis"
        assert AnalysisStage.CONTRAST.value == "contrast"
        assert AnalysisStage.CLOTHING.value == "clothing"
        assert AnalysisStage.PROFILE_BUILD.value == "profile_build"


class TestProfileBuilder:
    """Tests for ProfileBuilder."""
    
    def test_builder_pattern(self):
        """Test builder pattern method chaining."""
        builder = ProfileBuilder()
        
        result = (
            builder
            .set_body_metrics(BodyMetrics(body_shape=BodyShape.RECTANGLE))
            .set_face_shape(FaceShape.OVAL)
            .set_contrast_level(ContrastLevel.HIGH)
        )
        
        assert result is builder
    
    def test_build_minimal_profile(self):
        """Test building a minimal profile."""
        builder = ProfileBuilder()
        builder.set_contrast_level(ContrastLevel.MEDIUM)
        
        profile = builder.build()
        
        assert profile is not None
        assert isinstance(profile, StyleProfile)
        assert profile.contrast_level == ContrastLevel.MEDIUM
    
    def test_build_full_profile(self):
        """Test building a full profile."""
        body = BodyMetrics(body_shape=BodyShape.HOURGLASS, height_cm=168)
        skin = SkinAnalysis(skin_tone=SkinTone.MEDIUM, undertone=Undertone.WARM)
        
        builder = ProfileBuilder()
        builder.set_body_metrics(body)
        builder.set_skin_analysis(skin)
        builder.set_contrast_level(ContrastLevel.HIGH)
        
        profile = builder.build()
        
        assert profile.body_metrics is not None
        assert profile.body_metrics.body_shape == BodyShape.HOURGLASS
        assert profile.skin_analysis.skin_tone == SkinTone.MEDIUM
        assert profile.contrast_level == ContrastLevel.HIGH
    
    def test_compute_visual_weight_high_contrast(self):
        """Test visual weight computation with high contrast."""
        builder = ProfileBuilder()
        
        skin = SkinAnalysis(skin_tone=SkinTone.VERY_DARK)
        
        weight = builder._compute_visual_weight(
            skin_analysis=skin,
            hair_analysis=None,
            contrast_level=ContrastLevel.VERY_HIGH,
        )
        
        # High contrast = heavier visual weight
        assert weight in [VisualWeight.HEAVY, VisualWeight.MEDIUM_HEAVY]
    
    def test_builder_reset(self):
        """Test resetting builder."""
        builder = ProfileBuilder()
        builder.set_contrast_level(ContrastLevel.HIGH)
        
        builder.reset()
        
        profile = builder.build()
        assert profile.contrast_level is None


class TestStyleProfilePipeline:
    """Tests for StyleProfilePipeline."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return Image.new("RGB", (640, 480), (200, 180, 160))
    
    @pytest.fixture
    def temp_image_file(self, sample_image):
        """Create a temporary image file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            sample_image.save(f.name)
            yield f.name
            os.unlink(f.name)
    
    def test_pipeline_creation_default(self):
        """Test creating pipeline with default config."""
        pipeline = StyleProfilePipeline()
        
        assert pipeline.config is not None
        assert pipeline.config.enable_body_analysis is True
    
    def test_pipeline_creation_custom_config(self):
        """Test creating pipeline with custom config."""
        config = PipelineConfig(
            enable_body_analysis=False,
            enable_clothing_detection=True,
        )
        
        pipeline = StyleProfilePipeline(config)
        
        assert pipeline.config.enable_body_analysis is False
        assert pipeline.config.enable_clothing_detection is True
    
    def test_pipeline_lazy_initialization(self):
        """Test that analyzers are lazily initialized."""
        pipeline = StyleProfilePipeline()
        
        # Initially None
        assert pipeline._body_analyzer is None
        assert pipeline._face_analyzer is None
        
        # Accessing property initializes
        _ = pipeline.color_analyzer
        assert pipeline._color_analyzer is not None
    
    @patch("src.layer3_context.user_profile.body_analyzer.BodyAnalyzer.analyze")
    @patch("src.layer3_context.user_profile.face_analyzer.FaceAnalyzer.analyze")
    @patch("src.layer3_context.user_profile.color_analyzer.ColorAnalyzer.analyze")
    @patch("src.layer3_context.user_profile.hair_analyzer.HairAnalyzer.analyze")
    def test_pipeline_analyze_with_mocks(
        self,
        mock_hair_analyze,
        mock_color_analyze,
        mock_face_analyze,
        mock_body_analyze,
        sample_image,
    ):
        """Test pipeline analyze with mocked analyzers."""
        # Setup mocks
        mock_body_analyze.return_value = BodyMetrics(
            body_shape=BodyShape.RECTANGLE,
        )
        mock_face_analyze.return_value = None
        mock_color_analyze.return_value = SkinAnalysis(
            skin_tone=SkinTone.MEDIUM,
            undertone=Undertone.NEUTRAL,
            skin_tone_confidence=0.85,
            undertone_confidence=0.85,
        )
        mock_hair_analyze.return_value = None
        
        # Run pipeline
        pipeline = StyleProfilePipeline()
        result = pipeline.analyze(sample_image)
        
        # Verify result
        assert result is not None
        assert isinstance(result, PipelineResult)
        assert result.profile is not None
        assert AnalysisStage.PROFILE_BUILD.value in result.stages_completed
    
    def test_pipeline_analyze_from_path(self, temp_image_file):
        """Test pipeline can analyze from file path."""
        config = PipelineConfig.minimal()
        config.enable_body_analysis = False
        config.enable_color_analysis = False
        
        pipeline = StyleProfilePipeline(config)
        result = pipeline.analyze(temp_image_file)
        
        assert result is not None
        assert result.profile is not None
    
    def test_pipeline_with_metadata(self, sample_image):
        """Test pipeline with height/weight metadata."""
        config = PipelineConfig(
            enable_body_analysis=False,
            enable_face_analysis=False,
            enable_hair_analysis=False,
            enable_contrast_analysis=False,
            height_cm=175.0,
            weight_kg=70.0,
        )
        
        pipeline = StyleProfilePipeline(config)
        result = pipeline.analyze(sample_image)
        
        assert result is not None
    
    def test_pipeline_context_manager(self, sample_image):
        """Test pipeline as context manager."""
        config = PipelineConfig.minimal()
        config.enable_body_analysis = False
        config.enable_color_analysis = False
        
        with StyleProfilePipeline(config) as pipeline:
            result = pipeline.analyze(sample_image)
            assert result is not None
    
    def test_pipeline_result_contains_timing(self, sample_image):
        """Test that result contains processing time."""
        config = PipelineConfig.minimal()
        config.enable_body_analysis = False
        config.enable_color_analysis = False
        
        pipeline = StyleProfilePipeline(config)
        result = pipeline.analyze(sample_image)
        
        assert result.processing_time_ms > 0


class TestExtractStyleProfile:
    """Tests for extract_style_profile convenience function."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return Image.new("RGB", (640, 480), (200, 180, 160))
    
    @patch("src.layer3_context.user_profile.pipeline.StyleProfilePipeline.analyze")
    def test_convenience_function(self, mock_analyze, sample_image):
        """Test the convenience function."""
        # Setup mock
        mock_result = PipelineResult(
            profile=StyleProfile(contrast_level=ContrastLevel.MEDIUM),
            stages_completed=[],
            stages_failed=[],
            intermediate_results={},
            processing_time_ms=100.0,
        )
        mock_analyze.return_value = mock_result
        
        # Call convenience function
        profile = extract_style_profile(sample_image, height_cm=175)
        
        # Verify
        assert profile is not None
        assert profile.contrast_level == ContrastLevel.MEDIUM


class TestIntegrationScenarios:
    """Integration-style tests for common scenarios."""
    
    @pytest.fixture
    def sample_person_image(self):
        """Create a more realistic sample image."""
        # 480x640 portrait orientation
        img = Image.new("RGB", (480, 640))
        
        # Simple gradients to simulate a person
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Head region (skin tone)
        draw.ellipse([180, 50, 300, 200], fill=(200, 160, 140))
        
        # Hair region
        draw.rectangle([180, 30, 300, 80], fill=(40, 30, 25))
        
        # Body/clothes
        draw.rectangle([150, 200, 330, 500], fill=(50, 80, 120))  # Blue top
        draw.rectangle([150, 400, 330, 640], fill=(30, 30, 35))   # Dark pants
        
        return img
    
    def test_minimal_analysis_scenario(self, sample_person_image):
        """Test minimal analysis on sample person."""
        config = PipelineConfig.minimal()
        config.enable_body_analysis = False  # Skip body to avoid MediaPipe
        
        pipeline = StyleProfilePipeline(config)
        result = pipeline.analyze(sample_person_image)
        
        assert result is not None
        assert result.profile is not None
        assert len(result.stages_failed) == 0 or "body_detection" not in result.stages_failed
    
    def test_color_only_scenario(self, sample_person_image):
        """Test color-only analysis."""
        config = PipelineConfig(
            enable_body_analysis=False,
            enable_face_analysis=False,
            enable_hair_analysis=False,
            enable_contrast_analysis=False,
            enable_clothing_detection=False,
        )
        
        pipeline = StyleProfilePipeline(config)
        result = pipeline.analyze(sample_person_image)
        
        # Color analysis should still run
        assert result.profile is not None
        assert AnalysisStage.SKIN_TONE.value in result.stages_completed
