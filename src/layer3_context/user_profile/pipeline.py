"""
Style Profile Pipeline
======================

Main orchestrator for extracting user style profile from a photo.
Coordinates all analyzers and builds the final profile.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum
from PIL import Image
import json

from .models import StyleProfile
from .body_analyzer import BodyAnalyzer
from .face_analyzer import FaceAnalyzer
from .color_analyzer import ColorAnalyzer
from .hair_analyzer import HairAnalyzer
from .contrast_analyzer import ContrastAnalyzer
from .clothing_detector import ClothingDetector
from .profile_builder import ProfileBuilder

logger = logging.getLogger(__name__)


class AnalysisStage(str, Enum):
    """Stages in the profile extraction pipeline."""
    BODY_DETECTION = "body_detection"
    FACE_DETECTION = "face_detection"
    SKIN_TONE = "skin_tone"
    HAIR_ANALYSIS = "hair_analysis"
    CONTRAST = "contrast"
    CLOTHING = "clothing"
    PROFILE_BUILD = "profile_build"


@dataclass
class PipelineConfig:
    """Configuration for the style profile pipeline."""
    
    # Enable/disable stages
    enable_body_analysis: bool = True
    enable_face_analysis: bool = True
    enable_color_analysis: bool = True
    enable_hair_analysis: bool = True
    enable_contrast_analysis: bool = True
    enable_clothing_detection: bool = False  # Optional by default
    
    # User-provided metadata
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    
    # Detection confidence thresholds
    min_detection_confidence: float = 0.5
    min_face_confidence: float = 0.5
    
    # Output options
    save_intermediate: bool = False
    output_dir: Optional[str] = None
    
    @classmethod
    def default(cls) -> "PipelineConfig":
        """Get default configuration."""
        return cls()
    
    @classmethod
    def minimal(cls) -> "PipelineConfig":
        """Minimal analysis - just body and color."""
        return cls(
            enable_face_analysis=False,
            enable_hair_analysis=False,
            enable_contrast_analysis=False,
            enable_clothing_detection=False,
        )
    
    @classmethod
    def full(cls) -> "PipelineConfig":
        """Full analysis including clothing detection."""
        return cls(enable_clothing_detection=True)


@dataclass
class PipelineResult:
    """Result of the style profile extraction pipeline."""
    
    profile: StyleProfile
    stages_completed: list
    stages_failed: list
    intermediate_results: Dict[str, Any]
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "profile": self.profile.to_dict() if self.profile else None,
            "stages_completed": self.stages_completed,
            "stages_failed": self.stages_failed,
            "processing_time_ms": self.processing_time_ms,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class StyleProfilePipeline:
    """
    Main pipeline for extracting style profile from user photo.
    
    Coordinates multiple analyzers:
    1. Body Analyzer - body proportions and shape
    2. Face Analyzer - facial proportions and face shape
    3. Color Analyzer - skin tone and undertone
    4. Hair Analyzer - hair color
    5. Contrast Analyzer - contrast level
    6. Clothing Detector - current clothing (optional)
    7. Profile Builder - combines all into StyleProfile
    
    Usage:
        pipeline = StyleProfilePipeline()
        result = pipeline.analyze("user_photo.jpg", height_cm=175)
        print(result.profile)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig.default()
        
        # Initialize analyzers lazily
        self._body_analyzer: Optional[BodyAnalyzer] = None
        self._face_analyzer: Optional[FaceAnalyzer] = None
        self._color_analyzer: Optional[ColorAnalyzer] = None
        self._hair_analyzer: Optional[HairAnalyzer] = None
        self._contrast_analyzer: Optional[ContrastAnalyzer] = None
        self._clothing_detector: Optional[ClothingDetector] = None
        self._profile_builder: Optional[ProfileBuilder] = None
    
    @property
    def body_analyzer(self) -> BodyAnalyzer:
        if self._body_analyzer is None:
            self._body_analyzer = BodyAnalyzer(
                min_detection_confidence=self.config.min_detection_confidence
            )
        return self._body_analyzer
    
    @property
    def face_analyzer(self) -> FaceAnalyzer:
        if self._face_analyzer is None:
            self._face_analyzer = FaceAnalyzer(
                min_detection_confidence=self.config.min_face_confidence
            )
        return self._face_analyzer
    
    @property
    def color_analyzer(self) -> ColorAnalyzer:
        if self._color_analyzer is None:
            self._color_analyzer = ColorAnalyzer()
        return self._color_analyzer
    
    @property
    def hair_analyzer(self) -> HairAnalyzer:
        if self._hair_analyzer is None:
            self._hair_analyzer = HairAnalyzer()
        return self._hair_analyzer
    
    @property
    def contrast_analyzer(self) -> ContrastAnalyzer:
        if self._contrast_analyzer is None:
            self._contrast_analyzer = ContrastAnalyzer()
        return self._contrast_analyzer
    
    @property
    def clothing_detector(self) -> ClothingDetector:
        if self._clothing_detector is None:
            self._clothing_detector = ClothingDetector()
        return self._clothing_detector
    
    @property
    def profile_builder(self) -> ProfileBuilder:
        if self._profile_builder is None:
            self._profile_builder = ProfileBuilder()
        return self._profile_builder
    
    def analyze(
        self,
        image: Union[str, Path, Image.Image],
        height_cm: Optional[float] = None,
        weight_kg: Optional[float] = None,
    ) -> PipelineResult:
        """
        Analyze user photo and extract style profile.
        
        Args:
            image: Path to image or PIL Image
            height_cm: Optional user height in centimeters
            weight_kg: Optional user weight in kilograms
            
        Returns:
            PipelineResult with StyleProfile and metadata
        """
        import time
        start_time = time.time()
        
        stages_completed = []
        stages_failed = []
        intermediate = {}
        
        # Load image
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        else:
            img = image
        
        # Use config values if not provided
        height = height_cm or self.config.height_cm
        weight = weight_kg or self.config.weight_kg
        
        # Reset profile builder
        self.profile_builder.reset()
        
        # Stage 1: Body Detection
        body_metrics = None
        if self.config.enable_body_analysis:
            try:
                body_metrics = self.body_analyzer.analyze(
                    img,
                    height_cm=height,
                    weight_kg=weight,
                )
                self.profile_builder.set_body_metrics(body_metrics)
                stages_completed.append(AnalysisStage.BODY_DETECTION.value)
                intermediate["body"] = body_metrics
                logger.info("Body analysis complete")
            except Exception as e:
                logger.error(f"Body analysis failed: {e}")
                stages_failed.append(AnalysisStage.BODY_DETECTION.value)
        
        # Stage 2: Face Detection
        face_landmarks = None
        face_shape = None
        face_bbox = None
        if self.config.enable_face_analysis:
            try:
                face_landmarks = self.face_analyzer.analyze(img)
                if face_landmarks:
                    face_shape = self.face_analyzer.classify_face_shape(face_landmarks)
                    face_bbox = face_landmarks.bbox
                    self.profile_builder.set_face_landmarks(face_landmarks)
                    self.profile_builder.set_face_shape(face_shape)
                    intermediate["face_landmarks"] = face_landmarks
                    intermediate["face_shape"] = face_shape
                stages_completed.append(AnalysisStage.FACE_DETECTION.value)
                logger.info(f"Face analysis complete: shape={face_shape}")
            except Exception as e:
                logger.error(f"Face analysis failed: {e}")
                stages_failed.append(AnalysisStage.FACE_DETECTION.value)
        
        # Stage 3: Skin Tone Detection
        skin_analysis = None
        if self.config.enable_color_analysis:
            try:
                skin_analysis = self.color_analyzer.analyze(img, face_bbox=face_bbox)
                self.profile_builder.set_skin_analysis(skin_analysis)
                stages_completed.append(AnalysisStage.SKIN_TONE.value)
                intermediate["skin"] = skin_analysis
                logger.info(f"Color analysis complete: {skin_analysis.skin_tone}, {skin_analysis.undertone}")
            except Exception as e:
                logger.error(f"Color analysis failed: {e}")
                stages_failed.append(AnalysisStage.SKIN_TONE.value)
        
        # Stage 4: Hair Analysis
        hair_analysis = None
        if self.config.enable_hair_analysis:
            try:
                hair_analysis = self.hair_analyzer.analyze(img, face_bbox=face_bbox)
                self.profile_builder.set_hair_analysis(hair_analysis)
                stages_completed.append(AnalysisStage.HAIR_ANALYSIS.value)
                intermediate["hair"] = hair_analysis
                if hair_analysis:
                    logger.info(f"Hair analysis complete: {hair_analysis.hair_color}")
            except Exception as e:
                logger.error(f"Hair analysis failed: {e}")
                stages_failed.append(AnalysisStage.HAIR_ANALYSIS.value)
        
        # Stage 5: Contrast Analysis
        contrast_level = None
        if self.config.enable_contrast_analysis:
            try:
                contrast_level = self.contrast_analyzer.analyze(
                    skin_analysis=skin_analysis,
                    hair_analysis=hair_analysis,
                )
                self.profile_builder.set_contrast_level(contrast_level)
                stages_completed.append(AnalysisStage.CONTRAST.value)
                intermediate["contrast"] = contrast_level
                logger.info(f"Contrast analysis complete: {contrast_level}")
            except Exception as e:
                logger.error(f"Contrast analysis failed: {e}")
                stages_failed.append(AnalysisStage.CONTRAST.value)
        
        # Stage 6: Clothing Detection (Optional)
        detected_clothing = None
        if self.config.enable_clothing_detection:
            try:
                body_bbox = None
                if body_metrics and body_metrics.landmarks:
                    # Estimate body bbox from landmarks
                    body_bbox = self._estimate_body_bbox(body_metrics.landmarks, img.size)
                
                detected_clothing = self.clothing_detector.detect(img, body_bbox=body_bbox)
                self.profile_builder.set_detected_clothing(detected_clothing)
                stages_completed.append(AnalysisStage.CLOTHING.value)
                intermediate["clothing"] = detected_clothing
                if detected_clothing:
                    logger.info("Clothing detection complete")
            except Exception as e:
                logger.error(f"Clothing detection failed: {e}")
                stages_failed.append(AnalysisStage.CLOTHING.value)
        
        # Stage 7: Build Profile
        try:
            profile = self.profile_builder.build()
            stages_completed.append(AnalysisStage.PROFILE_BUILD.value)
            logger.info("Profile build complete")
        except Exception as e:
            logger.error(f"Profile build failed: {e}")
            stages_failed.append(AnalysisStage.PROFILE_BUILD.value)
            profile = None
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Save intermediate results if configured
        if self.config.save_intermediate and self.config.output_dir:
            self._save_intermediate(intermediate, self.config.output_dir)
        
        return PipelineResult(
            profile=profile,
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            intermediate_results=intermediate,
            processing_time_ms=processing_time,
        )
    
    def _estimate_body_bbox(
        self,
        landmarks: Dict,
        image_size: tuple
    ) -> tuple:
        """Estimate body bounding box from landmarks."""
        width, height = image_size
        
        # Get all x, y coordinates
        x_coords = []
        y_coords = []
        
        for name, (x, y) in landmarks.items():
            x_coords.append(x * width)
            y_coords.append(y * height)
        
        if not x_coords:
            return None
        
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _save_intermediate(self, results: Dict, output_dir: str):
        """Save intermediate results to disk."""
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable = {}
        for key, value in results.items():
            if hasattr(value, "to_dict"):
                serializable[key] = value.to_dict()
            elif hasattr(value, "__dict__"):
                serializable[key] = str(value)
            else:
                serializable[key] = str(value)
        
        with open(output_path / "intermediate_results.json", "w") as f:
            json.dump(serializable, f, indent=2)
    
    def close(self):
        """Release all resources."""
        if self._body_analyzer:
            self._body_analyzer.close()
        if self._face_analyzer:
            self._face_analyzer.close()
        if self._hair_analyzer:
            self._hair_analyzer.close()
        if self._clothing_detector:
            self._clothing_detector.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Convenience function
def extract_style_profile(
    image: Union[str, Path, Image.Image],
    height_cm: Optional[float] = None,
    weight_kg: Optional[float] = None,
    enable_clothing: bool = False,
) -> StyleProfile:
    """
    Convenience function to extract style profile from image.
    
    Args:
        image: Path to image or PIL Image
        height_cm: Optional user height in centimeters
        weight_kg: Optional user weight in kilograms
        enable_clothing: Whether to detect current clothing
        
    Returns:
        StyleProfile
    
    Example:
        profile = extract_style_profile("photo.jpg", height_cm=175)
        print(f"Face shape: {profile.face_metrics['face_shape']}")
        print(f"Skin tone: {profile.color_profile.skin_tone}")
    """
    config = PipelineConfig(
        enable_clothing_detection=enable_clothing,
        height_cm=height_cm,
        weight_kg=weight_kg,
    )
    
    with StyleProfilePipeline(config) as pipeline:
        result = pipeline.analyze(image)
        return result.profile
