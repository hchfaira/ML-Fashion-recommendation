"""
User Style Profile Extraction Pipeline
=======================================

Extracts styling-relevant attributes from user photos for the fashion context engine.

Pipeline Stages:
1. Body Analysis - Pose detection, body metrics, body shape classification
2. Face Analysis - Face detection, face shape, facial landmarks
3. Color Analysis - Skin tone, undertone detection
4. Hair Analysis - Hair segmentation, color classification
5. Contrast & Visual Weight - Personal color analysis
6. Clothing Detection - Optional visible clothing analysis
7. Profile Builder - Combines all into structured StyleProfile

Example usage:
    from src.layer3_context.user_profile import StyleProfilePipeline
    
    pipeline = StyleProfilePipeline()
    profile = pipeline.analyze(
        image_path="person.jpg",
        height_cm=178,
        weight_kg=72
    )
    
    print(profile.to_dict())
    # {
    #     "body_type": "rectangle",
    #     "bmi": 22.7,
    #     "shoulder_hip_ratio": 1.05,
    #     "face_shape": "oval",
    #     "skin_tone": "medium",
    #     "undertone": "warm",
    #     "hair_color": "dark_brown",
    #     "contrast_level": "high",
    #     "visual_weight": "strong"
    # }
"""

from .models import (
    StyleProfile,
    BodyMetrics,
    BodyShape,
    FaceShape,
    FacialLandmarks,
    SkinTone,
    Undertone,
    HairColor,
    SkinAnalysis,
    HairAnalysis,
    ContrastAnalysis,
    ContrastLevel,
    VisualWeight,
    DetectedClothing,
    ProfileInput,
)
from .body_analyzer import BodyAnalyzer
from .face_analyzer import FaceAnalyzer
from .color_analyzer import ColorAnalyzer
from .hair_analyzer import HairAnalyzer
from .contrast_analyzer import ContrastAnalyzer
from .clothing_detector import ClothingDetector
from .profile_builder import ProfileBuilder
from .pipeline import StyleProfilePipeline, PipelineConfig, PipelineResult, extract_style_profile

__all__ = [
    # Main Pipeline
    "StyleProfilePipeline",
    "PipelineConfig",
    "PipelineResult",
    "extract_style_profile",
    
    # Data Models
    "StyleProfile",
    "BodyMetrics",
    "BodyShape",
    "FaceShape",
    "FacialLandmarks",
    "SkinTone",
    "Undertone",
    "HairColor",
    "SkinAnalysis",
    "HairAnalysis",
    "ContrastAnalysis",
    "ContrastLevel",
    "VisualWeight",
    "DetectedClothing",
    "ProfileInput",
    
    # Analyzers
    "BodyAnalyzer",
    "FaceAnalyzer",
    "ColorAnalyzer",
    "HairAnalyzer",
    "ContrastAnalyzer",
    "ClothingDetector",
    "ProfileBuilder",
]
