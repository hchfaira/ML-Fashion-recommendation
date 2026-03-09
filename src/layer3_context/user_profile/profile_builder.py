"""
Profile Builder
================

Combines all analysis results into a unified StyleProfile.
Handles missing data and computes derived attributes.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from .models import (
    StyleProfile,
    BodyMetrics,
    FacialLandmarks,
    FaceShape,
    SkinAnalysis,
    HairAnalysis,
    ContrastLevel,
    VisualWeight,
    DetectedClothing,
    SkinTone,
    HairColor,
    BodyShape,
)

logger = logging.getLogger(__name__)


class ProfileBuilder:
    """
    Builds a complete StyleProfile from individual analysis results.
    
    Handles:
    - Missing data with sensible defaults
    - Derived attribute computation
    - Confidence aggregation
    - Validation
    """
    
    # Visual weight estimation factors
    VISUAL_WEIGHT_FACTORS = {
        "contrast_high": 1.5,
        "contrast_low": 0.7,
        "dark_hair": 1.2,
        "light_hair": 0.8,
        "dark_skin": 1.1,
        "light_skin": 0.9,
    }
    
    def __init__(self):
        """Initialize profile builder."""
        self._analysis_results = {}
    
    def set_body_metrics(self, metrics: Optional[BodyMetrics]) -> "ProfileBuilder":
        """Set body metrics analysis result."""
        self._analysis_results["body"] = metrics
        return self
    
    def set_face_landmarks(self, landmarks: Optional[FacialLandmarks]) -> "ProfileBuilder":
        """Set facial landmarks analysis result."""
        self._analysis_results["face_landmarks"] = landmarks
        return self
    
    def set_face_shape(self, shape: Optional[FaceShape]) -> "ProfileBuilder":
        """Set face shape classification result."""
        self._analysis_results["face_shape"] = shape
        return self
    
    def set_skin_analysis(self, analysis: Optional[SkinAnalysis]) -> "ProfileBuilder":
        """Set skin analysis result."""
        self._analysis_results["skin"] = analysis
        return self
    
    def set_hair_analysis(self, analysis: Optional[HairAnalysis]) -> "ProfileBuilder":
        """Set hair analysis result."""
        self._analysis_results["hair"] = analysis
        return self
    
    def set_contrast_level(self, level: Optional[ContrastLevel]) -> "ProfileBuilder":
        """Set contrast level analysis result."""
        self._analysis_results["contrast"] = level
        return self
    
    def set_detected_clothing(self, clothing: Optional[DetectedClothing]) -> "ProfileBuilder":
        """Set detected clothing result."""
        self._analysis_results["clothing"] = clothing
        return self
    
    def build(self) -> StyleProfile:
        """
        Build the final StyleProfile from all analysis results.
        
        Returns:
            Complete StyleProfile
        """
        # Extract results with defaults
        body_metrics = self._analysis_results.get("body")
        face_landmarks = self._analysis_results.get("face_landmarks")
        face_shape = self._analysis_results.get("face_shape")
        skin_analysis = self._analysis_results.get("skin")
        hair_analysis = self._analysis_results.get("hair")
        contrast_level = self._analysis_results.get("contrast")
        detected_clothing = self._analysis_results.get("clothing")
        
        # Compute visual weight
        visual_weight = self._compute_visual_weight(
            skin_analysis=skin_analysis,
            hair_analysis=hair_analysis,
            contrast_level=contrast_level,
        )
        
        # Compute overall confidence
        confidence = self._compute_confidence()
        
        # Build the profile using the existing StyleProfile model
        profile = StyleProfile(
            created_at=datetime.utcnow(),
        )
        
        # Body analysis
        if body_metrics:
            profile.body_metrics = body_metrics
            profile.body_shape = body_metrics.body_shape
            profile.bmi = body_metrics.bmi
            profile.shoulder_hip_ratio = body_metrics.shoulder_hip_ratio
            profile.leg_torso_ratio = body_metrics.leg_torso_ratio
        
        # Face analysis
        profile.facial_landmarks = face_landmarks
        profile.face_shape = face_shape
        
        # Skin analysis
        if skin_analysis:
            profile.skin_analysis = skin_analysis
            profile.skin_tone = skin_analysis.skin_tone
            profile.undertone = skin_analysis.undertone
        
        # Hair analysis
        if hair_analysis:
            profile.hair_analysis = hair_analysis
            profile.hair_color = hair_analysis.hair_color
        
        # Contrast
        profile.contrast_level = contrast_level
        profile.visual_weight = visual_weight
        
        # Clothing
        profile.detected_clothing = detected_clothing
        
        # Metadata
        profile.overall_confidence = confidence
        
        return profile
    
    def _compute_visual_weight(
        self,
        skin_analysis: Optional[SkinAnalysis],
        hair_analysis: Optional[HairAnalysis],
        contrast_level: Optional[ContrastLevel],
    ) -> VisualWeight:
        """
        Compute visual weight based on coloring.
        
        Visual weight affects how "heavy" a person appears visually,
        which influences clothing recommendations (e.g., balancing
        visual weight with fabric weight and pattern scale).
        """
        score = 1.0  # Neutral baseline
        
        # Contrast contribution
        if contrast_level:
            if contrast_level in [ContrastLevel.HIGH, ContrastLevel.VERY_HIGH]:
                score *= self.VISUAL_WEIGHT_FACTORS["contrast_high"]
            elif contrast_level in [ContrastLevel.LOW, ContrastLevel.VERY_LOW]:
                score *= self.VISUAL_WEIGHT_FACTORS["contrast_low"]
        
        # Hair color contribution
        if hair_analysis and hair_analysis.hair_color:
            if hair_analysis.hair_color in [HairColor.BLACK, HairColor.DARK_BROWN]:
                score *= self.VISUAL_WEIGHT_FACTORS["dark_hair"]
            elif hair_analysis.hair_color in [HairColor.BLONDE, HairColor.WHITE, HairColor.PLATINUM]:
                score *= self.VISUAL_WEIGHT_FACTORS["light_hair"]
        
        # Skin tone contribution
        if skin_analysis and skin_analysis.skin_tone:
            if skin_analysis.skin_tone in [SkinTone.DARK, SkinTone.VERY_DARK]:
                score *= self.VISUAL_WEIGHT_FACTORS["dark_skin"]
            elif skin_analysis.skin_tone in [SkinTone.LIGHT, SkinTone.VERY_LIGHT]:
                score *= self.VISUAL_WEIGHT_FACTORS["light_skin"]
        
        # Classify
        if score >= 1.5:
            return VisualWeight.HEAVY
        elif score >= 1.2:
            return VisualWeight.MEDIUM_HEAVY
        elif score >= 0.9:
            return VisualWeight.MEDIUM
        elif score >= 0.7:
            return VisualWeight.MEDIUM_LIGHT
        else:
            return VisualWeight.LIGHT
    
    def _compute_confidence(self) -> float:
        """Compute overall confidence from individual results."""
        confidences = []
        
        # Body metrics confidence
        body = self._analysis_results.get("body")
        if body and hasattr(body, 'body_shape_confidence'):
            confidences.append(body.body_shape_confidence)
        
        # Face landmarks confidence
        face = self._analysis_results.get("face_landmarks")
        if face and hasattr(face, 'confidence'):
            confidences.append(face.confidence)
        
        # Skin analysis confidence
        skin = self._analysis_results.get("skin")
        if skin and hasattr(skin, 'skin_tone_confidence'):
            confidences.append(skin.skin_tone_confidence)
        
        # Hair analysis confidence
        hair = self._analysis_results.get("hair")
        if hair and hasattr(hair, 'hair_color_confidence'):
            confidences.append(hair.hair_color_confidence)
        
        # Detected clothing confidence
        clothing = self._analysis_results.get("clothing")
        if clothing and hasattr(clothing, 'confidence'):
            confidences.append(clothing.confidence)
        
        if not confidences:
            return 0.5  # Default medium confidence
        
        return sum(confidences) / len(confidences)
    
    def reset(self) -> "ProfileBuilder":
        """Reset builder for reuse."""
        self._analysis_results = {}
        return self
    
    def from_dict(self, data: Dict[str, Any]) -> "ProfileBuilder":
        """
        Set analysis results from a dictionary.
        
        Useful for reconstructing from saved data.
        """
        if "body" in data and data["body"]:
            self._analysis_results["body"] = BodyMetrics(**data["body"])
        
        if "face_landmarks" in data and data["face_landmarks"]:
            self._analysis_results["face_landmarks"] = FacialLandmarks(**data["face_landmarks"])
        
        if "face_shape" in data and data["face_shape"]:
            self._analysis_results["face_shape"] = FaceShape(data["face_shape"])
        
        if "skin" in data and data["skin"]:
            self._analysis_results["skin"] = SkinAnalysis(**data["skin"])
        
        if "hair" in data and data["hair"]:
            self._analysis_results["hair"] = HairAnalysis(**data["hair"])
        
        if "contrast" in data and data["contrast"]:
            self._analysis_results["contrast"] = ContrastLevel(data["contrast"])
        
        if "clothing" in data and data["clothing"]:
            self._analysis_results["clothing"] = DetectedClothing(**data["clothing"])
        
        return self
