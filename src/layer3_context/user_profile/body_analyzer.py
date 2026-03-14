"""
Body Analyzer
=============

Detects body landmarks, computes body metrics, and classifies body shape.
Uses MediaPipe Pose for landmark detection.
"""

import logging
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import numpy as np
from PIL import Image

from .models import (
    BodyKeypoints,
    BodyMetrics,
    BodyShape,
)

logger = logging.getLogger(__name__)


class BodyAnalyzer:
    """
    Analyzes body proportions and classifies body shape.
    
    Uses MediaPipe Pose for landmark detection and computes
    styling-relevant body metrics.
    """
    
    # MediaPipe landmark indices
    LANDMARK_INDICES = {
        "nose": 0,
        "left_eye_inner": 1,
        "left_eye": 2,
        "left_eye_outer": 3,
        "right_eye_inner": 4,
        "right_eye": 5,
        "right_eye_outer": 6,
        "left_ear": 7,
        "right_ear": 8,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }
    
    # Body shape thresholds
    SHAPE_THRESHOLDS = {
        "hourglass_shoulder_hip_diff": 0.05,  # Max diff for balanced
        "hourglass_waist_ratio": 0.75,        # Waist significantly smaller
        "inverted_triangle_ratio": 1.10,      # Shoulders 10%+ wider
        "triangle_ratio": 0.90,               # Hips 10%+ wider
        "rectangle_waist_diff": 0.15,         # Minimal waist definition
    }
    
    # Path to model file (downloaded on first use)
    MODEL_PATH = None
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize body analyzer.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
        """
        self.min_detection_confidence = min_detection_confidence
        self._pose_landmarker = None
        self._mp = None
    
    def _ensure_mediapipe(self):
        """Lazy load MediaPipe using new Tasks API."""
        if self._pose_landmarker is None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                self._mp = mp
                
                # Check if we need to download the model
                model_path = self._get_model_path()
                
                # Create options
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    output_segmentation_masks=False,
                    min_pose_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_detection_confidence,
                )
                
                # Create landmarker
                self._pose_landmarker = vision.PoseLandmarker.create_from_options(options)
                logger.info("MediaPipe PoseLandmarker initialized")
                
            except ImportError:
                logger.error("MediaPipe not installed. Install with: pip install mediapipe")
                raise ImportError("MediaPipe is required for body analysis")
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe: {e}")
                raise
    
    def _get_model_path(self) -> str:
        """Get path to pose landmarker model, downloading if needed."""
        import os
        import urllib.request
        
        # Check for cached model
        cache_dir = Path.home() / ".cache" / "mediapipe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "pose_landmarker_lite.task"
        
        if not model_path.exists():
            logger.info("Downloading pose landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"Model downloaded to {model_path}")
        
        return str(model_path)
    
    def analyze(
        self,
        image: Image.Image,
        height_cm: Optional[float] = None,
        weight_kg: Optional[float] = None
    ) -> BodyMetrics:
        """
        Analyze body from image.
        
        Args:
            image: PIL Image of person
            height_cm: Optional known height in cm
            weight_kg: Optional known weight in kg
            
        Returns:
            BodyMetrics with measurements and classification
        """
        self._ensure_mediapipe()
        
        # Convert PIL to MediaPipe Image
        img_rgb = np.array(image.convert("RGB"))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=img_rgb)
        
        # Run pose detection with new API
        detection_result = self._pose_landmarker.detect(mp_image)
        
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            logger.warning("No pose detected in image")
            return BodyMetrics(
                height_cm=height_cm,
                weight_kg=weight_kg,
                bmi=self._calculate_bmi(height_cm, weight_kg),
            )
        
        # Get first person's landmarks
        landmarks = detection_result.pose_landmarks[0]
        
        # Extract keypoints
        keypoints = self._extract_keypoints(landmarks, image.size)
        
        # Compute metrics
        metrics = self._compute_metrics(keypoints, height_cm, weight_kg, image.size)
        
        # Classify body shape
        metrics.body_shape = self._classify_body_shape(metrics)
        
        return metrics
    
    def _extract_keypoints(self, landmarks, image_size: Tuple[int, int]) -> BodyKeypoints:
        """Extract keypoints from MediaPipe landmarks (Tasks API format)."""
        width, height = image_size
        
        def get_point(idx: int) -> Optional[Tuple[float, float]]:
            if idx < len(landmarks):
                lm = landmarks[idx]
                # Tasks API uses visibility attribute directly
                visibility = getattr(lm, 'visibility', 1.0)
                if visibility > 0.5:
                    return (lm.x * width, lm.y * height)
            return None
        
        def get_confidence(idx: int) -> float:
            if idx < len(landmarks):
                return getattr(landmarks[idx], 'visibility', 0.0)
            return 0.0
        
        keypoints = BodyKeypoints(
            nose=get_point(self.LANDMARK_INDICES["nose"]),
            left_eye=get_point(self.LANDMARK_INDICES["left_eye"]),
            right_eye=get_point(self.LANDMARK_INDICES["right_eye"]),
            left_ear=get_point(self.LANDMARK_INDICES["left_ear"]),
            right_ear=get_point(self.LANDMARK_INDICES["right_ear"]),
            left_shoulder=get_point(self.LANDMARK_INDICES["left_shoulder"]),
            right_shoulder=get_point(self.LANDMARK_INDICES["right_shoulder"]),
            left_elbow=get_point(self.LANDMARK_INDICES["left_elbow"]),
            right_elbow=get_point(self.LANDMARK_INDICES["right_elbow"]),
            left_wrist=get_point(self.LANDMARK_INDICES["left_wrist"]),
            right_wrist=get_point(self.LANDMARK_INDICES["right_wrist"]),
            left_hip=get_point(self.LANDMARK_INDICES["left_hip"]),
            right_hip=get_point(self.LANDMARK_INDICES["right_hip"]),
            left_knee=get_point(self.LANDMARK_INDICES["left_knee"]),
            right_knee=get_point(self.LANDMARK_INDICES["right_knee"]),
            left_ankle=get_point(self.LANDMARK_INDICES["left_ankle"]),
            right_ankle=get_point(self.LANDMARK_INDICES["right_ankle"]),
            confidence={
                name: get_confidence(idx) 
                for name, idx in self.LANDMARK_INDICES.items()
            }
        )
        
        return keypoints
    
    def _compute_metrics(
        self,
        keypoints: BodyKeypoints,
        height_cm: Optional[float],
        weight_kg: Optional[float],
        image_size: Tuple[int, int]
    ) -> BodyMetrics:
        """Compute body metrics from keypoints."""
        
        # Get raw pixel measurements
        shoulder_width_px = keypoints.get_shoulder_width()
        hip_width_px = keypoints.get_hip_width()
        torso_length_px = keypoints.get_torso_length()
        leg_length_px = keypoints.get_leg_length()
        
        # Compute ratios
        shoulder_hip_ratio = None
        if shoulder_width_px and hip_width_px and hip_width_px > 0:
            shoulder_hip_ratio = shoulder_width_px / hip_width_px
        
        leg_torso_ratio = None
        if leg_length_px and torso_length_px and torso_length_px > 0:
            leg_torso_ratio = leg_length_px / torso_length_px
        
        # Calculate BMI if possible
        bmi = self._calculate_bmi(height_cm, weight_kg)
        bmi_category = self._get_bmi_category(bmi) if bmi else None
        
        # Estimate real measurements if height provided
        estimated_shoulder_cm = None
        estimated_hip_cm = None
        estimated_torso_cm = None
        estimated_leg_cm = None
        estimated_arm_length_cm = None
        estimated_inseam_cm = None
        
        if height_cm and keypoints.left_ankle and keypoints.nose:
            # Estimate pixel-to-cm ratio from visible body height
            visible_height_px = abs(keypoints.left_ankle[1] - keypoints.nose[1])
            if visible_height_px > 0:
                # Assume visible height is ~85% of total height (head to ankle)
                px_per_cm = visible_height_px / (height_cm * 0.85)
                
                if shoulder_width_px:
                    estimated_shoulder_cm = shoulder_width_px / px_per_cm
                if hip_width_px:
                    estimated_hip_cm = hip_width_px / px_per_cm
                if torso_length_px:
                    estimated_torso_cm = torso_length_px / px_per_cm
                if leg_length_px:
                    estimated_leg_cm = leg_length_px / px_per_cm
                
                # Estimate arm length (~45% of height typically)
                estimated_arm_length_cm = height_cm * 0.45
                
                # Estimate inseam (~45-47% of height)
                estimated_inseam_cm = height_cm * 0.46
        
        # Compute proportion analysis if height provided
        torso_proportion = None
        leg_proportion = None
        frame_size = None
        proportion_tips = []
        
        if height_cm:
            # Analyze torso proportion (torso should be ~30-33% of height)
            if estimated_torso_cm:
                torso_pct = (estimated_torso_cm / height_cm) * 100
                if torso_pct < 28:
                    torso_proportion = "short"
                    proportion_tips.append("Shorter torso - high-waisted bottoms elongate the torso")
                elif torso_pct > 35:
                    torso_proportion = "long"
                    proportion_tips.append("Longer torso - crop tops and tucked shirts balance proportions")
                else:
                    torso_proportion = "average"
            
            # Analyze leg proportion (legs should be ~45-50% of height)
            if estimated_leg_cm:
                leg_pct = (estimated_leg_cm / height_cm) * 100
                if leg_pct < 43:
                    leg_proportion = "short"
                    proportion_tips.append("Shorter legs - high-waisted styles and vertical lines elongate legs")
                elif leg_pct > 52:
                    leg_proportion = "long"
                    proportion_tips.append("Longer legs - low-rise and color blocking work well")
                else:
                    leg_proportion = "average"
            
            # Estimate frame size based on shoulder width relative to height
            if estimated_shoulder_cm:
                shoulder_to_height = estimated_shoulder_cm / height_cm
                if shoulder_to_height < 0.22:
                    frame_size = "small"
                elif shoulder_to_height > 0.26:
                    frame_size = "large"
                else:
                    frame_size = "medium"
        
        # Estimate clothing sizes
        estimated_top_size = None
        estimated_bottom_size = None
        estimated_dress_size = None
        
        if estimated_shoulder_cm:
            estimated_top_size = self._estimate_top_size(estimated_shoulder_cm, bmi)
        if estimated_hip_cm:
            estimated_bottom_size = self._estimate_bottom_size(estimated_hip_cm)
        if height_cm and estimated_hip_cm:
            estimated_dress_size = self._estimate_dress_size(estimated_hip_cm, height_cm)
        
        return BodyMetrics(
            shoulder_width_px=shoulder_width_px,
            hip_width_px=hip_width_px,
            torso_length_px=torso_length_px,
            leg_length_px=leg_length_px,
            shoulder_hip_ratio=shoulder_hip_ratio,
            leg_torso_ratio=leg_torso_ratio,
            height_cm=height_cm,
            weight_kg=weight_kg,
            estimated_shoulder_cm=estimated_shoulder_cm,
            estimated_hip_cm=estimated_hip_cm,
            estimated_torso_cm=estimated_torso_cm,
            estimated_leg_cm=estimated_leg_cm,
            estimated_arm_length_cm=estimated_arm_length_cm,
            estimated_inseam_cm=estimated_inseam_cm,
            torso_proportion=torso_proportion,
            leg_proportion=leg_proportion,
            frame_size=frame_size,
            estimated_top_size=estimated_top_size,
            estimated_bottom_size=estimated_bottom_size,
            estimated_dress_size=estimated_dress_size,
            bmi=bmi,
            bmi_category=bmi_category,
            keypoints=keypoints,
            proportion_tips=proportion_tips,
        )
    
    def _estimate_top_size(
        self,
        shoulder_cm: float,
        bmi: Optional[float] = None
    ) -> str:
        """Estimate top size from shoulder width."""
        # Shoulder width to size mapping (approximate)
        # These are rough estimates and vary by brand
        if shoulder_cm < 38:
            base_size = "XS"
        elif shoulder_cm < 41:
            base_size = "S"
        elif shoulder_cm < 44:
            base_size = "M"
        elif shoulder_cm < 47:
            base_size = "L"
        elif shoulder_cm < 50:
            base_size = "XL"
        else:
            base_size = "XXL"
        
        # Adjust for BMI if available
        if bmi and bmi > 28:
            sizes = ["XS", "S", "M", "L", "XL", "XXL", "3XL"]
            idx = sizes.index(base_size) if base_size in sizes else 2
            base_size = sizes[min(idx + 1, len(sizes) - 1)]
        
        return base_size
    
    def _estimate_bottom_size(self, hip_cm: float) -> str:
        """Estimate bottom size from hip width."""
        # Hip width to size mapping (approximate)
        if hip_cm < 88:
            return "XS"
        elif hip_cm < 93:
            return "S"
        elif hip_cm < 98:
            return "M"
        elif hip_cm < 103:
            return "L"
        elif hip_cm < 108:
            return "XL"
        else:
            return "XXL"
    
    def _estimate_dress_size(self, hip_cm: float, height_cm: float) -> str:
        """Estimate EU dress size from measurements."""
        # Approximate EU sizing based on hip circumference
        # (Note: hip_cm here is width, so we estimate circumference)
        hip_circumference = hip_cm * 2.5  # Rough estimate
        
        if hip_circumference < 88:
            return "EU 34"
        elif hip_circumference < 92:
            return "EU 36"
        elif hip_circumference < 96:
            return "EU 38"
        elif hip_circumference < 100:
            return "EU 40"
        elif hip_circumference < 104:
            return "EU 42"
        elif hip_circumference < 108:
            return "EU 44"
        elif hip_circumference < 112:
            return "EU 46"
        else:
            return "EU 48+"
    
    def _classify_body_shape(self, metrics: BodyMetrics) -> Optional[BodyShape]:
        """
        Classify body shape based on proportions.
        
        Classification logic:
        - HOURGLASS: Balanced shoulders/hips, defined waist
        - INVERTED_TRIANGLE: Shoulders notably wider than hips
        - TRIANGLE (Pear): Hips notably wider than shoulders  
        - RECTANGLE: Similar measurements throughout
        - OVAL (Apple): Based on BMI + waist emphasis
        - ATHLETIC: Based on proportions + muscle indicators
        """
        ratio = metrics.shoulder_hip_ratio
        
        if ratio is None:
            logger.warning("Cannot classify body shape: missing shoulder/hip ratio")
            return None
        
        # Use BMI as additional signal if available
        is_higher_bmi = metrics.bmi and metrics.bmi > 28
        
        # Inverted triangle: shoulders significantly wider
        if ratio > self.SHAPE_THRESHOLDS["inverted_triangle_ratio"]:
            return BodyShape.INVERTED_TRIANGLE
        
        # Triangle (pear): hips significantly wider
        if ratio < self.SHAPE_THRESHOLDS["triangle_ratio"]:
            return BodyShape.TRIANGLE
        
        # Oval (apple): higher BMI with relatively balanced proportions
        if is_higher_bmi and 0.95 <= ratio <= 1.05:
            return BodyShape.OVAL
        
        # Balanced proportions - could be hourglass or rectangle
        # Without waist measurement, we estimate based on other factors
        if 0.95 <= ratio <= 1.05:
            # Default to rectangle for balanced without waist data
            # Hourglass requires visible waist definition
            return BodyShape.RECTANGLE
        
        # Athletic: slightly broader shoulders, good proportions
        if 1.0 <= ratio <= 1.10 and metrics.leg_torso_ratio and metrics.leg_torso_ratio > 1.0:
            return BodyShape.ATHLETIC
        
        # Fallback to rectangle
        return BodyShape.RECTANGLE
    
    def _calculate_bmi(
        self,
        height_cm: Optional[float],
        weight_kg: Optional[float]
    ) -> Optional[float]:
        """Calculate BMI from height and weight."""
        if height_cm and weight_kg and height_cm > 0:
            height_m = height_cm / 100
            return weight_kg / (height_m ** 2)
        return None
    
    def _get_bmi_category(self, bmi: float) -> str:
        """Get BMI category."""
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25:
            return "normal"
        elif bmi < 30:
            return "overweight"
        else:
            return "obese"
    
    def close(self):
        """Release resources."""
        if self._pose_landmarker is not None:
            try:
                self._pose_landmarker.close()
            except Exception:
                pass
            self._pose_landmarker = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
