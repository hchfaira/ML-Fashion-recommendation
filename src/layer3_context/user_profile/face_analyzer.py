"""
Face Analyzer
=============

Detects facial landmarks and classifies face shape.
Uses MediaPipe Face Mesh for landmark detection.
"""

import logging
from typing import Optional, Tuple, Dict, List
import numpy as np
from PIL import Image

from .models import FacialLandmarks, FaceShape

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """
    Analyzes facial proportions and classifies face shape.
    
    Uses MediaPipe Face Mesh for precise landmark detection.
    """
    
    # Key landmark indices for face shape analysis
    # MediaPipe Face Mesh has 468 landmarks
    LANDMARK_INDICES = {
        # Face outline (chin to temples)
        "chin_bottom": 152,
        "chin_left": 172,
        "chin_right": 397,
        "jaw_left": 234,
        "jaw_right": 454,
        
        # Cheekbones
        "cheekbone_left": 123,
        "cheekbone_right": 352,
        
        # Forehead (approximated from hairline landmarks)
        "forehead_left": 71,
        "forehead_right": 301,
        "forehead_top": 10,
        
        # Face height reference
        "hairline_center": 10,
        
        # Temples
        "temple_left": 54,
        "temple_right": 284,
    }
    
    # Face shape classification thresholds
    SHAPE_THRESHOLDS = {
        "round_ratio_min": 0.85,      # Width/height ratio for round
        "round_ratio_max": 1.00,
        "square_jaw_ratio": 0.85,     # Jaw close to cheekbone width
        "heart_forehead_ratio": 1.05, # Forehead wider than jaw
        "diamond_cheek_ratio": 1.10,  # Cheekbones widest
        "oblong_ratio_max": 0.70,     # Very narrow
    }
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize face analyzer.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
        """
        self.min_detection_confidence = min_detection_confidence
        self._face_mesh = None
        self._mp_face_mesh = None
    
    def _ensure_mediapipe(self):
        """Lazy load MediaPipe Face Landmarker (Tasks API)."""
        if self._face_mesh is None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                self._mp = mp
                
                # Get model path
                model_path = self._get_model_path()
                
                # Create options
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                    min_face_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_detection_confidence,
                )
                
                self._face_mesh = vision.FaceLandmarker.create_from_options(options)
                logger.info("MediaPipe FaceLandmarker initialized")
            except ImportError:
                logger.error("MediaPipe not installed. Install with: pip install mediapipe")
                raise ImportError("MediaPipe is required for face analysis")
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe: {e}")
                raise
    
    def _get_model_path(self) -> str:
        """Get path to face landmarker model, downloading if needed."""
        import urllib.request
        from pathlib import Path
        
        cache_dir = Path.home() / ".cache" / "mediapipe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "face_landmarker.task"
        
        if not model_path.exists():
            logger.info("Downloading face landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"Model downloaded to {model_path}")
        
        return str(model_path)
    
    def analyze(self, image: Image.Image) -> Optional[FacialLandmarks]:
        """
        Analyze face from image.
        
        Args:
            image: PIL Image containing face
            
        Returns:
            FacialLandmarks with measurements, or None if no face detected
        """
        self._ensure_mediapipe()
        
        # Convert PIL to MediaPipe Image
        img_rgb = np.array(image.convert("RGB"))
        height, width = img_rgb.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=img_rgb)
        
        # Run face landmarker detection
        detection_result = self._face_mesh.detect(mp_image)
        
        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            logger.warning("No face detected in image")
            return None
        
        # Use first detected face
        face_landmarks = detection_result.face_landmarks[0]
        
        # Extract measurements
        landmarks = self._extract_measurements(face_landmarks, (width, height))
        
        return landmarks
    
    def classify_face_shape(self, landmarks: FacialLandmarks) -> Optional[FaceShape]:
        """
        Classify face shape based on facial proportions.
        
        Args:
            landmarks: Extracted facial landmarks
            
        Returns:
            FaceShape classification
        """
        if not landmarks:
            return None
        
        # Get key ratios
        face_ratio = landmarks.face_ratio
        jaw_cheek_ratio = landmarks.jaw_to_cheekbone_ratio
        
        if face_ratio is None:
            logger.warning("Cannot classify face shape: missing face ratio")
            return None
        
        # Estimate forehead-to-jaw ratio
        forehead_jaw_ratio = None
        if landmarks.forehead_width and landmarks.jaw_width and landmarks.jaw_width > 0:
            forehead_jaw_ratio = landmarks.forehead_width / landmarks.jaw_width
        
        # Classification logic
        
        # Oval: balanced proportions, slightly longer than wide
        if 0.70 <= face_ratio <= 0.85 and jaw_cheek_ratio and 0.75 <= jaw_cheek_ratio <= 0.90:
            return FaceShape.OVAL
        
        # Round: width similar to height, soft features
        if face_ratio >= self.SHAPE_THRESHOLDS["round_ratio_min"]:
            return FaceShape.ROUND
        
        # Square: strong jawline, width near height
        if jaw_cheek_ratio and jaw_cheek_ratio >= self.SHAPE_THRESHOLDS["square_jaw_ratio"]:
            if face_ratio >= 0.80:
                return FaceShape.SQUARE
        
        # Heart: wide forehead, narrow chin
        if forehead_jaw_ratio and forehead_jaw_ratio >= self.SHAPE_THRESHOLDS["heart_forehead_ratio"]:
            return FaceShape.HEART
        
        # Diamond: cheekbones widest, narrow forehead and chin
        if jaw_cheek_ratio and jaw_cheek_ratio < 0.75:
            if forehead_jaw_ratio and forehead_jaw_ratio < 1.0:
                return FaceShape.DIAMOND
        
        # Oblong/Rectangle: significantly longer than wide
        if face_ratio <= self.SHAPE_THRESHOLDS["oblong_ratio_max"]:
            return FaceShape.OBLONG
        
        # Rectangle: angular, longer than wide
        if face_ratio <= 0.75 and jaw_cheek_ratio and jaw_cheek_ratio >= 0.85:
            return FaceShape.RECTANGLE
        
        # Default to oval for balanced faces
        return FaceShape.OVAL
    
    def _extract_measurements(
        self,
        face_landmarks,
        image_size: Tuple[int, int]
    ) -> FacialLandmarks:
        """Extract measurements from face mesh landmarks (Tasks API format)."""
        width, height = image_size
        
        def get_point(idx: int) -> Tuple[float, float]:
            # Tasks API returns list of NormalizedLandmark objects
            lm = face_landmarks[idx]
            return (lm.x * width, lm.y * height)
        
        def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
            return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) ** 0.5
        
        # Extract key points
        chin_bottom = get_point(self.LANDMARK_INDICES["chin_bottom"])
        forehead_top = get_point(self.LANDMARK_INDICES["forehead_top"])
        
        jaw_left = get_point(self.LANDMARK_INDICES["jaw_left"])
        jaw_right = get_point(self.LANDMARK_INDICES["jaw_right"])
        
        cheekbone_left = get_point(self.LANDMARK_INDICES["cheekbone_left"])
        cheekbone_right = get_point(self.LANDMARK_INDICES["cheekbone_right"])
        
        forehead_left = get_point(self.LANDMARK_INDICES["forehead_left"])
        forehead_right = get_point(self.LANDMARK_INDICES["forehead_right"])
        
        temple_left = get_point(self.LANDMARK_INDICES["temple_left"])
        temple_right = get_point(self.LANDMARK_INDICES["temple_right"])
        
        # Compute measurements
        face_height = distance(forehead_top, chin_bottom)
        face_width = distance(temple_left, temple_right)
        jaw_width = distance(jaw_left, jaw_right)
        cheekbone_width = distance(cheekbone_left, cheekbone_right)
        forehead_width = distance(forehead_left, forehead_right)
        
        # Estimate chin length (bottom of face to jaw line)
        chin_length = distance(chin_bottom, (
            (jaw_left[0] + jaw_right[0]) / 2,
            (jaw_left[1] + jaw_right[1]) / 2
        ))
        
        # Compute bounding box
        num_landmarks = len(face_landmarks)
        all_x = [face_landmarks[i].x * width for i in range(num_landmarks)]
        all_y = [face_landmarks[i].y * height for i in range(num_landmarks)]
        bbox = (
            int(min(all_x)),
            int(min(all_y)),
            int(max(all_x) - min(all_x)),
            int(max(all_y) - min(all_y))
        )
        
        # Store key landmark points
        landmark_dict = {
            "chin_bottom": chin_bottom,
            "forehead_top": forehead_top,
            "jaw_left": jaw_left,
            "jaw_right": jaw_right,
            "cheekbone_left": cheekbone_left,
            "cheekbone_right": cheekbone_right,
            "forehead_left": forehead_left,
            "forehead_right": forehead_right,
        }
        
        return FacialLandmarks(
            face_height=face_height,
            face_width=face_width,
            jaw_width=jaw_width,
            cheekbone_width=cheekbone_width,
            forehead_width=forehead_width,
            chin_length=chin_length,
            bbox=bbox,
            landmarks=landmark_dict,
            confidence=0.9,  # MediaPipe is generally confident
        )
    
    def close(self):
        """Release resources."""
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass
            self._face_mesh = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
