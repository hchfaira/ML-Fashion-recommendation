"""
User Style Profile - Data Models
=================================

Data classes and enums for user style profile extraction.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import json


# =============================================================================
# Enums
# =============================================================================

class BodyShape(str, Enum):
    """Body shape classifications based on shoulder/hip/waist proportions."""
    RECTANGLE = "rectangle"       # Shoulders ≈ hips, minimal waist definition
    TRIANGLE = "triangle"         # Hips wider than shoulders (pear)
    INVERTED_TRIANGLE = "inverted_triangle"  # Shoulders wider than hips
    HOURGLASS = "hourglass"       # Balanced shoulders/hips, defined waist
    OVAL = "oval"                 # Rounded midsection (apple)
    ATHLETIC = "athletic"         # Muscular, balanced proportions


class FaceShape(str, Enum):
    """Face shape classifications based on facial proportions."""
    OVAL = "oval"           # Balanced, slightly longer than wide
    ROUND = "round"         # Width ≈ height, soft features
    SQUARE = "square"       # Strong jawline, equal width/height
    HEART = "heart"         # Wide forehead, narrow chin
    DIAMOND = "diamond"     # Narrow forehead/chin, wide cheekbones
    RECTANGLE = "rectangle" # Longer than wide, angular
    OBLONG = "oblong"       # Very long, narrow


class SkinTone(str, Enum):
    """Skin tone depth categories."""
    VERY_LIGHT = "very_light"       # Porcelain, ivory
    LIGHT = "light"                 # Fair, light beige
    MEDIUM_LIGHT = "medium_light"   # Light-medium
    MEDIUM = "medium"               # Medium beige, olive
    MEDIUM_DARK = "medium_dark"     # Tan, caramel
    OLIVE = "olive"                 # Olive, tan
    DARK = "dark"                   # Dark brown
    VERY_DARK = "very_dark"         # Very deep
    DEEP = "deep"                   # Very deep, ebony


class Undertone(str, Enum):
    """Skin undertone classifications."""
    COOL = "cool"       # Pink, red, blue undertones
    WARM = "warm"       # Yellow, golden, peachy undertones
    NEUTRAL = "neutral" # Mix of warm and cool


class HairColor(str, Enum):
    """Hair color classifications."""
    BLACK = "black"
    DARK_BROWN = "dark_brown"
    MEDIUM_BROWN = "medium_brown"
    LIGHT_BROWN = "light_brown"
    DARK_BLONDE = "dark_blonde"
    LIGHT_BLONDE = "light_blonde"
    BLONDE = "blonde"           # General blonde
    PLATINUM = "platinum"       # Very light blonde
    RED = "red"
    AUBURN = "auburn"
    GRAY = "gray"
    GREY = "grey"               # Alternate spelling
    WHITE = "white"
    OTHER = "other"  # Dyed colors, etc.


class ContrastLevel(str, Enum):
    """Contrast between features (hair/skin/eyes)."""
    VERY_LOW = "very_low"   # Very similar tones
    LOW = "low"             # Similar tones throughout
    MEDIUM = "medium"       # Moderate contrast
    HIGH = "high"           # Strong contrast (dark hair, light skin)
    VERY_HIGH = "very_high" # Very strong contrast


class VisualWeight(str, Enum):
    """Visual weight/presence of facial features."""
    LIGHT = "light"               # Delicate features, soft
    MEDIUM_LIGHT = "medium_light" # Between light and medium
    MEDIUM = "medium"             # Balanced features
    MEDIUM_HEAVY = "medium_heavy" # Between medium and heavy
    HEAVY = "heavy"               # Bold features, prominent
    STRONG = "strong"             # Bold features, prominent (alias)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BodyKeypoints:
    """Body landmark keypoints from pose detection."""
    # Head
    nose: Optional[Tuple[float, float]] = None
    left_eye: Optional[Tuple[float, float]] = None
    right_eye: Optional[Tuple[float, float]] = None
    left_ear: Optional[Tuple[float, float]] = None
    right_ear: Optional[Tuple[float, float]] = None
    
    # Upper body
    left_shoulder: Optional[Tuple[float, float]] = None
    right_shoulder: Optional[Tuple[float, float]] = None
    left_elbow: Optional[Tuple[float, float]] = None
    right_elbow: Optional[Tuple[float, float]] = None
    left_wrist: Optional[Tuple[float, float]] = None
    right_wrist: Optional[Tuple[float, float]] = None
    
    # Torso
    left_hip: Optional[Tuple[float, float]] = None
    right_hip: Optional[Tuple[float, float]] = None
    
    # Lower body
    left_knee: Optional[Tuple[float, float]] = None
    right_knee: Optional[Tuple[float, float]] = None
    left_ankle: Optional[Tuple[float, float]] = None
    right_ankle: Optional[Tuple[float, float]] = None
    
    # Confidence scores
    confidence: Dict[str, float] = field(default_factory=dict)
    
    def get_shoulder_width(self) -> Optional[float]:
        """Calculate shoulder width in pixels."""
        if self.left_shoulder and self.right_shoulder:
            dx = self.right_shoulder[0] - self.left_shoulder[0]
            dy = self.right_shoulder[1] - self.left_shoulder[1]
            return (dx**2 + dy**2) ** 0.5
        return None
    
    def get_hip_width(self) -> Optional[float]:
        """Calculate hip width in pixels."""
        if self.left_hip and self.right_hip:
            dx = self.right_hip[0] - self.left_hip[0]
            dy = self.right_hip[1] - self.left_hip[1]
            return (dx**2 + dy**2) ** 0.5
        return None
    
    def get_torso_length(self) -> Optional[float]:
        """Calculate torso length (shoulder to hip midpoint)."""
        if all([self.left_shoulder, self.right_shoulder, self.left_hip, self.right_hip]):
            shoulder_mid = (
                (self.left_shoulder[0] + self.right_shoulder[0]) / 2,
                (self.left_shoulder[1] + self.right_shoulder[1]) / 2
            )
            hip_mid = (
                (self.left_hip[0] + self.right_hip[0]) / 2,
                (self.left_hip[1] + self.right_hip[1]) / 2
            )
            dx = hip_mid[0] - shoulder_mid[0]
            dy = hip_mid[1] - shoulder_mid[1]
            return (dx**2 + dy**2) ** 0.5
        return None
    
    def get_leg_length(self) -> Optional[float]:
        """Calculate leg length (hip to ankle average)."""
        lengths = []
        
        if self.left_hip and self.left_ankle:
            dx = self.left_ankle[0] - self.left_hip[0]
            dy = self.left_ankle[1] - self.left_hip[1]
            lengths.append((dx**2 + dy**2) ** 0.5)
        
        if self.right_hip and self.right_ankle:
            dx = self.right_ankle[0] - self.right_hip[0]
            dy = self.right_ankle[1] - self.right_hip[1]
            lengths.append((dx**2 + dy**2) ** 0.5)
        
        return sum(lengths) / len(lengths) if lengths else None


@dataclass
class FacialLandmarks:
    """Facial landmark measurements."""
    # Key measurements (in pixels)
    face_height: Optional[float] = None
    face_width: Optional[float] = None
    jaw_width: Optional[float] = None
    cheekbone_width: Optional[float] = None
    forehead_width: Optional[float] = None
    chin_length: Optional[float] = None
    
    # Face bounding box
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    
    # Landmark points (if available)
    landmarks: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Confidence
    confidence: float = 0.0
    
    @property
    def face_ratio(self) -> Optional[float]:
        """Face width to height ratio."""
        if self.face_width and self.face_height and self.face_height > 0:
            return self.face_width / self.face_height
        return None
    
    @property
    def jaw_to_cheekbone_ratio(self) -> Optional[float]:
        """Jaw width to cheekbone width ratio."""
        if self.jaw_width and self.cheekbone_width and self.cheekbone_width > 0:
            return self.jaw_width / self.cheekbone_width
        return None


@dataclass
class BodyMetrics:
    """Computed body metrics and proportions."""
    # Raw measurements (in pixels, relative)
    shoulder_width_px: Optional[float] = None
    hip_width_px: Optional[float] = None
    torso_length_px: Optional[float] = None
    leg_length_px: Optional[float] = None
    
    # Ratios
    shoulder_hip_ratio: Optional[float] = None
    leg_torso_ratio: Optional[float] = None
    waist_hip_ratio: Optional[float] = None  # If waist detectable
    
    # Estimated real measurements (if height provided)
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    estimated_shoulder_cm: Optional[float] = None
    estimated_hip_cm: Optional[float] = None
    estimated_torso_cm: Optional[float] = None
    estimated_leg_cm: Optional[float] = None
    estimated_arm_length_cm: Optional[float] = None
    estimated_inseam_cm: Optional[float] = None
    
    # BMI (if height and weight provided)
    bmi: Optional[float] = None
    bmi_category: Optional[str] = None
    
    # Body proportions analysis (requires height)
    torso_proportion: Optional[str] = None   # "short", "average", "long"
    leg_proportion: Optional[str] = None     # "short", "average", "long"
    frame_size: Optional[str] = None         # "small", "medium", "large"
    
    # Clothing size estimations (based on measurements)
    estimated_top_size: Optional[str] = None     # XS, S, M, L, XL, etc.
    estimated_bottom_size: Optional[str] = None  # XS, S, M, L, XL, etc.
    estimated_dress_size: Optional[str] = None   # EU sizes
    
    # Body shape classification
    body_shape: Optional[BodyShape] = None
    body_shape_confidence: float = 0.0
    
    # Enhanced morphology — 3D estimation
    waist_width_px: Optional[float] = None        # Estimated waist width in pixels
    waist_width_ratio: Optional[float] = None      # waist_width / shoulder_width
    body_shape_secondary: Optional[BodyShape] = None
    body_shape_scores: Dict[str, float] = field(default_factory=dict)  # normalised, sum ≈ 1
    
    # Keypoints
    keypoints: Optional[BodyKeypoints] = None
    
    # Styling recommendations based on proportions
    proportion_tips: List[str] = field(default_factory=list)


@dataclass
class SkinAnalysis:
    """Skin tone and undertone analysis results."""
    # Main classifications
    skin_tone: Optional[SkinTone] = None
    undertone: Optional[Undertone] = None
    
    # Color values
    dominant_skin_rgb: Optional[Tuple[int, int, int]] = None
    dominant_skin_lab: Optional[Tuple[float, float, float]] = None
    
    # Confidence
    skin_tone_confidence: float = 0.0
    undertone_confidence: float = 0.0
    
    # Detected skin region
    skin_mask_coverage: float = 0.0  # Percentage of face that is skin
    
    # 12-season colour analysis
    chroma: Optional[str] = None              # "clear" or "muted"
    season_sub: Optional[str] = None          # e.g. "Light Spring", "Deep Winter", …
    season_confidence: float = 0.0
    
    @property
    def confidence(self) -> float:
        """Average confidence across all measurements."""
        return (self.skin_tone_confidence + self.undertone_confidence) / 2


@dataclass
class HairAnalysis:
    """Hair color analysis results."""
    # Main classification
    hair_color: Optional[HairColor] = None
    
    # Color values
    dominant_hair_rgb: Optional[Tuple[int, int, int]] = None
    dominant_hair_lab: Optional[Tuple[float, float, float]] = None
    
    # Confidence
    hair_color_confidence: float = 0.0
    
    # Hair region info
    hair_mask_coverage: float = 0.0


@dataclass
class ContrastAnalysis:
    """Contrast and visual weight analysis."""
    # Contrast between features
    contrast_level: Optional[ContrastLevel] = None
    hair_skin_contrast: float = 0.0  # 0-1 scale
    
    # Visual weight
    visual_weight: Optional[VisualWeight] = None
    
    # Contributing factors
    jaw_sharpness: float = 0.0  # 0-1, sharper = stronger
    feature_size: float = 0.0   # 0-1, larger = stronger
    
    # Confidence
    contrast_confidence: float = 0.0
    visual_weight_confidence: float = 0.0


@dataclass
class DetectedClothing:
    """Optional detected clothing from the image."""
    top_type: Optional[str] = None
    top_color: Optional[str] = None
    top_pattern: Optional[str] = None
    
    bottom_type: Optional[str] = None
    bottom_color: Optional[str] = None
    bottom_pattern: Optional[str] = None
    
    # Overall silhouette
    silhouette: Optional[str] = None
    
    # Detection confidence
    confidence: float = 0.0


@dataclass
class StyleProfile:
    """
    Complete user style profile.
    
    Contains all extracted styling attributes from user photo analysis.
    Used by the context engine for personalized recommendations.
    """
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    image_source: Optional[str] = None
    
    # Body analysis
    body_metrics: Optional[BodyMetrics] = None
    body_shape: Optional[BodyShape] = None
    bmi: Optional[float] = None
    shoulder_hip_ratio: Optional[float] = None
    leg_torso_ratio: Optional[float] = None
    
    # Face analysis
    face_shape: Optional[FaceShape] = None
    facial_landmarks: Optional[FacialLandmarks] = None
    
    # Color analysis
    skin_analysis: Optional[SkinAnalysis] = None
    skin_tone: Optional[SkinTone] = None
    undertone: Optional[Undertone] = None
    
    # 12-season colour analysis (propagated from SkinAnalysis)
    season_sub: Optional[str] = None
    chroma: Optional[str] = None
    season_confidence: float = 0.0
    
    # Hair analysis
    hair_analysis: Optional[HairAnalysis] = None
    hair_color: Optional[HairColor] = None
    
    # Contrast & visual weight
    contrast_analysis: Optional[ContrastAnalysis] = None
    contrast_level: Optional[ContrastLevel] = None
    visual_weight: Optional[VisualWeight] = None
    
    # Enhanced morphology (propagated from BodyMetrics)
    body_shape_secondary: Optional[BodyShape] = None
    body_shape_scores: Dict[str, float] = field(default_factory=dict)
    waist_hip_ratio: Optional[float] = None
    
    # Optional clothing detection
    detected_clothing: Optional[DetectedClothing] = None
    
    # Overall confidence
    overall_confidence: float = 0.0
    
    # Processing metadata
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "body_type": self.body_shape.value if self.body_shape else None,
            "bmi": round(self.bmi, 1) if self.bmi else None,
            "shoulder_hip_ratio": round(self.shoulder_hip_ratio, 2) if self.shoulder_hip_ratio else None,
            "leg_torso_ratio": round(self.leg_torso_ratio, 2) if self.leg_torso_ratio else None,
            "face_shape": self.face_shape.value if self.face_shape else None,
            "skin_tone": self.skin_tone.value if self.skin_tone else None,
            "undertone": self.undertone.value if self.undertone else None,
            "hair_color": self.hair_color.value if self.hair_color else None,
            "contrast_level": self.contrast_level.value if self.contrast_level else None,
            "visual_weight": self.visual_weight.value if self.visual_weight else None,
            # 12-season colour fields
            "season_sub": self.season_sub,
            "chroma": self.chroma,
            "season_confidence": round(self.season_confidence, 2) if self.season_confidence else None,
            # Enhanced morphology fields
            "body_shape_secondary": self.body_shape_secondary.value if self.body_shape_secondary else None,
            "body_shape_scores": {k: round(v, 3) for k, v in self.body_shape_scores.items()} if self.body_shape_scores else {},
            "waist_hip_ratio": round(self.waist_hip_ratio, 3) if self.waist_hip_ratio else None,
        }
        
        # Add composite skin tone string
        if self.skin_tone and self.undertone:
            result["skin_tone_full"] = f"{self.skin_tone.value} {self.undertone.value}"
        
        # Add metadata
        result["_metadata"] = {
            "created_at": self.created_at.isoformat(),
            "overall_confidence": round(self.overall_confidence, 2),
            "processing_time_ms": round(self.processing_time_ms, 1),
            "warnings": self.warnings,
        }
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StyleProfile":
        """Create StyleProfile from dictionary."""
        profile = cls()
        
        if data.get("body_type"):
            profile.body_shape = BodyShape(data["body_type"])
        profile.bmi = data.get("bmi")
        profile.shoulder_hip_ratio = data.get("shoulder_hip_ratio")
        profile.leg_torso_ratio = data.get("leg_torso_ratio")
        
        if data.get("face_shape"):
            profile.face_shape = FaceShape(data["face_shape"])
        
        if data.get("skin_tone"):
            profile.skin_tone = SkinTone(data["skin_tone"])
        if data.get("undertone"):
            profile.undertone = Undertone(data["undertone"])
        
        if data.get("hair_color"):
            profile.hair_color = HairColor(data["hair_color"])
        
        if data.get("contrast_level"):
            profile.contrast_level = ContrastLevel(data["contrast_level"])
        if data.get("visual_weight"):
            profile.visual_weight = VisualWeight(data["visual_weight"])
        
        # 12-season colour fields
        profile.season_sub = data.get("season_sub")
        profile.chroma = data.get("chroma")
        profile.season_confidence = data.get("season_confidence", 0.0)
        
        # Enhanced morphology fields
        if data.get("body_shape_secondary"):
            profile.body_shape_secondary = BodyShape(data["body_shape_secondary"])
        profile.body_shape_scores = data.get("body_shape_scores", {})
        profile.waist_hip_ratio = data.get("waist_hip_ratio")
        
        if data.get("_metadata"):
            profile.overall_confidence = data["_metadata"].get("overall_confidence", 0.0)
            profile.warnings = data["_metadata"].get("warnings", [])
        
        return profile


@dataclass
class ProfileInput:
    """Input for style profile extraction."""
    image_path: Optional[str] = None
    image_bytes: Optional[bytes] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    
    # Optional hints
    known_body_type: Optional[str] = None
    known_skin_tone: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validate input and return list of errors."""
        errors = []
        
        if not self.image_path and not self.image_bytes:
            errors.append("Either image_path or image_bytes must be provided")
        
        if self.height_cm is not None and (self.height_cm < 100 or self.height_cm > 250):
            errors.append(f"Invalid height_cm: {self.height_cm} (expected 100-250)")
        
        if self.weight_kg is not None and (self.weight_kg < 30 or self.weight_kg > 300):
            errors.append(f"Invalid weight_kg: {self.weight_kg} (expected 30-300)")
        
        return errors
