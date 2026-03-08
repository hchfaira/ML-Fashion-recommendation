"""
Data Models for Layer 0 Pipeline
=================================

Defines the data structures used throughout the garment extraction pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from enum import Enum
import numpy as np
from PIL import Image

from .taxonomy import GarmentCategory


# ============================================================================
# Detection Models
# ============================================================================

@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates and metadata."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, width, height) tuple."""
        return (self.x1, self.y1, self.width, self.height)
    
    @classmethod
    def from_tuple(cls, box: Tuple[int, int, int, int]) -> "BoundingBox":
        """Create from (x1, y1, x2, y2) tuple."""
        return cls(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
    
    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this box intersects with another."""
        return not (
            self.x2 < other.x1 or
            self.x1 > other.x2 or
            self.y2 < other.y1 or
            self.y1 > other.y2
        )
    
    def intersection_over_union(self, other: "BoundingBox") -> float:
        """Calculate IoU with another box."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class Detection:
    """Single detection from GroundingDINO."""
    label: str
    box: BoundingBox
    score: float
    category: GarmentCategory = field(default=GarmentCategory.UNKNOWN)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "class": self.label,
            "box": self.box.to_tuple(),
            "score": self.score,
            "category": self.category.value
        }


@dataclass
class DetectionResult:
    """Collection of detections from GroundingDINO."""
    detections: List[Detection] = field(default_factory=list)
    image_size: Tuple[int, int] = (0, 0)  # (height, width)
    
    def __len__(self) -> int:
        return len(self.detections)
    
    def __iter__(self):
        return iter(self.detections)
    
    def __getitem__(self, idx):
        return self.detections[idx]
    
    def filter_by_score(self, min_score: float) -> "DetectionResult":
        """Filter detections by minimum score."""
        filtered = [d for d in self.detections if d.score >= min_score]
        return DetectionResult(detections=filtered, image_size=self.image_size)
    
    def filter_by_category(self, category: GarmentCategory) -> "DetectionResult":
        """Filter detections by category."""
        filtered = [d for d in self.detections if d.category == category]
        return DetectionResult(detections=filtered, image_size=self.image_size)
    
    def get_best_detection(self) -> Optional[Detection]:
        """Get the highest-scoring detection."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.score)
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert all detections to list of dictionaries."""
        return [d.to_dict() for d in self.detections]


# ============================================================================
# Parsing Models
# ============================================================================

@dataclass
class ParsingResult:
    """Result from SCHP human parsing."""
    parsing_map: np.ndarray  # (H, W) with label IDs
    label_areas: Dict[int, int] = field(default_factory=dict)  # label_id -> pixel count
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.parsing_map.shape
    
    def get_mask_for_label(self, label_id: int) -> np.ndarray:
        """Get binary mask for a specific label."""
        return (self.parsing_map == label_id).astype(np.uint8) * 255
    
    def get_mask_for_category(self, category: GarmentCategory) -> np.ndarray:
        """Get binary mask for all labels in a category."""
        from .taxonomy import SCHP_TO_TAXONOMY
        
        mask = np.zeros(self.parsing_map.shape, dtype=np.uint8)
        for label_id, cat in SCHP_TO_TAXONOMY.items():
            if cat == category:
                mask = np.maximum(mask, (self.parsing_map == label_id).astype(np.uint8) * 255)
        return mask
    
    def get_present_labels(self) -> List[int]:
        """Get list of label IDs present in the parsing map."""
        return list(np.unique(self.parsing_map))
    
    def get_present_categories(self) -> List[GarmentCategory]:
        """Get list of garment categories present in the parsing map."""
        from .taxonomy import SCHP_TO_TAXONOMY
        
        present = set()
        for label_id in self.get_present_labels():
            if label_id in SCHP_TO_TAXONOMY:
                present.add(SCHP_TO_TAXONOMY[label_id])
        return list(present)


# ============================================================================
# Mask Models
# ============================================================================

@dataclass
class RefinedMask:
    """Result from SAM mask refinement."""
    mask: np.ndarray  # (H, W) binary mask
    score: float = 1.0
    box: Optional[BoundingBox] = None
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.mask.shape
    
    @property
    def area(self) -> int:
        return int(np.sum(self.mask > 0))
    
    def to_binary(self) -> np.ndarray:
        """Ensure mask is binary (0 or 255)."""
        return ((self.mask > 127).astype(np.uint8)) * 255


@dataclass
class FusedMask:
    """Result from mask fusion."""
    mask: np.ndarray  # (H, W) binary mask
    detection: Optional[Detection] = None
    category: GarmentCategory = field(default=GarmentCategory.UNKNOWN)
    label: str = ""
    confidence: float = 1.0
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.mask.shape
    
    @property
    def area(self) -> int:
        return int(np.sum(self.mask > 0))


# ============================================================================
# Garment Output Models
# ============================================================================

@dataclass
class ExtractedGarment:
    """
    Final extracted garment with RGBA image and metadata.
    This is the primary output of the pipeline.
    """
    image: Image.Image  # RGBA image with transparent background
    category: GarmentCategory
    label: str  # Specific garment type (e.g., "t-shirt", "jeans")
    confidence: float
    
    # Source information
    source_path: Optional[Path] = None
    bounding_box: Optional[BoundingBox] = None
    
    # Mask information
    mask: Optional[np.ndarray] = None
    area: int = 0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, output_path: Path) -> Path:
        """Save garment image to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(output_path, "PNG")
        return output_path
    
    def get_filename(self, index: int = 1) -> str:
        """
        Generate filename based on category and label.
        Format: {category}_{index:03d}_{label}.png
        """
        safe_label = self.label.replace(" ", "_").replace("-", "_")
        return f"{self.category.value}_{index:03d}_{safe_label}.png"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding image and mask)."""
        return {
            "category": self.category.value,
            "label": self.label,
            "confidence": self.confidence,
            "source_path": str(self.source_path) if self.source_path else None,
            "bounding_box": self.bounding_box.to_tuple() if self.bounding_box else None,
            "area": self.area,
            "metadata": self.metadata
        }


@dataclass
class ExtractionResult:
    """
    Complete result from the extraction pipeline.
    Contains all extracted garments from a single image.
    """
    garments: List[ExtractedGarment] = field(default_factory=list)
    source_image_path: Optional[Path] = None
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.garments)
    
    def __iter__(self):
        return iter(self.garments)
    
    def __getitem__(self, idx):
        return self.garments[idx]
    
    def filter_by_category(self, category: GarmentCategory) -> List[ExtractedGarment]:
        """Get garments of a specific category."""
        return [g for g in self.garments if g.category == category]
    
    def get_best_per_category(self) -> Dict[GarmentCategory, ExtractedGarment]:
        """Get the highest-confidence garment for each category."""
        best = {}
        for garment in self.garments:
            if garment.category not in best or garment.confidence > best[garment.category].confidence:
                best[garment.category] = garment
        return best
    
    def save_all(self, output_dir: Path) -> List[Path]:
        """
        Save all garments to directory with proper naming.
        
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track indices per category
        category_counts: Dict[GarmentCategory, int] = {}
        saved_paths = []
        
        for garment in self.garments:
            cat = garment.category
            idx = category_counts.get(cat, 0) + 1
            category_counts[cat] = idx
            
            filename = garment.get_filename(index=idx)
            output_path = output_dir / filename
            garment.save(output_path)
            saved_paths.append(output_path)
        
        return saved_paths
    
    def to_summary(self) -> Dict[str, Any]:
        """Get summary of extraction results."""
        category_counts = {}
        for garment in self.garments:
            cat = garment.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "total_garments": len(self.garments),
            "categories": category_counts,
            "source_image": str(self.source_image_path) if self.source_image_path else None,
            "processing_time_ms": self.processing_time_ms,
            "errors": self.errors
        }
