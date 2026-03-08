"""
Layer 0: Advanced Garment Segmentation
=======================================

This layer handles the extraction and segmentation of garments from images
using a multi-stage pipeline for precise garment extraction:

Pipeline:
1. YOLOv8 -> Real-time garment & accessory detection
2. SegFormer -> Fast semantic clothing segmentation
3. Mask Fusion (Detection + Segmentation)
4. OpenCV -> Mask cleaning & contour smoothing
5. Garment RGBA Extraction (transparent background)

Responsibilities:
- Extract individual garments from images with high precision
- Remove backgrounds with clean edges
- Classify garment regions (top, bottom, dress, etc.)
- Resize and center garments on transparent canvas
- Prepare images for attribute extraction (Layer 1)
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from src.core import get_logger

logger = get_logger(__name__)

# ============================================================================
# Constants
# ============================================================================
DEFAULT_CANVAS_SIZE = 512

# Model configurations
YOLO_MODEL_NAME = "yolov8n.pt"  # YOLOv8 nano for speed, can use yolov8s/m/l/x
YOLO_FASHION_MODEL = "yolov8n-fashion.pt"  # Custom fashion-trained model if available
SEGFORMER_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
SEGFORMER_FASHION_MODEL = "mattmdjaga/segformer_b2_clothes"  # Fashion-specific model

# YOLOv8 garment classes (COCO + custom fashion classes)
YOLO_FASHION_CLASSES = {
    # Standard COCO classes that include garments
    "person": 0,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "backpack": 24,
    # Fashion-specific classes (if using fine-tuned model)
    "shirt": 100,
    "pants": 101,
    "dress": 102,
    "jacket": 103,
    "shoes": 104,
    "hat": 105,
    "bag": 106,
}

# SegFormer clothing labels (from fashion-trained model)
SEGFORMER_LABELS = {
    0: "background",
    1: "hat",
    2: "hair",
    3: "sunglasses",
    4: "upper_clothes",  # top, shirt, blouse
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left_shoe",
    10: "right_shoe",
    11: "face",
    12: "left_leg",
    13: "right_leg",
    14: "left_arm",
    15: "right_arm",
    16: "bag",
    17: "scarf",
}

# Garment-related SegFormer labels
GARMENT_LABELS = {
    "top": [4],               # upper_clothes
    "bottom": [5, 6],         # skirt, pants
    "dress": [7],             # dress
    "shoes": [9, 10],         # left_shoe, right_shoe
    "accessories": [1, 3, 8, 16, 17],  # hat, sunglasses, belt, bag, scarf
}

# Text prompts for YOLO (used with some variants)
GARMENT_PROMPTS = {
    "all": ["shirt", "t-shirt", "blouse", "top", "sweater", "jacket", "coat", 
            "pants", "jeans", "trousers", "shorts", "skirt", "dress", 
            "shoes", "sneakers", "boots", "hat", "bag"],
    "top": ["shirt", "t-shirt", "blouse", "top", "sweater", "hoodie", "polo"],
    "outerwear": ["jacket", "coat", "blazer", "cardigan", "vest"],
    "bottom": ["pants", "jeans", "trousers", "shorts", "skirt", "leggings"],
    "dress": ["dress", "gown", "jumpsuit", "romper"],
    "shoes": ["shoes", "sneakers", "boots", "sandals", "heels", "loafers"],
    "accessories": ["hat", "cap", "bag", "scarf", "belt", "watch", "sunglasses"],
}


class GarmentType(Enum):
    """Types of garments that can be detected."""
    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"
    OUTERWEAR = "outerwear"
    SHOES = "shoes"
    ACCESSORIES = "accessories"
    UNKNOWN = "unknown"


@dataclass
class SegmentedGarment:
    """Represents a segmented garment with its image and metadata."""
    image: Image.Image  # RGBA image with transparent background
    original_path: Optional[Path] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    area: int = 0
    confidence: float = 1.0
    garment_type: GarmentType = GarmentType.UNKNOWN
    label: str = ""
    mask: Optional[np.ndarray] = None  # Original binary mask


@dataclass
class DetectionResult:
    """Result from YOLOv8 detection."""
    boxes: List[Tuple[int, int, int, int]]  # List of (x1, y1, x2, y2)
    labels: List[str]
    scores: List[float]
    class_ids: List[int] = field(default_factory=list)


# ============================================================================
# Advanced Garment Segmenter (YOLOv8 + SegFormer)
# ============================================================================

class AdvancedGarmentSegmenter:
    """
    Advanced garment segmentation using multi-stage pipeline:
    
    1. YOLOv8: Real-time garment & accessory detection
    2. SegFormer: Fast semantic clothing segmentation
    3. Mask Fusion: Combine detection boxes with segmentation masks
    4. OpenCV: Mask cleaning and contour smoothing
    
    This provides fast and accurate segmentation for fashion items.
    """
    
    def __init__(
        self,
        canvas_size: int = DEFAULT_CANVAS_SIZE,
        device: Optional[str] = None,
        use_segformer: bool = True,
        use_yolo: bool = True,
        yolo_model: str = YOLO_MODEL_NAME,
        segformer_model: str = SEGFORMER_FASHION_MODEL,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ):
        """
        Initialize the advanced segmenter.
        
        Args:
            canvas_size: Output canvas size (default 512x512)
            device: Device to run on (cuda/cpu/mps)
            use_segformer: Whether to use SegFormer for semantic segmentation
            use_yolo: Whether to use YOLOv8 for detection
            yolo_model: YOLOv8 model name or path
            segformer_model: SegFormer model name from HuggingFace
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.canvas_size = canvas_size
        self._device = device
        self.use_segformer = use_segformer
        self.use_yolo = use_yolo
        self.yolo_model_name = yolo_model
        self.segformer_model_name = segformer_model
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Lazy-loaded models
        self._yolo_model = None
        self._segformer_model = None
        self._segformer_processor = None
        
        logger.info(f"AdvancedGarmentSegmenter initialized (canvas_size={canvas_size})")
        logger.info(f"  - YOLOv8: {'enabled' if use_yolo else 'disabled'} ({yolo_model})")
        logger.info(f"  - SegFormer: {'enabled' if use_segformer else 'disabled'} ({segformer_model})")
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if self._device:
            return self._device
        
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        except ImportError:
            self._device = "cpu"
        
        return self._device
    
    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------
    
    def _load_yolo(self) -> None:
        """Load YOLOv8 model for object detection."""
        if self._yolo_model is not None:
            return
        
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOv8.\n"
                "Install with: pip install ultralytics"
            )
        
        logger.info(f"Loading YOLOv8 model: {self.yolo_model_name}...")
        
        try:
            # Try to load custom fashion model first
            self._yolo_model = YOLO(self.yolo_model_name)
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load {self.yolo_model_name}: {e}")
            # Fallback to default model
            self._yolo_model = YOLO("yolov8n.pt")
            logger.info("Loaded default YOLOv8n model")
    
    def _load_segformer(self) -> None:
        """Load SegFormer model for semantic segmentation."""
        if self._segformer_model is not None:
            return
        
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for SegFormer.\n"
                "Install with: pip install transformers"
            )
        
        logger.info(f"Loading SegFormer model: {self.segformer_model_name}...")
        
        try:
            self._segformer_processor = SegformerImageProcessor.from_pretrained(
                self.segformer_model_name
            )
            self._segformer_model = SegformerForSemanticSegmentation.from_pretrained(
                self.segformer_model_name
            )
            self._segformer_model.to(self._get_device())
            self._segformer_model.eval()
            logger.info("SegFormer model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SegFormer model: {e}")
            raise
    
    # -------------------------------------------------------------------------
    # Detection Stage (YOLOv8)
    # -------------------------------------------------------------------------
    
    def _detect_garments(
        self, 
        image: np.ndarray,
        classes: Optional[List[str]] = None
    ) -> DetectionResult:
        """
        Detect garments in image using YOLOv8.
        
        Args:
            image: RGB image as numpy array
            classes: Optional list of class names to filter
            
        Returns:
            DetectionResult with boxes, labels, and scores
        """
        if not self.use_yolo:
            # Return full image as single detection
            h, w = image.shape[:2]
            return DetectionResult(
                boxes=[(0, 0, w, h)],
                labels=["garment"],
                scores=[1.0],
                class_ids=[0]
            )
        
        self._load_yolo()
        
        # Run YOLOv8 detection
        results = self._yolo_model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        boxes = []
        labels = []
        scores = []
        class_ids = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box, cls, conf in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy()
            ):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)
                class_name = result.names.get(class_id, "unknown")
                
                # Filter for fashion-related items if using generic model
                if self._is_fashion_item(class_name):
                    boxes.append((x1, y1, x2, y2))
                    labels.append(class_name)
                    scores.append(float(conf))
                    class_ids.append(class_id)
        
        # If no fashion items detected, return full image
        if not boxes:
            h, w = image.shape[:2]
            return DetectionResult(
                boxes=[(0, 0, w, h)],
                labels=["garment"],
                scores=[0.5],
                class_ids=[0]
            )
        
        return DetectionResult(
            boxes=boxes,
            labels=labels,
            scores=scores,
            class_ids=class_ids
        )
    
    def _is_fashion_item(self, class_name: str) -> bool:
        """Check if detected class is a fashion item."""
        fashion_keywords = [
            "person", "handbag", "backpack", "tie", "suitcase",
            "shirt", "pants", "dress", "jacket", "coat", "shoes",
            "hat", "bag", "top", "bottom", "skirt", "sweater"
        ]
        return any(kw in class_name.lower() for kw in fashion_keywords)
    
    # -------------------------------------------------------------------------
    # Semantic Segmentation Stage (SegFormer)
    # -------------------------------------------------------------------------
    
    def _segment_semantic(self, image: np.ndarray) -> np.ndarray:
        """
        Generate semantic segmentation map using SegFormer.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Segmentation map where each pixel has a label ID
        """
        if not self.use_segformer:
            # Return a mask covering the whole image (assume everything is garment)
            return np.ones(image.shape[:2], dtype=np.uint8) * 4  # upper_clothes
        
        self._load_segformer()
        
        import torch
        
        # Preprocess image
        pil_image = Image.fromarray(image)
        inputs = self._segformer_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self._get_device()) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self._segformer_model(**inputs)
            logits = outputs.logits
        
        # Upsample to original size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False
        )
        
        # Get segmentation map
        seg_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        
        return seg_map.astype(np.uint8)
    
    def _get_garment_mask_from_segmentation(
        self, 
        seg_map: np.ndarray,
        garment_types: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract garment mask from segmentation map.
        
        Args:
            seg_map: Semantic segmentation map
            garment_types: Optional list of garment types to extract
            
        Returns:
            Binary mask of garment regions
        """
        mask = np.zeros(seg_map.shape, dtype=np.uint8)
        
        if garment_types is None:
            # Extract all garment types
            garment_types = list(GARMENT_LABELS.keys())
        
        for gtype in garment_types:
            if gtype in GARMENT_LABELS:
                for label_id in GARMENT_LABELS[gtype]:
                    mask[seg_map == label_id] = 255
        
        return mask
    
    def _get_garment_type_from_segmentation(self, seg_map: np.ndarray) -> GarmentType:
        """Determine dominant garment type from segmentation map."""
        # Count pixels for each garment type
        type_counts = {}
        for gtype, label_ids in GARMENT_LABELS.items():
            count = sum(np.sum(seg_map == lid) for lid in label_ids)
            type_counts[gtype] = count
        
        if not any(type_counts.values()):
            return GarmentType.UNKNOWN
        
        dominant_type = max(type_counts, key=type_counts.get)
        
        try:
            return GarmentType(dominant_type)
        except ValueError:
            return GarmentType.UNKNOWN
    
    # -------------------------------------------------------------------------
    # Mask Fusion Stage
    # -------------------------------------------------------------------------
    
    def _fuse_masks(
        self,
        detection_box: Tuple[int, int, int, int],
        segmentation_mask: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Fuse detection bounding box with segmentation mask.
        
        The fusion strategy:
        1. Create a box mask from detection
        2. AND with segmentation mask to get precise garment region
        3. If segmentation mask is empty in box region, use box mask
        
        Args:
            detection_box: (x1, y1, x2, y2) bounding box
            segmentation_mask: Binary segmentation mask
            image_shape: (height, width) of original image
            
        Returns:
            Fused binary mask
        """
        h, w = image_shape
        
        # Create box mask
        box_mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = detection_box
        box_mask[y1:y2, x1:x2] = 255
        
        # Fuse with segmentation mask (intersection)
        fused_mask = cv2.bitwise_and(box_mask, segmentation_mask)
        
        # If fused mask is too small, use box mask instead
        fused_area = np.sum(fused_mask > 0)
        box_area = (x2 - x1) * (y2 - y1)
        
        if fused_area < box_area * 0.1:  # Less than 10% coverage
            logger.debug("Segmentation mask too sparse, using detection box")
            return box_mask
        
        return fused_mask
    
    # -------------------------------------------------------------------------
    # Mask Cleaning Stage (OpenCV)
    # -------------------------------------------------------------------------
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean and refine mask using OpenCV operations.
        
        Operations:
        1. Morphological closing to fill holes
        2. Morphological opening to remove noise
        3. Gaussian blur for smooth edges
        4. Threshold to get clean binary mask
        5. Find largest contour to remove small artifacts
        """
        # Morphological closing (fill holes)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Morphological opening (remove noise)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # Gaussian blur for smooth edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Threshold
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Keep only largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest], -1, 255, -1)
        
        return mask
    
    def _smooth_contours(self, mask: np.ndarray, epsilon_factor: float = 0.005) -> np.ndarray:
        """
        Smooth mask contours using contour approximation.
        
        Args:
            mask: Binary mask
            epsilon_factor: Factor for contour approximation (smaller = more detail)
            
        Returns:
            Mask with smoothed contours
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        smoothed_mask = np.zeros_like(mask)
        
        for contour in contours:
            # Approximate contour
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Draw smoothed contour
            cv2.drawContours(smoothed_mask, [approx], -1, 255, -1)
        
        return smoothed_mask
    
    def _feather_edges(self, mask: np.ndarray, feather_amount: int = 3) -> np.ndarray:
        """Apply feathering to mask edges for smoother blending."""
        # Create distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Normalize and create alpha gradient at edges
        dist_normalized = np.clip(dist / feather_amount, 0, 1)
        
        return (dist_normalized * 255).astype(np.uint8)
    
    # -------------------------------------------------------------------------
    # Main Segmentation Pipeline
    # -------------------------------------------------------------------------
    
    def segment_garment(
        self,
        image_path: Union[str, Path],
        garment_types: Optional[List[str]] = None,
        smooth_contours: bool = True
    ) -> SegmentedGarment:
        """
        Extract and segment a garment from an image using the full pipeline.
        
        Pipeline:
        1. YOLOv8 -> Detect garment bounding box
        2. SegFormer -> Generate semantic segmentation
        3. Mask Fusion -> Combine detection and segmentation
        4. OpenCV -> Clean and smooth mask edges
        
        Args:
            image_path: Path to the image file
            garment_types: Optional list of garment types to extract
            smooth_contours: Whether to smooth mask contours
            
        Returns:
            SegmentedGarment with RGBA image
        """
        image_path = Path(image_path)
        logger.info(f"Segmenting garment: {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Stage 1: Detect garments with YOLOv8
        logger.debug("Stage 1: YOLOv8 detection")
        try:
            detection = self._detect_garments(image_rgb)
        except Exception as e:
            logger.warning(f"YOLOv8 failed: {e}, using full image")
            detection = DetectionResult(
                boxes=[(0, 0, w, h)],
                labels=["garment"],
                scores=[1.0],
                class_ids=[0]
            )
        
        if not detection.boxes:
            logger.warning("No garments detected, using full image")
            detection = DetectionResult(
                boxes=[(0, 0, w, h)],
                labels=["garment"],
                scores=[0.5],
                class_ids=[0]
            )
        
        # Use the highest confidence detection
        best_idx = np.argmax(detection.scores)
        box = detection.boxes[best_idx]
        label = detection.labels[best_idx]
        confidence = detection.scores[best_idx]
        
        logger.debug(f"Best detection: {label} ({confidence:.2f})")
        
        # Stage 2: Semantic segmentation with SegFormer
        logger.debug("Stage 2: SegFormer segmentation")
        try:
            seg_map = self._segment_semantic(image_rgb)
            segmentation_mask = self._get_garment_mask_from_segmentation(seg_map, garment_types)
            garment_type = self._get_garment_type_from_segmentation(seg_map)
        except Exception as e:
            logger.warning(f"SegFormer failed: {e}, using detection only")
            segmentation_mask = np.ones((h, w), dtype=np.uint8) * 255
            garment_type = self._get_garment_type_from_label(label)
        
        # Stage 3: Mask fusion
        logger.debug("Stage 3: Mask fusion")
        fused_mask = self._fuse_masks(box, segmentation_mask, (h, w))
        
        # Stage 4: Mask cleaning with OpenCV
        logger.debug("Stage 4: OpenCV mask cleaning")
        mask = self._clean_mask(fused_mask)
        
        if smooth_contours:
            mask = self._smooth_contours(mask)
        
        # Create RGBA output
        rgba = self._apply_mask(image_rgb, mask)
        
        # Crop to bounding box of mask
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)
            crop_rgba = rgba[y:y+bh, x:x+bw]
            final_box = (x, y, bw, bh)
        else:
            x1, y1, x2, y2 = box
            crop_rgba = rgba[y1:y2, x1:x2]
            final_box = (x1, y1, x2-x1, y2-y1)
        
        # Convert to PIL and resize
        pil_image = Image.fromarray(crop_rgba)
        final_image = self._resize_and_center(pil_image)
        
        # Infer garment type from label if not from segmentation
        if garment_type == GarmentType.UNKNOWN:
            garment_type = self._get_garment_type_from_label(label)
        
        return SegmentedGarment(
            image=final_image,
            original_path=image_path,
            bounding_box=final_box,
            area=int(np.sum(mask > 0)),
            confidence=confidence,
            garment_type=garment_type,
            label=label,
            mask=mask
        )
    
    def _get_garment_type_from_label(self, label: str) -> GarmentType:
        """Infer garment type from detection label."""
        label_lower = label.lower()
        
        if any(w in label_lower for w in ["shirt", "t-shirt", "blouse", "top", "sweater", "hoodie", "polo"]):
            return GarmentType.TOP
        elif any(w in label_lower for w in ["jacket", "coat", "blazer", "cardigan"]):
            return GarmentType.OUTERWEAR
        elif any(w in label_lower for w in ["pants", "jeans", "trousers", "shorts", "skirt"]):
            return GarmentType.BOTTOM
        elif any(w in label_lower for w in ["dress", "gown", "jumpsuit"]):
            return GarmentType.DRESS
        elif any(w in label_lower for w in ["shoe", "sneaker", "boot", "sandal", "heel"]):
            return GarmentType.SHOES
        elif any(w in label_lower for w in ["hat", "bag", "scarf", "belt", "watch"]):
            return GarmentType.ACCESSORIES
        
        return GarmentType.UNKNOWN
    
    def _create_box_mask(self, shape: Tuple[int, int], box: Tuple[int, int, int, int]) -> np.ndarray:
        """Create a simple rectangular mask from bounding box."""
        mask = np.zeros(shape, dtype=np.uint8)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255
        return mask
    
    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to image to create RGBA with transparency."""
        rgba = np.dstack((image, mask))
        return rgba
    
    def _resize_and_center(self, img: Image.Image) -> Image.Image:
        """Resize and center image on transparent canvas."""
        w, h = img.size
        if w == 0 or h == 0:
            return Image.new("RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 0))
        
        scale = self.canvas_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        canvas = Image.new("RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 0))
        x = (self.canvas_size - img.width) // 2
        y = (self.canvas_size - img.height) // 2
        canvas.paste(img, (x, y), img)
        
        return canvas
    
    def segment_folder(
        self,
        folder_path: Union[str, Path],
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    ) -> List[SegmentedGarment]:
        """Segment all garments in a folder."""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        image_files = [
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        
        logger.info(f"Found {len(image_files)} images in {folder_path}")
        
        segmented = []
        for img_path in sorted(image_files):
            try:
                garment = self.segment_garment(img_path)
                segmented.append(garment)
                logger.info(f"  Segmented: {img_path.name} ({garment.label})")
            except Exception as e:
                logger.warning(f"  Failed: {img_path.name}: {e}")
        
        return segmented
    
    def save_segmented(
        self,
        garment: SegmentedGarment,
        output_path: Union[str, Path]
    ) -> Path:
        """Save a segmented garment to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        garment.image.save(output_path, "PNG")
        return output_path


# ============================================================================
# Simple Segmenter (Fallback without advanced models)
# ============================================================================

class SimpleGarmentSegmenter:
    """
    Simple garment segmenter using basic image processing.
    Use this when YOLOv8/SegFormer are not available.
    
    Uses:
    - GrabCut for foreground extraction
    - Edge detection for contour finding
    - Morphological operations for cleanup
    """
    
    def __init__(self, canvas_size: int = DEFAULT_CANVAS_SIZE):
        self.canvas_size = canvas_size
        logger.info(f"SimpleGarmentSegmenter initialized (canvas_size={canvas_size})")
    
    def segment_garment(self, image_path: Union[str, Path]) -> SegmentedGarment:
        """
        Segment garment using GrabCut algorithm.
        """
        image_path = Path(image_path)
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Use GrabCut for foreground extraction
        mask = self._grabcut_segment(image)
        
        # Clean the mask
        mask = self._clean_mask(mask)
        
        # Get bounding box
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)
        else:
            x, y, bw, bh = 0, 0, w, h
        
        # Create RGBA
        rgba = np.dstack((image_rgb, mask))
        
        # Crop to content
        if coords is not None:
            crop = rgba[y:y+bh, x:x+bw]
        else:
            crop = rgba
        
        pil_image = Image.fromarray(crop)
        final_image = self._resize_and_center(pil_image)
        
        return SegmentedGarment(
            image=final_image,
            original_path=image_path,
            bounding_box=(x, y, bw, bh),
            area=int(np.sum(mask > 0)),
            confidence=0.7,
            garment_type=GarmentType.UNKNOWN,
            label="garment"
        )
    
    def _grabcut_segment(self, image: np.ndarray) -> np.ndarray:
        """Use GrabCut algorithm for foreground extraction."""
        h, w = image.shape[:2]
        
        # Initialize mask
        mask = np.zeros((h, w), np.uint8)
        
        # Define rectangle for initial foreground (center region)
        margin = 10
        rect = (margin, margin, w - 2*margin, h - 2*margin)
        
        # Allocate arrays for GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Run GrabCut
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
        except Exception:
            # Fallback to edge-based detection
            mask2 = self._edge_based_segment(image)
        
        return mask2
    
    def _edge_based_segment(self, image: np.ndarray) -> np.ndarray:
        """Fallback edge-based segmentation."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest], -1, 255, -1)
            
            # Fill interior
            kernel = np.ones((10, 10), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=1)
        
        return mask
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean mask using morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Keep largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest], -1, 255, -1)
        
        return mask
    
    def _resize_and_center(self, img: Image.Image) -> Image.Image:
        """Resize and center on transparent canvas."""
        w, h = img.size
        if w == 0 or h == 0:
            return Image.new("RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 0))
        
        scale = self.canvas_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        canvas = Image.new("RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 0))
        x = (self.canvas_size - img.width) // 2
        y = (self.canvas_size - img.height) // 2
        canvas.paste(img, (x, y), img)
        
        return canvas
    
    def segment_folder(
        self,
        folder_path: Union[str, Path],
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    ) -> List[SegmentedGarment]:
        """Segment all garments in a folder."""
        folder_path = Path(folder_path)
        
        image_files = [
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        
        segmented = []
        for img_path in sorted(image_files):
            try:
                garment = self.segment_garment(img_path)
                segmented.append(garment)
            except Exception as e:
                logger.warning(f"Failed to segment {img_path.name}: {e}")
        
        return segmented


# ============================================================================
# Legacy GarmentSegmenter (SAM-only, kept for backward compatibility)
# ============================================================================

class GarmentSegmenter:
    """
    Legacy SAM-based segmenter. Consider using AdvancedGarmentSegmenter instead.
    Kept for backward compatibility.
    
    Note: The recommended approach is now YOLOv8 + SegFormer.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "vit_h",
        device: Optional[str] = None,
        canvas_size: int = DEFAULT_CANVAS_SIZE
    ):
        self.canvas_size = canvas_size
        self.checkpoint_path = checkpoint_path or "sam_vit_h_4b8939.pth"
        self.model_type = model_type
        self._sam = None
        self._mask_generator = None
        self._device = device
        
        logger.info(f"GarmentSegmenter (legacy) initialized (canvas_size={canvas_size})")
    
    def _ensure_model_loaded(self) -> None:
        """Lazy load SAM model."""
        if self._sam is not None:
            return
        
        try:
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError("segment_anything is required")
        
        if self._device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {self.checkpoint_path}")
        
        self._sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self._sam.to(self._device)
        self._mask_generator = SamAutomaticMaskGenerator(self._sam)
    
    def segment_garment(self, image_path: Union[str, Path]) -> SegmentedGarment:
        """Segment using SAM automatic mask generation."""
        self._ensure_model_loaded()
        
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        masks = self._mask_generator.generate(image_rgb)
        
        if not masks:
            return self._create_fallback(image_rgb, image_path)
        
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        selected = masks[0]
        mask = selected["segmentation"].astype(np.uint8) * 255
        
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        
        crop = image_rgb[y:y+h, x:x+w]
        mask_crop = mask[y:y+h, x:x+w]
        
        rgba = np.dstack((crop, mask_crop))
        pil_image = Image.fromarray(rgba)
        final_image = self._resize_and_center(pil_image)
        
        return SegmentedGarment(
            image=final_image,
            original_path=image_path,
            bounding_box=(x, y, w, h),
            area=selected["area"],
            confidence=selected.get("predicted_iou", 1.0)
        )
    
    def _create_fallback(self, image_rgb: np.ndarray, image_path: Path) -> SegmentedGarment:
        """Fallback when SAM fails."""
        h, w = image_rgb.shape[:2]
        alpha = np.ones((h, w), dtype=np.uint8) * 255
        rgba = np.dstack((image_rgb, alpha))
        pil_image = Image.fromarray(rgba)
        final_image = self._resize_and_center(pil_image)
        
        return SegmentedGarment(
            image=final_image,
            original_path=image_path,
            bounding_box=(0, 0, w, h),
            area=w * h,
            confidence=0.5
        )
    
    def _resize_and_center(self, img: Image.Image) -> Image.Image:
        """Resize and center on canvas."""
        w, h = img.size
        scale = self.canvas_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        canvas = Image.new("RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 0))
        x = (self.canvas_size - img.width) // 2
        y = (self.canvas_size - img.height) // 2
        canvas.paste(img, (x, y), img)
        
        return canvas
    
    def segment_folder(self, folder_path: Union[str, Path], extensions=(".jpg", ".jpeg", ".png", ".webp")) -> List[SegmentedGarment]:
        """Segment all images in folder."""
        folder_path = Path(folder_path)
        image_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in extensions]
        
        segmented = []
        for img_path in sorted(image_files):
            try:
                garment = self.segment_garment(img_path)
                segmented.append(garment)
            except Exception as e:
                logger.warning(f"Failed: {img_path.name}: {e}")
        
        return segmented
    
    def save_segmented(self, garment: SegmentedGarment, output_path: Union[str, Path]) -> Path:
        """Save segmented garment."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        garment.image.save(output_path, "PNG")
        return output_path
