"""
SCHP Human Parsing
==================

Self-Correction Human Parsing (SCHP) for clothing region classification.

Generates semantic segmentation maps that classify each pixel into
clothing categories (upper-clothes, pants, dress, etc.).

Maps SCHP labels to GARMENT_TAXONOMY categories for consistency.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple
from abc import ABC, abstractmethod

from src.core import get_logger
from .taxonomy import (
    SCHP_LABELS,
    SCHP_TO_TAXONOMY,
    GarmentCategory,
    get_schp_category,
    is_garment_schp_label
)
from .models import ParsingResult

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# SCHP model paths
SCHP_MODEL_PATH = "models/schp/exp-schp-201908301523-atr.pth"
SCHP_CONFIG_PATH = "models/schp/atr_schp_config.yaml"

# Number of classes in LIP/ATR dataset
NUM_CLASSES = 20


# ============================================================================
# Abstract Base Class
# ============================================================================

class BaseHumanParser(ABC):
    """Abstract base class for human parsing."""
    
    @abstractmethod
    def parse(
        self,
        image: Union[np.ndarray, Image.Image, Path, str]
    ) -> ParsingResult:
        """
        Parse human figure to classify clothing regions.
        
        Args:
            image: Input image
            
        Returns:
            ParsingResult with semantic segmentation map
        """
        pass


# ============================================================================
# SCHP Human Parser
# ============================================================================

class SCHPParser(BaseHumanParser):
    """
    Human parsing using Self-Correction Human Parsing (SCHP) model.
    
    Produces a pixel-wise classification map where each pixel is
    labeled with a clothing category from SCHP_LABELS.
    """
    
    def __init__(
        self,
        model_path: str = SCHP_MODEL_PATH,
        config_path: str = SCHP_CONFIG_PATH,
        device: Optional[str] = None,
        input_size: Tuple[int, int] = (473, 473)
    ):
        """
        Initialize the SCHP parser.
        
        Args:
            model_path: Path to SCHP model checkpoint
            config_path: Path to SCHP config file
            device: Device to run on (cuda/cpu/mps)
            input_size: Input size for the model
        """
        self.model_path = model_path
        self.config_path = config_path
        self._device = device
        self.input_size = input_size
        
        # Lazy-loaded model
        self._model = None
        self._transform = None
        
        logger.info(f"SCHPParser initialized (input_size={input_size})")
    
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
    
    def _load_model(self) -> None:
        """Load the SCHP model."""
        if self._model is not None:
            return
        
        try:
            import torch
            import torchvision.transforms as transforms
        except ImportError:
            raise ImportError(
                "PyTorch is required for SCHP.\n"
                "Install with: pip install torch torchvision"
            )
        
        # Check for model file
        if not Path(self.model_path).exists():
            logger.warning(
                f"SCHP model not found: {self.model_path}. "
                "Using heuristic-based parsing as fallback."
            )
            self._model = "heuristic"
            return
        
        logger.info("Loading SCHP model...")
        
        try:
            # Import SCHP model architecture
            # Note: This requires the SCHP repo to be available
            from schp.networks import get_model
            
            self._model = get_model(num_classes=NUM_CLASSES)
            self._model.load_state_dict(torch.load(self.model_path, map_location=self._get_device()))
            self._model.to(self._get_device())
            self._model.eval()
            
            # Setup transform
            self._transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info(f"SCHP model loaded on {self._get_device()}")
            
        except (ImportError, Exception) as e:
            logger.warning(f"Could not load SCHP model: {e}. Using heuristic fallback.")
            self._model = "heuristic"
    
    def _load_image(
        self, 
        image: Union[np.ndarray, Image.Image, Path, str]
    ) -> Tuple[np.ndarray, Image.Image, Tuple[int, int]]:
        """Load and prepare image."""
        if isinstance(image, (str, Path)):
            image_pil = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image_pil = image.convert("RGB")
        else:
            # Numpy array
            if image.ndim == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            image_pil = Image.fromarray(image)
        
        image_np = np.array(image_pil)
        original_size = image_pil.size  # (W, H)
        
        return image_np, image_pil, (original_size[1], original_size[0])  # (H, W)
    
    def parse(
        self,
        image: Union[np.ndarray, Image.Image, Path, str]
    ) -> ParsingResult:
        """
        Parse human figure to classify clothing regions.
        
        Args:
            image: Input image
            
        Returns:
            ParsingResult with semantic segmentation map
        """
        self._load_model()
        
        image_np, image_pil, original_size = self._load_image(image)
        h, w = original_size
        
        # Use heuristic if model not available
        if self._model == "heuristic":
            return self._heuristic_parsing(image_np)
        
        import torch
        
        # Transform image
        input_tensor = self._transform(image_pil).unsqueeze(0).to(self._get_device())
        
        # Run inference
        with torch.no_grad():
            output = self._model(input_tensor)
            
            # Get argmax predictions
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            parsing = output.argmax(dim=1).squeeze().cpu().numpy()
        
        # Resize back to original size
        from PIL import Image as PILImage
        parsing_pil = PILImage.fromarray(parsing.astype(np.uint8))
        parsing_resized = np.array(parsing_pil.resize((w, h), PILImage.NEAREST))
        
        # Calculate label areas
        label_areas = {}
        for label_id in range(NUM_CLASSES):
            area = int(np.sum(parsing_resized == label_id))
            if area > 0:
                label_areas[label_id] = area
        
        result = ParsingResult(
            parsing_map=parsing_resized,
            label_areas=label_areas
        )
        
        logger.debug(f"Parsed {len(label_areas)} regions")
        return result
    
    def _heuristic_parsing(self, image: np.ndarray) -> ParsingResult:
        """
        Heuristic-based parsing when SCHP model is not available.
        
        Uses position-based heuristics to estimate clothing regions:
        - Upper 35% → upper_clothes (5)
        - Middle 30% → dress region if wide, otherwise upper continues
        - Lower 30% → pants (9)
        - Bottom 15% → shoes (18/19)
        """
        h, w = image.shape[:2]
        parsing_map = np.zeros((h, w), dtype=np.uint8)
        
        # Define regions
        upper_end = int(h * 0.35)
        middle_end = int(h * 0.65)
        pants_end = int(h * 0.85)
        
        # Upper region - tops
        parsing_map[:upper_end, :] = 5  # upper_clothes
        
        # Middle region - could be dress or continuation of top/start of pants
        parsing_map[upper_end:middle_end, :] = 5  # upper_clothes
        
        # Lower region - pants
        parsing_map[middle_end:pants_end, :] = 9  # pants
        
        # Bottom region - shoes
        parsing_map[pants_end:, :w//2] = 18  # left_shoe
        parsing_map[pants_end:, w//2:] = 19  # right_shoe
        
        # Calculate label areas
        label_areas = {}
        for label_id in [5, 9, 18, 19]:
            area = int(np.sum(parsing_map == label_id))
            if area > 0:
                label_areas[label_id] = area
        
        return ParsingResult(
            parsing_map=parsing_map,
            label_areas=label_areas
        )
    
    def get_garment_masks(
        self,
        parsing_result: ParsingResult
    ) -> Dict[GarmentCategory, np.ndarray]:
        """
        Extract binary masks for each garment category from parsing result.
        
        Args:
            parsing_result: Result from parse()
            
        Returns:
            Dictionary mapping category to binary mask
        """
        masks = {}
        
        for category in GarmentCategory:
            if category == GarmentCategory.UNKNOWN:
                continue
            
            mask = parsing_result.get_mask_for_category(category)
            if np.any(mask):
                masks[category] = mask
        
        return masks


# ============================================================================
# Fallback Parser
# ============================================================================

class FallbackParser(BaseHumanParser):
    """
    Simple fallback parser that uses position-based heuristics.
    Use when SCHP is not available.
    """
    
    def __init__(self):
        logger.info("FallbackParser initialized (heuristic-based)")
    
    def parse(
        self,
        image: Union[np.ndarray, Image.Image, Path, str]
    ) -> ParsingResult:
        """Parse using position-based heuristics."""
        # Load image
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        h, w = image.shape[:2]
        parsing_map = np.zeros((h, w), dtype=np.uint8)
        
        # Simple vertical segmentation
        parsing_map[:int(h * 0.35), :] = 5  # upper_clothes
        parsing_map[int(h * 0.35):int(h * 0.65), :] = 5  # upper_clothes
        parsing_map[int(h * 0.65):int(h * 0.85), :] = 9  # pants
        parsing_map[int(h * 0.85):, :] = 18  # shoes
        
        label_areas = {
            5: int(np.sum(parsing_map == 5)),
            9: int(np.sum(parsing_map == 9)),
            18: int(np.sum(parsing_map == 18))
        }
        
        return ParsingResult(
            parsing_map=parsing_map,
            label_areas=label_areas
        )


# ============================================================================
# Factory Function
# ============================================================================

def get_parser(
    use_schp: bool = True,
    **kwargs
) -> BaseHumanParser:
    """
    Factory function to get the appropriate parser.
    
    Args:
        use_schp: Whether to try loading SCHP model
        **kwargs: Additional arguments for the parser
        
    Returns:
        Parser instance
    """
    if use_schp:
        try:
            import torch
            return SCHPParser(**kwargs)
        except ImportError:
            logger.warning("PyTorch not available, using fallback parser")
    
    return FallbackParser()
