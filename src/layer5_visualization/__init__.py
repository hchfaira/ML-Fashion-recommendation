"""
Layer 5: Outfit Visualization
==============================

This layer generates visual representations of outfit recommendations,
including catalogue images showing the selected garments.

Responsibilities:
- Generate catalogue images from selected outfits
- Create visual outfit comparisons
- Export outfit images for sharing
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
import math

from src.core import get_logger
from src.core.models import Garment

logger = get_logger(__name__)

# Constants
DEFAULT_GARMENT_SIZE = 400
DEFAULT_PADDING = 20
DEFAULT_BACKGROUND_COLOR = (255, 255, 255, 255)  # White
DEFAULT_BORDER_COLOR = (200, 200, 200, 255)  # Light gray


@dataclass
class OutfitVisualization:
    """Represents a generated outfit visualization."""
    image: Image.Image
    outfit_name: str
    garment_count: int
    score: Optional[float] = None
    output_path: Optional[Path] = None


class OutfitVisualizer:
    """
    Generates visual representations of outfit recommendations.
    
    This is Layer 5 of the pipeline - creating catalogue images
    from the best outfit combinations.
    """
    
    def __init__(
        self,
        garment_size: int = DEFAULT_GARMENT_SIZE,
        padding: int = DEFAULT_PADDING,
        background_color: Tuple[int, int, int, int] = DEFAULT_BACKGROUND_COLOR,
        show_labels: bool = True,
        show_score: bool = True
    ):
        """
        Initialize the outfit visualizer.
        
        Args:
            garment_size: Size of each garment cell in pixels
            padding: Padding between garments
            background_color: Background color (RGBA)
            show_labels: Whether to show category labels
            show_score: Whether to show outfit score
        """
        self.garment_size = garment_size
        self.padding = padding
        self.background_color = background_color
        self.show_labels = show_labels
        self.show_score = show_score
        
        # Try to load a font, fallback to default
        self._font = None
        self._font_small = None
        self._load_fonts()
        
        logger.info(f"OutfitVisualizer initialized (size={garment_size})")
    
    def _load_fonts(self) -> None:
        """Load fonts for labels."""
        try:
            # Try common font paths
            font_paths = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/SFNSText.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:/Windows/Fonts/arial.ttf",
            ]
            
            for font_path in font_paths:
                if Path(font_path).exists():
                    self._font = ImageFont.truetype(font_path, 24)
                    self._font_small = ImageFont.truetype(font_path, 16)
                    break
        except Exception:
            pass
        
        if self._font is None:
            self._font = ImageFont.load_default()
            self._font_small = ImageFont.load_default()
    
    def create_outfit_catalogue(
        self,
        garments: List[Garment],
        outfit_name: str = "Best Outfit",
        score: Optional[float] = None,
        columns: Optional[int] = None,
        segmented_images: Optional[Dict[str, Image.Image]] = None
    ) -> OutfitVisualization:
        """
        Create a catalogue image showing all garments in an outfit.
        
        Args:
            garments: List of Garment objects
            outfit_name: Name/title for the outfit
            score: Overall outfit score (0-1)
            columns: Number of columns (auto-calculated if None)
            segmented_images: Dict mapping garment.id to segmented PIL Image
            
        Returns:
            OutfitVisualization with the catalogue image
        """
        if not garments:
            raise ValueError("Cannot create catalogue with no garments")
        
        logger.info(f"Creating outfit catalogue: {outfit_name} ({len(garments)} garments)")
        
        # Calculate grid dimensions
        n = len(garments)
        if columns is None:
            columns = min(n, 3)  # Max 3 columns
        rows = math.ceil(n / columns)
        
        # Calculate canvas size
        cell_size = self.garment_size + self.padding * 2
        header_height = 80 if (self.show_labels or self.show_score) else 0
        
        canvas_width = columns * cell_size + self.padding * 2
        canvas_height = rows * cell_size + header_height + self.padding * 2
        
        # Create canvas
        canvas = Image.new("RGBA", (canvas_width, canvas_height), self.background_color)
        draw = ImageDraw.Draw(canvas)
        
        # Draw header
        if header_height > 0:
            self._draw_header(draw, canvas_width, outfit_name, score)
        
        # Draw garments
        for i, garment in enumerate(garments):
            col = i % columns
            row = i // columns
            
            x = self.padding + col * cell_size + self.padding
            y = header_height + self.padding + row * cell_size + self.padding
            
            # Get garment image
            garment_img = self._get_garment_image(garment, segmented_images)
            
            # Draw garment cell
            self._draw_garment_cell(
                canvas, draw, garment_img, garment, 
                x, y, self.garment_size
            )
        
        return OutfitVisualization(
            image=canvas,
            outfit_name=outfit_name,
            garment_count=len(garments),
            score=score
        )
    
    def _draw_header(
        self, 
        draw: ImageDraw.Draw, 
        width: int, 
        title: str, 
        score: Optional[float]
    ) -> None:
        """Draw the header with title and score."""
        # Title
        title_bbox = draw.textbbox((0, 0), title, font=self._font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, 15), title, fill=(50, 50, 50, 255), font=self._font)
        
        # Score
        if score is not None and self.show_score:
            score_text = f"Score: {score:.1%}"
            
            # Color based on score
            if score >= 0.8:
                color = (46, 204, 113, 255)  # Green
            elif score >= 0.6:
                color = (241, 196, 15, 255)  # Yellow
            else:
                color = (231, 76, 60, 255)  # Red
            
            score_bbox = draw.textbbox((0, 0), score_text, font=self._font_small)
            score_width = score_bbox[2] - score_bbox[0]
            score_x = (width - score_width) // 2
            draw.text((score_x, 45), score_text, fill=color, font=self._font_small)
    
    def _get_garment_image(
        self, 
        garment: Garment,
        segmented_images: Optional[Dict[str, Image.Image]]
    ) -> Optional[Image.Image]:
        """Get the image for a garment."""
        # Check segmented images dict
        if segmented_images and garment.id in segmented_images:
            return segmented_images[garment.id]
        
        # Try to load from image_path
        if garment.image_path and Path(garment.image_path).exists():
            try:
                img = Image.open(garment.image_path)
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                return img
            except Exception as e:
                logger.warning(f"Could not load image {garment.image_path}: {e}")
        
        return None
    
    def _draw_garment_cell(
        self,
        canvas: Image.Image,
        draw: ImageDraw.Draw,
        garment_img: Optional[Image.Image],
        garment: Garment,
        x: int,
        y: int,
        size: int
    ) -> None:
        """Draw a single garment cell."""
        # Draw border
        draw.rectangle(
            [x - 2, y - 2, x + size + 2, y + size + 2],
            outline=DEFAULT_BORDER_COLOR,
            width=2
        )
        
        # Draw garment image or placeholder
        if garment_img:
            # Resize to fit
            img_resized = self._resize_to_fit(garment_img, size)
            
            # Center in cell
            paste_x = x + (size - img_resized.width) // 2
            paste_y = y + (size - img_resized.height) // 2
            
            # Paste with alpha
            canvas.paste(img_resized, (paste_x, paste_y), img_resized)
        else:
            # Draw placeholder
            self._draw_placeholder(draw, garment, x, y, size)
        
        # Draw label
        if self.show_labels:
            label = self._get_garment_label(garment)
            label_bbox = draw.textbbox((0, 0), label, font=self._font_small)
            label_width = label_bbox[2] - label_bbox[0]
            label_x = x + (size - label_width) // 2
            label_y = y + size + 5
            
            # Background for label
            draw.rectangle(
                [label_x - 3, label_y - 2, label_x + label_width + 3, label_y + 18],
                fill=(255, 255, 255, 200)
            )
            draw.text((label_x, label_y), label, fill=(80, 80, 80, 255), font=self._font_small)
    
    def _resize_to_fit(self, img: Image.Image, size: int) -> Image.Image:
        """Resize image to fit within size while maintaining aspect ratio."""
        w, h = img.size
        scale = min(size / w, size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    def _draw_placeholder(
        self, 
        draw: ImageDraw.Draw, 
        garment: Garment, 
        x: int, 
        y: int, 
        size: int
    ) -> None:
        """Draw a placeholder when image is not available."""
        # Background
        color = self._get_placeholder_color(garment)
        draw.rectangle([x, y, x + size, y + size], fill=color)
        
        # Category text
        category = garment.attributes.category.value.upper()
        cat_bbox = draw.textbbox((0, 0), category, font=self._font)
        cat_width = cat_bbox[2] - cat_bbox[0]
        cat_x = x + (size - cat_width) // 2
        cat_y = y + size // 2 - 12
        draw.text((cat_x, cat_y), category, fill=(255, 255, 255, 255), font=self._font)
    
    def _get_placeholder_color(self, garment: Garment) -> Tuple[int, int, int, int]:
        """Get a color for placeholder based on garment attributes."""
        color_name = garment.attributes.color.primary.lower() if garment.attributes.color else "gray"
        
        color_map = {
            "white": (240, 240, 240, 255),
            "black": (50, 50, 50, 255),
            "gray": (150, 150, 150, 255),
            "grey": (150, 150, 150, 255),
            "red": (231, 76, 60, 255),
            "blue": (52, 152, 219, 255),
            "navy": (44, 62, 80, 255),
            "green": (46, 204, 113, 255),
            "yellow": (241, 196, 15, 255),
            "orange": (230, 126, 34, 255),
            "purple": (155, 89, 182, 255),
            "pink": (255, 192, 203, 255),
            "brown": (139, 90, 43, 255),
            "beige": (245, 245, 220, 255),
            "cream": (255, 253, 208, 255),
            "tan": (210, 180, 140, 255),
            "lavender": (230, 230, 250, 255),
            "lilac": (200, 162, 200, 255),
        }
        
        return color_map.get(color_name, (180, 180, 180, 255))
    
    def _get_garment_label(self, garment: Garment) -> str:
        """Generate a label for a garment."""
        parts = []
        
        # Color
        if garment.attributes.color and garment.attributes.color.primary:
            parts.append(garment.attributes.color.primary.title())
        
        # Subcategory or category
        if garment.attributes.subcategory:
            parts.append(garment.attributes.subcategory.title())
        else:
            parts.append(garment.attributes.category.value.title())
        
        return " ".join(parts)
    
    def create_comparison_catalogue(
        self,
        outfits: List[Tuple[str, List[Garment], float]],
        title: str = "Outfit Comparison",
        segmented_images: Optional[Dict[str, Image.Image]] = None
    ) -> OutfitVisualization:
        """
        Create a comparison image showing multiple outfits side by side.
        
        Args:
            outfits: List of (name, garments, score) tuples
            title: Title for the comparison
            segmented_images: Dict mapping garment.id to segmented PIL Image
            
        Returns:
            OutfitVisualization with comparison image
        """
        if not outfits:
            raise ValueError("No outfits to compare")
        
        # Create individual catalogues
        individual_images = []
        for name, garments, score in outfits:
            viz = self.create_outfit_catalogue(
                garments=garments,
                outfit_name=name,
                score=score,
                columns=1,  # Single column for comparison
                segmented_images=segmented_images
            )
            individual_images.append(viz.image)
        
        # Calculate combined canvas size
        total_width = sum(img.width for img in individual_images) + self.padding * (len(individual_images) + 1)
        max_height = max(img.height for img in individual_images)
        header_height = 50
        
        # Create canvas
        canvas = Image.new("RGBA", (total_width, max_height + header_height), self.background_color)
        draw = ImageDraw.Draw(canvas)
        
        # Draw title
        title_bbox = draw.textbbox((0, 0), title, font=self._font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((total_width - title_width) // 2, 10), title, fill=(50, 50, 50, 255), font=self._font)
        
        # Paste individual images
        x = self.padding
        for img in individual_images:
            canvas.paste(img, (x, header_height), img)
            x += img.width + self.padding
        
        return OutfitVisualization(
            image=canvas,
            outfit_name=title,
            garment_count=sum(len(g) for _, g, _ in outfits),
            score=outfits[0][2] if outfits else None
        )
    
    def save_visualization(
        self,
        visualization: OutfitVisualization,
        output_path: Union[str, Path],
        format: str = "PNG"
    ) -> Path:
        """
        Save a visualization to file.
        
        Args:
            visualization: OutfitVisualization to save
            output_path: Output file path
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to RGB for JPEG
        if format.upper() == "JPEG":
            img = visualization.image.convert("RGB")
        else:
            img = visualization.image
        
        img.save(output_path, format)
        visualization.output_path = output_path
        
        logger.info(f"Saved visualization to: {output_path}")
        return output_path


def create_outfit_image(
    garments: List[Garment],
    output_path: Union[str, Path],
    outfit_name: str = "Best Outfit",
    score: Optional[float] = None,
    segmented_images: Optional[Dict[str, Image.Image]] = None
) -> Path:
    """
    Convenience function to create and save an outfit catalogue image.
    
    Args:
        garments: List of Garment objects
        output_path: Where to save the image
        outfit_name: Name for the outfit
        score: Outfit score
        segmented_images: Optional dict of pre-segmented images
        
    Returns:
        Path to saved image
    """
    visualizer = OutfitVisualizer()
    viz = visualizer.create_outfit_catalogue(
        garments=garments,
        outfit_name=outfit_name,
        score=score,
        segmented_images=segmented_images
    )
    return visualizer.save_visualization(viz, output_path)
