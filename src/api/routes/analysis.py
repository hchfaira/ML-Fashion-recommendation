"""
Analysis API Routes
Endpoints for analyzing images and outfits.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional

from src.layer1_vision import VisionService, ClothingSegmenter, AttributeExtractor
from src.layer2_style import ColorHarmonyAnalyzer, SilhouetteAnalyzer
from src.core import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Lazy-loaded service instances
_vision_service = None
_segmenter = None
_attribute_extractor = None
_color_analyzer = None
_silhouette_analyzer = None


def get_vision_service() -> VisionService:
    """Get or create VisionService instance."""
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service


def get_segmenter() -> ClothingSegmenter:
    """Get or create ClothingSegmenter instance."""
    global _segmenter
    if _segmenter is None:
        _segmenter = ClothingSegmenter()
    return _segmenter


def get_attribute_extractor() -> AttributeExtractor:
    """Get or create AttributeExtractor instance."""
    global _attribute_extractor
    if _attribute_extractor is None:
        _attribute_extractor = AttributeExtractor()
    return _attribute_extractor


def get_color_analyzer() -> ColorHarmonyAnalyzer:
    """Get or create ColorHarmonyAnalyzer instance."""
    global _color_analyzer
    if _color_analyzer is None:
        _color_analyzer = ColorHarmonyAnalyzer()
    return _color_analyzer


def get_silhouette_analyzer() -> SilhouetteAnalyzer:
    """Get or create SilhouetteAnalyzer instance."""
    global _silhouette_analyzer
    if _silhouette_analyzer is None:
        _silhouette_analyzer = SilhouetteAnalyzer()
    return _silhouette_analyzer


# For backward compatibility with tests
vision_service = None
segmenter = None
attribute_extractor = None
color_analyzer = None
silhouette_analyzer = None


@router.post("/image")
async def analyze_image(
    image_url: Optional[str] = None,
    image: Optional[UploadFile] = File(None)
):
    """
    Analyze a clothing image and extract all attributes.
    
    Uses Layer 1 vision to:
    - Identify garment type
    - Extract colors
    - Detect pattern
    - Determine style attributes
    """
    if not image_url and not image:
        raise HTTPException(
            status_code=400,
            detail="Either image_url or image file required"
        )
    
    try:
        vs = get_vision_service()
        if image:
            image_bytes = await image.read()
            analysis = await vs.analyze_image(image_bytes)
        else:
            analysis = await vs.analyze_image(image_url)
        
        return {
            "status": "success",
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/outfit-image")
async def analyze_outfit_image(
    image_url: Optional[str] = None,
    image: Optional[UploadFile] = File(None)
):
    """
    Analyze a complete outfit image.
    
    Segments and analyzes each visible garment separately.
    """
    if not image_url and not image:
        raise HTTPException(
            status_code=400,
            detail="Either image_url or image file required"
        )
    
    try:
        source = image_url
        if image:
            # Would handle file upload
            source = image_url  # Simplified
        
        # Segment the outfit
        seg = get_segmenter()
        items = await seg.segment_outfit(source)
        
        return {
            "status": "success",
            "items_found": len(items),
            "items": items
        }
        
    except Exception as e:
        logger.error(f"Outfit analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/color-harmony")
async def check_color_harmony(colors: list[str]):
    """
    Check color harmony between given colors.
    """
    if len(colors) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 colors to analyze harmony"
        )
    
    # Create mock garments with colors for analysis
    from src.core.models import Garment, GarmentAttributes, ColorInfo, GarmentCategory
    
    garments = []
    for i, color in enumerate(colors):
        garment = Garment(
            id=f"temp_{i}",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                color=ColorInfo(primary=color, hex_codes=[])
            )
        )
        garments.append(garment)
    
    ca = get_color_analyzer()
    score = ca.analyze_outfit_colors(garments)
    
    return {
        "colors": colors,
        "harmony_score": score,
        "recommendation": "good" if score >= 0.7 else "could be improved"
    }


@router.get("/complementary-colors/{color}")
async def get_complementary_colors(color: str):
    """
    Get colors that complement the given color.
    """
    ca = get_color_analyzer()
    complementary = ca.get_complementary_colors(color)
    
    return {
        "input_color": color,
        "complementary_colors": complementary
    }
