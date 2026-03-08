"""
Wardrobe API Routes
Endpoints for managing user wardrobes.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional
import uuid

from src.core.models import Garment, GarmentAttributes
from src.layer1_vision import VisionService, AttributeExtractor, EmbeddingGenerator
from src.core import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Lazy-loaded service instances
_vision_service = None
_attribute_extractor = None
_embedding_generator = None


def get_vision_service() -> VisionService:
    """Get or create VisionService instance."""
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service


def get_attribute_extractor() -> AttributeExtractor:
    """Get or create AttributeExtractor instance."""
    global _attribute_extractor
    if _attribute_extractor is None:
        _attribute_extractor = AttributeExtractor()
    return _attribute_extractor


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create EmbeddingGenerator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


# For backward compatibility with tests
vision_service = None
attribute_extractor = None
embedding_generator = None

# In-memory storage (would be database in production)
_wardrobes = {}


@router.post("/items", response_model=Garment)
async def add_garment(
    user_id: str,
    image_url: Optional[str] = None,
    image: Optional[UploadFile] = File(None)
):
    """
    Add a new garment to user's wardrobe.
    
    Analyzes the image using Layer 1 vision to extract
    attributes and generate embeddings.
    """
    if not image_url and not image:
        raise HTTPException(
            status_code=400,
            detail="Either image_url or image file required"
        )
    
    try:
        # Get image source
        if image:
            image_bytes = await image.read()
            image_source = image_bytes
        else:
            image_source = image_url
        
        # Analyze with vision service
        vs = get_vision_service()
        analysis = await vs.analyze_image(image_source)
        
        # Extract attributes
        extractor = get_attribute_extractor()
        attributes = await extractor.extract_attributes(
            image_url if image_url else f"data:image/jpeg;base64,..."
        )
        
        # Create garment
        garment = Garment(
            id=f"garment_{uuid.uuid4().hex[:12]}",
            image_url=image_url,
            attributes=attributes
        )
        
        # Generate embedding
        emb_gen = get_embedding_generator()
        garment.embedding = await emb_gen.generate_embedding(garment)
        
        # Store in wardrobe
        if user_id not in _wardrobes:
            _wardrobes[user_id] = []
        _wardrobes[user_id].append(garment)
        
        return garment
        
    except Exception as e:
        logger.error(f"Failed to add garment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/items", response_model=List[Garment])
async def get_wardrobe(user_id: str):
    """Get all items in user's wardrobe."""
    return _wardrobes.get(user_id, [])


@router.get("/items/{garment_id}", response_model=Garment)
async def get_garment(user_id: str, garment_id: str):
    """Get a specific garment."""
    wardrobe = _wardrobes.get(user_id, [])
    
    for garment in wardrobe:
        if garment.id == garment_id:
            return garment
    
    raise HTTPException(status_code=404, detail="Garment not found")


@router.delete("/items/{garment_id}")
async def remove_garment(user_id: str, garment_id: str):
    """Remove a garment from wardrobe."""
    if user_id not in _wardrobes:
        raise HTTPException(status_code=404, detail="Wardrobe not found")
    
    wardrobe = _wardrobes[user_id]
    _wardrobes[user_id] = [g for g in wardrobe if g.id != garment_id]
    
    return {"status": "deleted", "garment_id": garment_id}


@router.post("/bulk-upload")
async def bulk_upload(
    user_id: str,
    images: List[UploadFile] = File(...)
):
    """Upload multiple garment images at once."""
    results = {
        "successful": [],
        "failed": []
    }
    
    for image in images:
        try:
            garment = await add_garment(user_id=user_id, image=image)
            results["successful"].append(garment.id)
        except Exception as e:
            results["failed"].append({
                "filename": image.filename,
                "error": str(e)
            })
    
    return results
