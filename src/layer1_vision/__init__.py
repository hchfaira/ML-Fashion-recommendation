# Layer 1: Vision & Clothing Parsing
from .vision_service import VisionService
from .segmentation import ClothingSegmenter
from .attribute_extractor import (
    AttributeExtractor, 
    ExtractionMode, 
    DEFAULT_EXTRACTION_MODE
)
from .embedding_generator import EmbeddingGenerator

__all__ = [
    "VisionService",
    "ClothingSegmenter",
    "AttributeExtractor",
    "ExtractionMode",
    "DEFAULT_EXTRACTION_MODE",
    "EmbeddingGenerator"
]
