# Layer 1: Vision & Clothing Parsing
from .vision_service import VisionService
from .segmentation import ClothingSegmenter
from .attribute_extractor import (
    AttributeExtractor,
    ExtractionMode,
    DEFAULT_EXTRACTION_MODE
)
from .embedding_generator import EmbeddingGenerator
from .vision_cache import VisionCache, get_vision_cache
from .local_classifier import LocalGarmentClassifier, ClassificationResult
from .vision_queue import VisionQueue, VisionJob, JobStatus

__all__ = [
    "VisionService",
    "ClothingSegmenter",
    "AttributeExtractor",
    "ExtractionMode",
    "DEFAULT_EXTRACTION_MODE",
    "EmbeddingGenerator",
    # Solution 1 — persistent vision cache
    "VisionCache",
    "get_vision_cache",
    # Solution 4 — lightweight local pre-classifier
    "LocalGarmentClassifier",
    "ClassificationResult",
    # Solutions 3+5 — async queue worker system
    "VisionQueue",
    "VisionJob",
    "JobStatus",
]
