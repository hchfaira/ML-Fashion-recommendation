"""Custom exception classes for the Fashion Recommendation System."""


class FashionRecommenderError(Exception):
    """Base exception for the fashion recommender system."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class VisionProcessingError(FashionRecommenderError):
    """Exception raised when vision/image processing fails."""
    pass


class StyleModelError(FashionRecommenderError):
    """Exception raised when style intelligence model fails."""
    pass


class ContextEngineError(FashionRecommenderError):
    """Exception raised when context engine fails."""
    pass


class LLMError(FashionRecommenderError):
    """Exception raised when LLM processing fails."""
    pass


class EmbeddingError(FashionRecommenderError):
    """Exception raised when embedding generation fails."""
    pass


class CompatibilityError(FashionRecommenderError):
    """Exception raised when compatibility scoring fails."""
    pass
