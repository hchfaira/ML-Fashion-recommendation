"""
Tests for Layer 1: Embedding Generator - Unit Tests

Tests cover:
- Text conversion from garments
- Cache operations
- Similarity calculations
- Mock embedding generation

Note: Integration tests with real API calls are in tests/integration/
"""
import pytest
import numpy as np
from unittest.mock import MagicMock

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory,
    ColorInfo, PatternInfo, FormalityLevel, Season
)
from src.layer1_vision.embedding_generator import EmbeddingGenerator


# ============== Constants ==============

EMBEDDING_DIM = 3072


# ============== Helper Functions ==============

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    
    dot_product = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def generate_mock_embedding(seed=None, dim=EMBEDDING_DIM):
    """Generate a mock embedding vector for testing."""
    if seed is not None:
        np.random.seed(seed)
    
    vec = np.random.randn(dim)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def find_most_similar(query_embedding, embeddings, top_k=5):
    """Find the most similar items to a query embedding."""
    similarities = []
    for item_id, embedding in embeddings:
        sim = cosine_similarity(query_embedding, embedding)
        similarities.append((item_id, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# ============== Fixtures ==============

@pytest.fixture
def mock_embedding_generator():
    """Create a mocked EmbeddingGenerator for unit tests."""
    generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
    generator.settings = MagicMock()
    generator.settings.google_api_key = "fake-key"
    generator.model = "gemini-embedding-001"
    generator.provider = "gemini"
    generator._cache = {}
    generator._genai = MagicMock()
    return generator


@pytest.fixture
def sample_top():
    """Create a sample top garment."""
    return Garment(
        id="white_tshirt",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            subcategory="t-shirt",
            color=ColorInfo(primary="white", hex_codes=["#FFFFFF"]),
            pattern=PatternInfo(type="solid"),
            formality_level=FormalityLevel.CASUAL,
            season_suitable=[Season.SPRING, Season.SUMMER],
            style_tags=["casual", "minimalist", "basic"],
            fit="regular"
        )
    )


@pytest.fixture
def sample_bottom():
    """Create a sample bottom garment."""
    return Garment(
        id="blue_jeans",
        attributes=GarmentAttributes(
            category=GarmentCategory.BOTTOM,
            subcategory="jeans",
            color=ColorInfo(primary="blue", hex_codes=["#1E40AF"]),
            pattern=PatternInfo(type="solid"),
            formality_level=FormalityLevel.CASUAL,
            season_suitable=[Season.SPRING, Season.FALL, Season.WINTER],
            style_tags=["casual", "classic", "denim"],
            fit="slim"
        )
    )


# ============== Test Classes ==============

@pytest.mark.unit
class TestGarmentToText:
    """Tests for garment to text conversion."""
    
    def test_garment_to_text_includes_category(
        self, mock_embedding_generator, sample_top
    ):
        """Test that text includes garment category."""
        text = mock_embedding_generator._garment_to_text(sample_top)
        
        assert "top" in text.lower() or "TOP" in text
    
    def test_garment_to_text_includes_subcategory(
        self, mock_embedding_generator, sample_top
    ):
        """Test that text includes subcategory."""
        text = mock_embedding_generator._garment_to_text(sample_top)
        
        assert "t-shirt" in text.lower()
    
    def test_garment_to_text_includes_color(
        self, mock_embedding_generator, sample_top
    ):
        """Test that text includes color."""
        text = mock_embedding_generator._garment_to_text(sample_top)
        
        assert "white" in text.lower()
    
    def test_garment_to_text_includes_pattern(
        self, mock_embedding_generator, sample_top
    ):
        """Test that text includes pattern."""
        text = mock_embedding_generator._garment_to_text(sample_top)
        
        assert "solid" in text.lower()
    
    def test_garment_to_text_includes_formality(
        self, mock_embedding_generator, sample_top
    ):
        """Test that text includes formality."""
        text = mock_embedding_generator._garment_to_text(sample_top)
        
        assert "casual" in text.lower()


@pytest.mark.unit
class TestCacheOperations:
    """Tests for embedding cache operations."""
    
    def test_cache_initially_empty(self, mock_embedding_generator):
        """Test that cache starts empty."""
        assert mock_embedding_generator.cache_size == 0
    
    def test_cache_size_increases(self, mock_embedding_generator):
        """Test that cache size increases when items added."""
        mock_embedding_generator._cache["test_key"] = [0.1] * 10
        
        assert mock_embedding_generator.cache_size == 1
    
    def test_cache_clear(self, mock_embedding_generator):
        """Test clearing the cache."""
        mock_embedding_generator._cache["test_key1"] = [0.1] * 10
        mock_embedding_generator._cache["test_key2"] = [0.2] * 10
        
        assert mock_embedding_generator.cache_size == 2
        
        mock_embedding_generator.clear_cache()
        
        assert mock_embedding_generator.cache_size == 0


@pytest.mark.unit
class TestCacheKeyGeneration:
    """Tests for cache key generation."""
    
    def test_different_garments_different_keys(
        self, mock_embedding_generator, sample_top, sample_bottom
    ):
        """Test that different garments have different cache keys."""
        key1 = mock_embedding_generator._get_cache_key(sample_top)
        key2 = mock_embedding_generator._get_cache_key(sample_bottom)
        
        assert key1 != key2
    
    def test_same_garment_same_key(
        self, mock_embedding_generator, sample_top
    ):
        """Test that same garment produces same cache key."""
        key1 = mock_embedding_generator._get_cache_key(sample_top)
        key2 = mock_embedding_generator._get_cache_key(sample_top)
        
        assert key1 == key2


@pytest.mark.unit
class TestSimilarityCalculations:
    """Tests for similarity calculation functions."""
    
    def test_cosine_similarity_identical_vectors(self):
        """Test that identical vectors have similarity of 1.0."""
        vec = generate_mock_embedding(seed=42)
        
        similarity = cosine_similarity(vec, vec)
        
        assert abs(similarity - 1.0) < 0.0001
    
    def test_cosine_similarity_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity of 0.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        
        similarity = cosine_similarity(vec1, vec2)
        
        assert abs(similarity) < 0.0001
    
    def test_cosine_similarity_opposite_vectors(self):
        """Test that opposite vectors have similarity of -1.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        
        similarity = cosine_similarity(vec1, vec2)
        
        assert abs(similarity + 1.0) < 0.0001
    
    def test_cosine_similarity_similar_vectors(self):
        """Test that similar vectors have high similarity."""
        vec1 = [1.0, 0.9, 0.8, 0.7]
        vec2 = [1.0, 0.85, 0.78, 0.72]
        
        similarity = cosine_similarity(vec1, vec2)
        
        assert similarity > 0.99


@pytest.mark.unit
class TestFindMostSimilar:
    """Tests for finding most similar items."""
    
    def test_find_most_similar_returns_correct_order(self):
        """Test that most similar is returned first."""
        query = [1.0, 0.0, 0.0, 0.0]
        embeddings = [
            ("item_a", [0.9, 0.1, 0.0, 0.0]),  # Very similar
            ("item_b", [0.0, 1.0, 0.0, 0.0]),  # Orthogonal
            ("item_c", [0.8, 0.2, 0.1, 0.0]),  # Similar
            ("item_d", [-0.9, 0.1, 0.0, 0.0]), # Opposite
        ]
        
        results = find_most_similar(query, embeddings, top_k=2)
        
        assert results[0][0] == "item_a"
        assert results[1][0] == "item_c"
    
    def test_find_most_similar_respects_top_k(self):
        """Test that top_k limits results."""
        query = [1.0, 0.0, 0.0]
        embeddings = [
            ("item_1", [1.0, 0.0, 0.0]),
            ("item_2", [0.9, 0.1, 0.0]),
            ("item_3", [0.8, 0.2, 0.0]),
            ("item_4", [0.7, 0.3, 0.0]),
        ]
        
        results = find_most_similar(query, embeddings, top_k=2)
        
        assert len(results) == 2
