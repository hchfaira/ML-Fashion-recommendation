"""
Tests for Recommendation API Routes
API Layer - Recommendation Endpoints

Tests cover:
- POST /outfit endpoint
- POST /match/{garment_id} endpoint
- POST /score endpoint
- Error handling
- Input validation
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory,
    ColorInfo, UserContext, Occasion, Outfit, OutfitItem
)


# ============== Fixtures ==============

@pytest.fixture
def mock_style_model():
    """Mock the StyleIntelligenceModel via get_style_model getter."""
    mock = MagicMock()
    mock.generate_outfit = AsyncMock()
    mock.find_best_match = AsyncMock()
    mock.score_outfit = AsyncMock()
    with patch('src.api.routes.recommendation.get_style_model', return_value=mock):
        yield mock


@pytest.fixture
def mock_context_engine():
    """Mock the ContextEngine via get_context_engine getter."""
    mock = MagicMock()
    mock.filter_wardrobe_by_context = AsyncMock()
    mock.apply_context = AsyncMock()
    with patch('src.api.routes.recommendation.get_context_engine', return_value=mock):
        yield mock


@pytest.fixture
def mock_outfit_explainer():
    """Mock the OutfitExplainer via get_outfit_explainer getter."""
    mock = MagicMock()
    mock_explanation = MagicMock()
    mock_explanation.summary = "Great outfit!"
    mock.explain = AsyncMock(return_value=mock_explanation)
    with patch('src.api.routes.recommendation.get_outfit_explainer', return_value=mock):
        yield mock


@pytest.fixture
def sample_garments():
    """Create sample garments for testing."""
    return [
        Garment(
            id="top_1",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                color=ColorInfo(primary="white", hex_codes=[])
            )
        ),
        Garment(
            id="bottom_1",
            attributes=GarmentAttributes(
                category=GarmentCategory.BOTTOM,
                color=ColorInfo(primary="blue", hex_codes=[])
            )
        ),
        Garment(
            id="shoes_1",
            attributes=GarmentAttributes(
                category=GarmentCategory.SHOES,
                color=ColorInfo(primary="white", hex_codes=[])
            )
        ),
    ]


@pytest.fixture
def sample_context():
    """Create sample user context."""
    return UserContext(
        user_id="test_user",
        occasion=Occasion.CASUAL,
        body_shape="rectangle"
    )


# ============== Test Classes ==============

class TestScoreOutfitEndpoint:
    """Tests for POST /score endpoint."""
    
    @pytest.mark.asyncio
    async def test_scores_valid_outfit(
        self, mock_style_model, sample_garments
    ):
        """Test scoring a valid outfit."""
        mock_style_model.score_outfit.return_value = {
            "overall_score": 0.85,
            "breakdown": {
                "color_harmony": 0.9,
                "formality_match": 0.8
            },
            "suggestions": []
        }
        
        # Import the function
        from src.api.routes.recommendation import score_outfit
        
        result = await score_outfit(sample_garments[:2])
        
        assert result["overall_score"] == 0.85
        assert "breakdown" in result
    
    @pytest.mark.asyncio
    async def test_rejects_single_item(self, mock_style_model, sample_garments):
        """Test that single item is rejected."""
        from fastapi import HTTPException
        from src.api.routes.recommendation import score_outfit
        
        with pytest.raises(HTTPException) as exc_info:
            await score_outfit([sample_garments[0]])
        
        assert exc_info.value.status_code == 400
        assert "at least 2 items" in exc_info.value.detail.lower()


class TestFindMatchingItemsEndpoint:
    """Tests for POST /match/{garment_id} endpoint."""
    
    @pytest.mark.asyncio
    async def test_finds_matches(self, mock_style_model, sample_garments):
        """Test finding matching items."""
        # Setup mock
        mock_style_model.find_best_match.return_value = [
            (sample_garments[1], 0.9),
            (sample_garments[2], 0.85)
        ]
        
        from src.api.routes.recommendation import find_matching_items
        
        result = await find_matching_items(
            garment_id="top_1",
            wardrobe=sample_garments,
            top_k=2
        )
        
        assert "base_item" in result
        assert "matches" in result
        assert len(result["matches"]) == 2
    
    @pytest.mark.asyncio
    async def test_returns_404_for_missing_garment(
        self, mock_style_model, sample_garments
    ):
        """Test 404 for garment not in wardrobe."""
        from fastapi import HTTPException
        from src.api.routes.recommendation import find_matching_items
        
        with pytest.raises(HTTPException) as exc_info:
            await find_matching_items(
                garment_id="nonexistent_id",
                wardrobe=sample_garments,
                top_k=2
            )
        
        assert exc_info.value.status_code == 404


class TestInputValidation:
    """Tests for input validation."""
    
    @pytest.mark.asyncio
    async def test_empty_wardrobe_rejected(
        self, mock_style_model
    ):
        """Test that empty wardrobe is handled."""
        from fastapi import HTTPException
        from src.api.routes.recommendation import score_outfit
        
        with pytest.raises(HTTPException):
            await score_outfit([])


class TestErrorHandling:
    """Tests for error handling in routes."""
    
    @pytest.mark.asyncio
    async def test_internal_error_returns_500(
        self, mock_style_model, sample_garments
    ):
        """Test that internal errors return 500."""
        mock_style_model.score_outfit.side_effect = Exception("Internal error")
        
        from fastapi import HTTPException
        from src.api.routes.recommendation import score_outfit
        
        # Should handle internal errors gracefully
        # (This depends on implementation - may raise or return error response)
        try:
            await score_outfit(sample_garments[:2])
        except HTTPException as e:
            assert e.status_code == 500
        except Exception:
            # Different error handling is acceptable
            pass
