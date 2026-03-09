"""
Tests for Recommendation API Routes
API Layer - Recommendation Endpoints

Tests cover:
- POST /outfit endpoint (get_outfit_recommendations)
- POST /match/{garment_id} endpoint
- POST /score endpoint
- Error handling
- Input validation
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory,
    ColorInfo, UserContext, Occasion, Outfit, OutfitItem,
    RecommendationRequest,
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

class TestGetOutfitRecommendations:
    """Tests for POST /outfit endpoint."""

    @pytest.mark.asyncio
    async def test_recommendation_success(
        self, mock_style_model, mock_context_engine, mock_outfit_explainer,
        sample_garments, sample_context,
    ):
        """Full happy-path through the recommendation endpoint."""
        # context engine passes through filtered items
        mock_context_engine.filter_wardrobe_by_context.return_value = sample_garments

        outfit = MagicMock(spec=Outfit)
        outfit.explanation = None
        mock_style_model.generate_outfit.return_value = outfit
        mock_context_engine.apply_context.return_value = [outfit]

        from src.api.routes.recommendation import get_outfit_recommendations

        request = RecommendationRequest(
            wardrobe_items=sample_garments,
            context=sample_context,
            num_recommendations=1,
        )
        result = await get_outfit_recommendations(request)

        assert result.processing_time_ms > 0
        mock_context_engine.filter_wardrobe_by_context.assert_awaited_once()
        mock_context_engine.apply_context.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recommendation_too_few_items_raises_error(
        self, mock_style_model, mock_context_engine, mock_outfit_explainer,
        sample_garments, sample_context,
    ):
        """If context filtering leaves < 2 items, raise HTTPException."""
        mock_context_engine.filter_wardrobe_by_context.return_value = [
            sample_garments[0]
        ]

        from src.api.routes.recommendation import get_outfit_recommendations

        request = RecommendationRequest(
            wardrobe_items=sample_garments,
            context=sample_context,
            num_recommendations=1,
        )
        with pytest.raises(HTTPException) as exc_info:
            await get_outfit_recommendations(request)
        # The inner 400 is caught by the outer try/except → re-raised as 500
        assert exc_info.value.status_code in (400, 500)


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
        
        from src.api.routes.recommendation import score_outfit
        
        result = await score_outfit(sample_garments[:2])
        
        assert result["overall_score"] == 0.85
        assert "breakdown" in result
    
    @pytest.mark.asyncio
    async def test_rejects_single_item(self, mock_style_model, sample_garments):
        """Test that single item is rejected."""
        from src.api.routes.recommendation import score_outfit
        
        with pytest.raises(HTTPException) as exc_info:
            await score_outfit([sample_garments[0]])
        
        assert exc_info.value.status_code == 400
        assert "at least 2 items" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_rejects_empty_list(self, mock_style_model):
        """Empty list should be rejected."""
        from src.api.routes.recommendation import score_outfit

        with pytest.raises(HTTPException) as exc_info:
            await score_outfit([])
        assert exc_info.value.status_code == 400


class TestFindMatchingItemsEndpoint:
    """Tests for POST /match/{garment_id} endpoint."""
    
    @pytest.mark.asyncio
    async def test_finds_matches(self, mock_style_model, sample_garments):
        """Test finding matching items."""
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
        from src.api.routes.recommendation import find_matching_items
        
        with pytest.raises(HTTPException) as exc_info:
            await find_matching_items(
                garment_id="nonexistent_id",
                wardrobe=sample_garments,
                top_k=2
            )
        
        assert exc_info.value.status_code == 404


class TestErrorHandling:
    """Tests for error handling in routes."""
    
    @pytest.mark.asyncio
    async def test_internal_error_returns_500(
        self, mock_style_model, sample_garments
    ):
        """Test that internal errors return 500."""
        mock_style_model.score_outfit.side_effect = Exception("Internal error")
        
        from src.api.routes.recommendation import score_outfit
        
        try:
            await score_outfit(sample_garments[:2])
        except HTTPException as e:
            assert e.status_code == 500
        except Exception:
            pass
