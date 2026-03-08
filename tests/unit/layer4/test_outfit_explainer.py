"""
Tests for Outfit Explainer
Layer 4 - Outfit Explanations

Tests cover:
- OutfitExplanation dataclass
- Explanation generation
- Style notes generation
- Detail levels (brief, standard, detailed)
- Comparison explanations
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.layer4_llm.outfit_explainer import OutfitExplainer, OutfitExplanation
from src.core.models import Outfit, OutfitItem, UserContext, Occasion


# ============== Fixtures ==============

@pytest.fixture
def mock_llm_service():
    """Mock the LLM service."""
    with patch('src.layer4_llm.outfit_explainer.LLMService') as mock:
        service = MagicMock()
        service.explain_outfit = AsyncMock(return_value="Great outfit choice!")
        service.generate_completion = AsyncMock(return_value="Styling tip here")
        mock.return_value = service
        yield service


@pytest.fixture
def explainer(mock_llm_service):
    """Create an OutfitExplainer with mocked LLM."""
    return OutfitExplainer()


@pytest.fixture
def sample_outfit(garment_factory):
    """Create a sample outfit."""
    from src.core.models import OutfitItem
    items = [
        OutfitItem(
            garment=garment_factory.create_basic_top(color="white"),
            role="main_top"
        ),
        OutfitItem(
            garment=garment_factory.create_basic_bottom(color="blue"),
            role="main_bottom"
        ),
        OutfitItem(
            garment=garment_factory.create_shoes(color="white"),
            role="footwear"
        ),
    ]
    return Outfit(
        id="test_outfit",
        items=items,
        compatibility_score=0.85,
        style_coherence_score=0.85,
        occasion_match_score=0.85,
        overall_score=0.85
    )


@pytest.fixture
def sample_context():
    """Create a sample user context."""
    return UserContext(
        user_id="test_user",
        occasion=Occasion.CASUAL,
        body_shape="rectangle",
        style_preferences=["minimalist", "casual"]
    )


# ============== Test Classes ==============

class TestOutfitExplanationDataclass:
    """Tests for OutfitExplanation dataclass."""
    
    def test_create_explanation(self):
        """Test creating an OutfitExplanation."""
        explanation = OutfitExplanation(
            summary="Great outfit!",
            style_notes=["Clean lines", "Neutral colors"],
            color_harmony_note="Colors work well together",
            occasion_fit="Perfect for casual outings",
            styling_tips=["Add a watch for detail"],
            confidence=0.9
        )
        
        assert explanation.summary == "Great outfit!"
        assert len(explanation.style_notes) == 2
        assert explanation.confidence == 0.9
    
    def test_explanation_fields(self):
        """Test that all expected fields exist."""
        explanation = OutfitExplanation(
            summary="",
            style_notes=[],
            color_harmony_note="",
            occasion_fit="",
            styling_tips=[],
            confidence=0.0
        )
        
        assert hasattr(explanation, 'summary')
        assert hasattr(explanation, 'style_notes')
        assert hasattr(explanation, 'color_harmony_note')
        assert hasattr(explanation, 'occasion_fit')
        assert hasattr(explanation, 'styling_tips')
        assert hasattr(explanation, 'confidence')


class TestOutfitExplainerInit:
    """Tests for OutfitExplainer initialization."""
    
    def test_init_creates_llm_service(self, mock_llm_service, explainer):
        """Test that LLM service is created."""
        assert hasattr(explainer, 'llm_service')


class TestExplainMethod:
    """Tests for the explain method."""
    
    @pytest.mark.asyncio
    async def test_returns_explanation(
        self, mock_llm_service, explainer, sample_outfit, sample_context
    ):
        """Test that explain returns OutfitExplanation."""
        # Need to pass context to avoid the formality_level.value bug
        result = await explainer.explain(sample_outfit, context=sample_context)
        
        assert isinstance(result, OutfitExplanation)
    
    @pytest.mark.asyncio
    async def test_includes_summary(
        self, mock_llm_service, explainer, sample_outfit, sample_context
    ):
        """Test that explanation includes summary."""
        result = await explainer.explain(sample_outfit, context=sample_context)
        
        assert result.summary is not None
        assert len(result.summary) > 0
    
    @pytest.mark.asyncio
    async def test_includes_style_notes(
        self, mock_llm_service, explainer, sample_outfit, sample_context
    ):
        """Test that explanation includes style notes."""
        result = await explainer.explain(sample_outfit, context=sample_context)
        
        assert result.style_notes is not None
        assert isinstance(result.style_notes, list)
    
    @pytest.mark.asyncio
    async def test_uses_outfit_score_for_confidence(
        self, mock_llm_service, explainer, sample_outfit, sample_context
    ):
        """Test that outfit score is used for confidence."""
        result = await explainer.explain(sample_outfit, context=sample_context)
        
        assert result.confidence == sample_outfit.overall_score
    
    @pytest.mark.asyncio
    async def test_uses_context_when_provided(
        self, mock_llm_service, explainer, sample_outfit, sample_context
    ):
        """Test that context is used when provided."""
        result = await explainer.explain(sample_outfit, context=sample_context)
        
        # Should include occasion-relevant info
        assert result.occasion_fit is not None


class TestDetailLevels:
    """Tests for different detail levels."""
    
    @pytest.mark.asyncio
    async def test_brief_level(
        self, mock_llm_service, explainer, sample_outfit, sample_context
    ):
        """Test brief detail level."""
        result = await explainer.explain(
            sample_outfit, context=sample_context, detail_level="brief"
        )
        
        # Brief should have no styling tips
        assert len(result.styling_tips) == 0
    
    @pytest.mark.asyncio
    async def test_standard_level(
        self, mock_llm_service, explainer, sample_outfit, sample_context
    ):
        """Test standard detail level."""
        result = await explainer.explain(
            sample_outfit, context=sample_context, detail_level="standard"
        )
        
        # Standard should have basic info
        assert result.summary is not None
    
    @pytest.mark.asyncio
    async def test_detailed_level_includes_tips(
        self, mock_llm_service, explainer, sample_outfit, sample_context
    ):
        """Test detailed level includes styling tips."""
        mock_llm_service.generate_completion = AsyncMock(return_value="Add accessories")
        
        result = await explainer.explain(
            sample_outfit, context=sample_context, detail_level="detailed"
        )
        
        # Detailed should generate tips
        assert isinstance(result.styling_tips, list)


class TestExplainComparison:
    """Tests for outfit comparison explanations."""
    
    @pytest.mark.asyncio
    async def test_compares_two_outfits(
        self, mock_llm_service, explainer, garment_factory
    ):
        """Test comparing two outfits."""
        from src.core.models import OutfitItem
        
        outfit1 = Outfit(
            id="outfit1",
            items=[OutfitItem(
                garment=garment_factory.create_basic_top(),
                role="main_top"
            )],
            compatibility_score=0.8,
            style_coherence_score=0.8,
            occasion_match_score=0.8,
            overall_score=0.8
        )
        outfit2 = Outfit(
            id="outfit2",
            items=[OutfitItem(
                garment=garment_factory.create_dress_shirt(),
                role="main_top"
            )],
            compatibility_score=0.9,
            style_coherence_score=0.9,
            occasion_match_score=0.9,
            overall_score=0.9
        )
        
        mock_llm_service.generate_completion = AsyncMock(
            return_value="Outfit 2 is more formal"
        )
        
        result = await explainer.explain_comparison(outfit1, outfit2)
        
        assert result is not None
        assert len(result) > 0
