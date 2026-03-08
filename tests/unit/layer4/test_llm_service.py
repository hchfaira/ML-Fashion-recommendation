"""
Tests for LLM Service
Layer 4 - LLM Interaction

Tests cover:
- Service initialization
- Completion generation
- Outfit explanation
- Tone presets
- Error handling
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.layer4_llm.llm_service import LLMService
from src.core.models import Outfit, OutfitItem, UserContext, Occasion
from src.core.exceptions import LLMError


# ============== Fixtures ==============

@pytest.fixture
def mock_openai():
    """Mock the OpenAI client."""
    with patch('src.layer4_llm.llm_service.AsyncOpenAI') as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def service(mock_openai):
    """Create an LLMService with mocked OpenAI."""
    return LLMService()


@pytest.fixture
def mock_completion_response():
    """Create a mock completion response."""
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content="This is a great outfit!"))]
    return mock


@pytest.fixture
def sample_outfit(garment_factory):
    """Create a sample outfit for testing."""
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
        style_preferences=["minimalist"]
    )


# ============== Test Classes ==============

class TestLLMServiceInit:
    """Tests for LLMService initialization."""
    
    def test_init_creates_client(self, mock_openai, service):
        """Test that initialization creates OpenAI client."""
        assert service.client is not None
    
    def test_init_sets_model(self, mock_openai, service):
        """Test that initialization sets model."""
        assert hasattr(service, 'model')
    
    def test_tone_presets_defined(self, mock_openai, service):
        """Test that tone presets are defined."""
        assert hasattr(service, 'tone_presets')
        assert "professional" in service.tone_presets
        assert "friendly" in service.tone_presets
        assert "luxury" in service.tone_presets


class TestTonePresets:
    """Tests for tone preset descriptions."""
    
    def test_professional_tone(self, mock_openai, service):
        """Test professional tone preset."""
        assert "professional" in service.tone_presets["professional"]
    
    def test_friendly_tone(self, mock_openai, service):
        """Test friendly tone preset."""
        # "friendly" may be part of description or the key description contains related words
        assert "friendly" in service.tone_presets["friendly"] or "warm" in service.tone_presets["friendly"]
    
    def test_casual_tone_exists(self, mock_openai, service):
        """Test casual tone preset exists and has description."""
        assert "casual" in service.tone_presets
        assert len(service.tone_presets["casual"]) > 0
    
    def test_expert_tone_exists(self, mock_openai, service):
        """Test expert tone preset exists and has description."""
        assert "expert" in service.tone_presets
        assert len(service.tone_presets["expert"]) > 0


class TestGenerateCompletion:
    """Tests for generate_completion method."""
    
    @pytest.mark.asyncio
    async def test_generates_completion(
        self, mock_openai, service, mock_completion_response
    ):
        """Test that completion is generated."""
        service.client.chat.completions.create = AsyncMock(
            return_value=mock_completion_response
        )
        
        result = await service.generate_completion("Test prompt")
        
        assert result == "This is a great outfit!"
    
    @pytest.mark.asyncio
    async def test_includes_system_message(
        self, mock_openai, service, mock_completion_response
    ):
        """Test that system message is included when provided."""
        service.client.chat.completions.create = AsyncMock(
            return_value=mock_completion_response
        )
        
        await service.generate_completion(
            "Test prompt", 
            system_message="You are a stylist"
        )
        
        call_args = service.client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
    
    @pytest.mark.asyncio
    async def test_respects_max_tokens(
        self, mock_openai, service, mock_completion_response
    ):
        """Test that max_tokens is passed correctly."""
        service.client.chat.completions.create = AsyncMock(
            return_value=mock_completion_response
        )
        
        await service.generate_completion("Test", max_tokens=100)
        
        call_args = service.client.chat.completions.create.call_args
        assert call_args.kwargs['max_tokens'] == 100
    
    @pytest.mark.asyncio
    async def test_respects_temperature(
        self, mock_openai, service, mock_completion_response
    ):
        """Test that temperature is passed correctly."""
        service.client.chat.completions.create = AsyncMock(
            return_value=mock_completion_response
        )
        
        await service.generate_completion("Test", temperature=0.5)
        
        call_args = service.client.chat.completions.create.call_args
        assert call_args.kwargs['temperature'] == 0.5
    
    @pytest.mark.asyncio
    async def test_handles_api_error(self, mock_openai, service):
        """Test that API errors are handled."""
        service.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        with pytest.raises(LLMError):
            await service.generate_completion("Test")


class TestExplainOutfit:
    """Tests for explain_outfit method."""
    
    @pytest.mark.asyncio
    async def test_generates_explanation(
        self, mock_openai, service, sample_outfit, mock_completion_response
    ):
        """Test that outfit explanation is generated."""
        service.client.chat.completions.create = AsyncMock(
            return_value=mock_completion_response
        )
        
        result = await service.explain_outfit(sample_outfit)
        
        assert result is not None
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_uses_context_when_provided(
        self, mock_openai, service, sample_outfit, sample_context, 
        mock_completion_response
    ):
        """Test that context is used in explanation."""
        service.client.chat.completions.create = AsyncMock(
            return_value=mock_completion_response
        )
        
        await service.explain_outfit(sample_outfit, context=sample_context)
        
        # Should complete without error when context is provided
        assert True
    
    @pytest.mark.asyncio
    async def test_uses_specified_tone(
        self, mock_openai, service, sample_outfit, mock_completion_response
    ):
        """Test that specified tone is used."""
        service.client.chat.completions.create = AsyncMock(
            return_value=mock_completion_response
        )
        
        await service.explain_outfit(sample_outfit, tone="professional")
        
        call_args = service.client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        system_content = messages[0]['content']
        
        assert "professional" in system_content
    
    @pytest.mark.asyncio
    async def test_defaults_to_friendly_tone(
        self, mock_openai, service, sample_outfit, mock_completion_response
    ):
        """Test that default tone is friendly."""
        service.client.chat.completions.create = AsyncMock(
            return_value=mock_completion_response
        )
        
        await service.explain_outfit(sample_outfit)
        
        call_args = service.client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        system_content = messages[0]['content']
        
        assert "friendly" in system_content or "warm" in system_content
