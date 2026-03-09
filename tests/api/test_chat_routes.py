"""
Tests for Chat API Routes
===========================

Covers:
- POST /chat/message
- POST /chat/start-session
- DELETE /chat/session/{session_id}
- GET  /chat/session/{session_id}/history
- _get_follow_up_suggestions helper
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from src.api.routes.chat import (
    send_message,
    start_chat_session,
    end_chat_session,
    get_chat_history,
    ChatMessage,
    ChatResponse,
    _get_follow_up_suggestions,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_conversation_handler():
    mock = MagicMock()
    mock.handle_message = AsyncMock(return_value="I suggest a navy blazer!")
    mock.contexts = {}
    mock.conversations = {}
    mock.summarize_preferences = AsyncMock(return_value={"color": "blue"})
    mock.clear_conversation = MagicMock()
    with patch(
        "src.api.routes.chat.get_conversation_handler", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_llm_service():
    mock = MagicMock()
    mock.generate_completion = AsyncMock(
        return_value="Hello! I'm your styling assistant."
    )
    with patch("src.api.routes.chat.get_llm_service", return_value=mock):
        yield mock


# ============================================================================
# POST /chat/message
# ============================================================================

class TestSendMessage:

    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_conversation_handler):
        msg = ChatMessage(
            session_id="sess_123",
            message="What should I wear today?",
        )
        result = await send_message(msg)
        assert isinstance(result, ChatResponse)
        assert result.session_id == "sess_123"
        assert "navy blazer" in result.response
        mock_conversation_handler.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_message_with_context(self, mock_conversation_handler):
        msg = ChatMessage(
            session_id="sess_456",
            message="Help with office outfit",
            context={"occasion": "business"},
        )
        result = await send_message(msg)
        assert result.session_id == "sess_456"

    @pytest.mark.asyncio
    async def test_send_message_returns_suggestions(
        self, mock_conversation_handler
    ):
        msg = ChatMessage(session_id="s1", message="office outfit")
        result = await send_message(msg)
        assert result.suggestions is not None
        assert isinstance(result.suggestions, list)

    @pytest.mark.asyncio
    async def test_send_message_error_raises_500(
        self, mock_conversation_handler
    ):
        mock_conversation_handler.handle_message = AsyncMock(
            side_effect=RuntimeError("LLM down")
        )
        msg = ChatMessage(session_id="s1", message="hello")
        with pytest.raises(HTTPException) as exc_info:
            await send_message(msg)
        assert exc_info.value.status_code == 500


# ============================================================================
# POST /chat/start-session
# ============================================================================

class TestStartSession:

    @pytest.mark.asyncio
    async def test_start_session(self, mock_llm_service):
        result = await start_chat_session(user_id="user_1")
        assert "session_id" in result
        assert result["session_id"].startswith("session_")
        assert "greeting" in result
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0


# ============================================================================
# DELETE /chat/session/{session_id}
# ============================================================================

class TestEndSession:

    @pytest.mark.asyncio
    async def test_end_session(self, mock_conversation_handler):
        result = await end_chat_session(session_id="sess_xyz")
        assert result["status"] == "session_ended"
        assert "learned_preferences" in result
        mock_conversation_handler.clear_conversation.assert_called_once_with(
            "sess_xyz"
        )


# ============================================================================
# GET /chat/session/{session_id}/history
# ============================================================================

class TestGetChatHistory:

    @pytest.mark.asyncio
    async def test_get_history_existing_session(
        self, mock_conversation_handler
    ):
        fake_msg = MagicMock()
        fake_msg.role = "user"
        fake_msg.content = "help me"
        mock_conversation_handler.conversations["sess_abc"] = [fake_msg]

        result = await get_chat_history(session_id="sess_abc", limit=10)
        assert result["session_id"] == "sess_abc"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_history_missing_session_raises_404(
        self, mock_conversation_handler
    ):
        with pytest.raises(HTTPException) as exc_info:
            await get_chat_history(session_id="nonexistent")
        assert exc_info.value.status_code == 404


# ============================================================================
# _get_follow_up_suggestions helper
# ============================================================================

class TestFollowUpSuggestions:

    def test_work_context(self):
        suggestions = _get_follow_up_suggestions("What about a work outfit?")
        assert any("casual" in s.lower() for s in suggestions)

    def test_office_context(self):
        suggestions = _get_follow_up_suggestions("I need something for the office")
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_date_context(self):
        suggestions = _get_follow_up_suggestions("Going on a date tonight")
        assert any("going" in s.lower() or "style" in s.lower() for s in suggestions)

    def test_color_context(self):
        suggestions = _get_follow_up_suggestions("I love the color blue")
        assert any("color" in s.lower() for s in suggestions)

    def test_generic_context(self):
        suggestions = _get_follow_up_suggestions("random message")
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
