"""
Chat API Routes
Conversational interface for styling advice.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.layer4_llm import ConversationHandler, LLMService
from src.core.models import UserContext, Occasion
from src.core import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Lazy-loaded service instances
_conversation_handler = None
_llm_service = None


def get_conversation_handler() -> ConversationHandler:
    """Get or create ConversationHandler instance."""
    global _conversation_handler
    if _conversation_handler is None:
        _conversation_handler = ConversationHandler()
    return _conversation_handler


def get_llm_service() -> LLMService:
    """Get or create LLMService instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


# For backward compatibility with tests
conversation_handler = None
llm_service = None


class ChatMessage(BaseModel):
    """Chat message request."""
    session_id: str
    message: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat message response."""
    session_id: str
    response: str
    suggestions: Optional[list[str]] = None


@router.post("/message", response_model=ChatResponse)
async def send_message(chat: ChatMessage):
    """
    Send a message in the styling conversation.
    
    Uses Layer 4 LLM for natural conversation about
    outfit recommendations and style advice.
    """
    try:
        conv_handler = get_conversation_handler()
        
        # Build context if provided
        conv_context = None
        if chat.context:
            conv_context = conv_handler.contexts.get(chat.session_id)
            if conv_context and chat.context.get("occasion"):
                try:
                    conv_context.user_context = UserContext(
                        occasion=Occasion(chat.context["occasion"])
                    )
                except:
                    pass
        
        # Get response
        response = await conv_handler.handle_message(
            session_id=chat.session_id,
            user_message=chat.message,
            context=conv_context
        )
        
        return ChatResponse(
            session_id=chat.session_id,
            response=response,
            suggestions=_get_follow_up_suggestions(chat.message)
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-session")
async def start_chat_session(user_id: Optional[str] = None):
    """
    Start a new chat session.
    """
    import uuid
    session_id = f"session_{uuid.uuid4().hex[:12]}"
    
    # Initialize with greeting
    llm = get_llm_service()
    greeting = await llm.generate_completion(
        prompt="Generate a brief, friendly greeting for a fashion styling assistant. One sentence only.",
        max_tokens=50
    )
    
    return {
        "session_id": session_id,
        "greeting": greeting,
        "suggestions": [
            "Help me find an outfit for work",
            "What goes well with jeans?",
            "I need something for a date night"
        ]
    }


@router.delete("/session/{session_id}")
async def end_chat_session(session_id: str):
    """
    End a chat session and optionally save preferences learned.
    """
    conv_handler = get_conversation_handler()
    
    # Extract preferences before clearing
    preferences = await conv_handler.summarize_preferences(session_id)
    
    # Clear the session
    conv_handler.clear_conversation(session_id)
    
    return {
        "status": "session_ended",
        "learned_preferences": preferences
    }


@router.get("/session/{session_id}/history")
async def get_chat_history(session_id: str, limit: int = 20):
    """
    Get chat history for a session.
    """
    conv_handler = get_conversation_handler()
    
    if session_id not in conv_handler.conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = conv_handler.conversations[session_id][-limit:]
    
    return {
        "session_id": session_id,
        "messages": [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    }


def _get_follow_up_suggestions(message: str) -> list[str]:
    """Generate contextual follow-up suggestions."""
    message_lower = message.lower()
    
    if "work" in message_lower or "office" in message_lower:
        return [
            "What about something more casual?",
            "Any color preferences?",
            "Show me business casual options"
        ]
    elif "date" in message_lower:
        return [
            "Where are you going?",
            "What's your style usually?",
            "Any colors you want to avoid?"
        ]
    elif "color" in message_lower:
        return [
            "What colors do you already have?",
            "Any patterns you like?",
            "Show me color combinations"
        ]
    
    return [
        "Tell me more about the occasion",
        "What's the weather like?",
        "Any style preferences?"
    ]
