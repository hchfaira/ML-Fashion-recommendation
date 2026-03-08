"""
Conversation Handler
Manages conversational interactions for styling advice.
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from config import get_config
from src.core.models import Outfit, UserContext
from src.core import get_logger
from .llm_service import LLMService

logger = get_logger(__name__)


@dataclass
class ConversationMessage:
    """A message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for ongoing conversation."""
    user_id: Optional[str] = None
    current_outfit: Optional[Outfit] = None
    user_context: Optional[UserContext] = None
    style_preferences_mentioned: List[str] = field(default_factory=list)
    questions_asked: List[str] = field(default_factory=list)


class ConversationHandler:
    """
    Handles conversational styling interactions.
    
    Manages multi-turn conversations about outfits,
    style advice, and recommendations.
    """
    
    def __init__(self):
        self.llm_service = LLMService()
        self.config = get_config()
        self.conversations: Dict[str, List[ConversationMessage]] = {}
        self.contexts: Dict[str, ConversationContext] = {}
        
        # Load conversation settings from configuration
        conv_settings = self.config.get_parameters("conversation_settings", default={})
        self.intent_patterns = conv_settings.get("intent_patterns", {
            "get_recommendation": ["recommend", "suggest", "what should i wear", "outfit for"],
            "explain": ["why", "explain", "how does", "tell me about"],
            "compare": ["which is better", "compare", "difference between"],
            "improve": ["how can i improve", "what's wrong", "better option"],
            "preference": ["i like", "i prefer", "i don't like", "avoid"],
            "occasion": ["for work", "for a date", "going to", "event"]
        })
        self.intent_system_additions = conv_settings.get("intent_system_additions", {})
        self._default_history_length = conv_settings.get("default_conversation_history_length", 10)
        
        # Load LLM parameters
        self._llm_params = self.config.get_parameters("model_parameters", "llm", default={})
    
    async def handle_message(
        self,
        session_id: str,
        user_message: str,
        context: Optional[ConversationContext] = None
    ) -> str:
        """
        Handle a user message in conversation.
        
        Args:
            session_id: Conversation session ID
            user_message: User's message
            context: Optional conversation context
            
        Returns:
            Assistant's response
        """
        # Initialize or get conversation
        if session_id not in self.conversations:
            self.conversations[session_id] = []
            self.contexts[session_id] = context or ConversationContext()
        
        # Add user message
        self.conversations[session_id].append(
            ConversationMessage(role="user", content=user_message)
        )
        
        # Detect intent
        intent = self._detect_intent(user_message)
        
        # Generate response based on intent
        response = await self._generate_response(
            session_id=session_id,
            user_message=user_message,
            intent=intent
        )
        
        # Add assistant response
        self.conversations[session_id].append(
            ConversationMessage(
                role="assistant",
                content=response,
                metadata={"intent": intent}
            )
        )
        
        return response
    
    async def _generate_response(
        self,
        session_id: str,
        user_message: str,
        intent: str
    ) -> str:
        """Generate response based on intent and context."""
        context = self.contexts.get(session_id, ConversationContext())
        history = self._get_conversation_history(session_id)
        
        system_prompt = self._get_system_prompt(intent, context)
        
        # Load prompt template from configuration
        prompt = self.config.get_prompt(
            "conversation_user",
            history=history,
            user_message=user_message
        )
        
        return await self.llm_service.generate_completion(
            prompt=prompt,
            system_message=system_prompt,
            max_tokens=self._llm_params.get("conversation_max_tokens", 300),
            temperature=self._llm_params.get("default_temperature", 0.7)
        )
    
    async def get_clarifying_question(
        self,
        session_id: str,
        missing_info: str
    ) -> str:
        """
        Generate a natural clarifying question.
        
        Args:
            session_id: Session ID
            missing_info: What information is needed
            
        Returns:
            Natural question to ask user
        """
        prompt = self.config.get_prompt(
            "conversation_clarifying",
            missing_info=missing_info
        )
        
        return await self.llm_service.generate_completion(
            prompt=prompt,
            max_tokens=self._llm_params.get("clarifying_max_tokens", 50),
            temperature=self._llm_params.get("temperatures", {}).get("creative", 0.8)
        )
    
    async def summarize_preferences(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Summarize preferences learned from conversation.
        
        Args:
            session_id: Session ID
            
        Returns:
            Extracted preferences
        """
        if session_id not in self.conversations:
            return {}
        
        history = self._get_conversation_history(session_id, max_messages=20)
        
        prompt = self.config.get_prompt(
            "conversation_extract_preferences",
            history=history
        )
        
        response = await self.llm_service.generate_completion(
            prompt=prompt,
            max_tokens=self._llm_params.get("preference_extraction_max_tokens", 300),
            temperature=self._llm_params.get("temperatures", {}).get("precise", 0.3)
        )
        
        # Parse JSON response
        try:
            import json
            # Clean up response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            return json.loads(response)
        except:
            return {}
    
    def update_context(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update conversation context."""
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext()
        
        context = self.contexts[session_id]
        
        if "current_outfit" in updates:
            context.current_outfit = updates["current_outfit"]
        if "user_context" in updates:
            context.user_context = updates["user_context"]
        if "style_preference" in updates:
            context.style_preferences_mentioned.append(updates["style_preference"])
    
    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history."""
        if session_id in self.conversations:
            del self.conversations[session_id]
        if session_id in self.contexts:
            del self.contexts[session_id]
    
    def _detect_intent(self, message: str) -> str:
        """Detect user intent from message."""
        message_lower = message.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in message_lower:
                    return intent
        
        return "general"
    
    def _get_conversation_history(
        self,
        session_id: str,
        max_messages: Optional[int] = None
    ) -> str:
        """Get formatted conversation history."""
        if session_id not in self.conversations:
            return ""
        
        if max_messages is None:
            max_messages = self._default_history_length
        
        messages = self.conversations[session_id][-max_messages:]
        
        formatted = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted)
    
    def _get_system_prompt(
        self,
        intent: str,
        context: ConversationContext
    ) -> str:
        """Get system prompt based on intent and context."""
        # Load base system prompt from configuration
        base = self.config.get_prompt("conversation_system")
        
        # Get intent addition from config
        intent_addition = self.intent_system_additions.get(intent, "")
        
        prompt = base + intent_addition
        
        # Add context if available
        if context.current_outfit:
            prompt += f"\nCurrently discussing an outfit with {len(context.current_outfit.items)} items."
        
        if context.style_preferences_mentioned:
            prompt += f"\nUser has mentioned liking: {', '.join(context.style_preferences_mentioned)}"
        
        return prompt
