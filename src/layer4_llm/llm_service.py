"""
LLM Service - Layer 4 Main Interface
Handles all LLM interactions for explanations and UX.

Note: The LLM does NOT make style intelligence decisions.
It humanizes and explains the recommendations from Layers 1-3.
"""
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI

from config import get_settings, get_config
from src.core.models import Outfit, UserContext
from src.core.exceptions import LLMError
from src.core import get_logger

logger = get_logger(__name__)


class LLMService:
    """
    LLM service for generating human-friendly explanations.
    
    The LLM's role is to:
    - Explain outfit choices
    - Adapt communication tone
    - Make recommendations feel personal
    
    The LLM does NOT:
    - Make style decisions
    - Score compatibility
    - Choose outfits
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.config = get_config()

        # Prefer OpenAI when the key is set; fall back to Gemini's OpenAI-compat endpoint
        if self.settings.openai_api_key:
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            self.model = self.settings.llm_model
        else:
            # Gemini OpenAI-compatible endpoint
            self.client = AsyncOpenAI(
                api_key=self.settings.google_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            # Use a Gemini model name — settings.llm_model is already "gemini-2.5-flash"
            self.model = self.settings.llm_model
        
        # Load tone presets from configuration
        self.tone_presets = self.config.get_parameters(
            "conversation_settings", 
            "tone_presets",
            default={
                "professional": "professional, concise, and informative",
                "friendly": "warm, friendly, and encouraging",
                "luxury": "sophisticated, refined, and aspirational",
                "casual": "relaxed, fun, and conversational",
                "expert": "knowledgeable, detailed, and authoritative"
            }
        )
        
        # Load LLM parameters
        self._llm_params = self.config.get_parameters("model_parameters", "llm", default={})
    
    def _get_param(self, key: str, default: Any) -> Any:
        """Get a parameter from LLM config with fallback."""
        return self._llm_params.get(key, default)
    
    async def generate_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a completion from the LLM.
        
        Args:
            prompt: User/assistant prompt
            system_message: Optional system context
            max_tokens: Maximum response length (defaults from config)
            temperature: Creativity level (defaults from config)
            
        Returns:
            Generated text
        """
        # Use config defaults if not specified
        if max_tokens is None:
            max_tokens = self._get_param("default_max_tokens", 500)
        if temperature is None:
            temperature = self._get_param("default_temperature", 0.7)
        
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise LLMError(f"Failed to generate completion: {str(e)}")
    
    async def explain_outfit(
        self,
        outfit: Outfit,
        context: Optional[UserContext] = None,
        tone: str = "friendly"
    ) -> str:
        """
        Generate a human-friendly explanation of an outfit.
        
        Args:
            outfit: The outfit to explain
            context: User context for personalization
            tone: Communication tone
            
        Returns:
            Natural language explanation
        """
        # Build outfit description from data
        outfit_desc = self._build_outfit_description(outfit)
        scores = self._build_score_summary(outfit)
        context_notes = self._build_context_notes(context) if context else ""
        
        tone_desc = self.tone_presets.get(tone, self.tone_presets["friendly"])
        
        # Load prompts from configuration
        system = self.config.get_prompt(
            "llm_explain_outfit_system",
            tone_description=tone_desc
        )
        
        prompt = self.config.get_prompt(
            "llm_explain_outfit_user",
            outfit_description=outfit_desc,
            scores=scores,
            context_notes=context_notes
        )
        
        return await self.generate_completion(
            prompt=prompt,
            system_message=system,
            max_tokens=self._get_param("explanation_max_tokens", 200),
            temperature=self._get_param("default_temperature", 0.7)
        )
    
    async def generate_styling_tip(
        self,
        outfit: Outfit,
        improvement_area: str,
        tone: str = "friendly"
    ) -> str:
        """
        Generate a styling tip for improving an outfit.
        
        Args:
            outfit: Current outfit
            improvement_area: Area to improve (color, formality, etc.)
            tone: Communication tone
            
        Returns:
            Styling tip text
        """
        tone_desc = self.tone_presets.get(tone, self.tone_presets["friendly"])
        
        prompt = self.config.get_prompt(
            "llm_styling_tip",
            tone_description=tone_desc,
            improvement_area=improvement_area,
            outfit_items=self._list_outfit_items(outfit)
        )
        
        return await self.generate_completion(
            prompt=prompt,
            max_tokens=self._get_param("tip_max_tokens", 100),
            temperature=self._get_param("temperatures", {}).get("creative", 0.8)
        )
    
    async def personalize_message(
        self,
        message: str,
        user_preferences: Dict,
        tone: str = "friendly"
    ) -> str:
        """
        Personalize a message based on user preferences.
        
        Args:
            message: Base message to personalize
            user_preferences: User's known preferences
            tone: Communication tone
            
        Returns:
            Personalized message
        """
        tone_desc = self.tone_presets.get(tone, self.tone_presets["friendly"])
        
        prompt = self.config.get_prompt(
            "llm_personalize_message",
            message=message,
            user_preferences=user_preferences,
            tone_description=tone_desc
        )
        
        return await self.generate_completion(
            prompt=prompt,
            max_tokens=self._get_param("personalization_max_tokens", 150),
            temperature=self._get_param("default_temperature", 0.7)
        )
    
    def _build_outfit_description(self, outfit: Outfit) -> str:
        """Build text description of outfit items."""
        lines = ["Outfit items:"]
        
        for item in outfit.items:
            garment = item.garment
            attrs = garment.attributes
            
            desc = f"- {attrs.category.value.title()}: "
            if attrs.subcategory:
                desc += f"{attrs.subcategory}, "
            desc += f"{attrs.color.primary}"
            if attrs.pattern.type != "solid":
                desc += f" {attrs.pattern.type}"
            if attrs.silhouette:
                desc += f", {attrs.silhouette} fit"
            
            lines.append(desc)
        
        return "\n".join(lines)
    
    def _build_score_summary(self, outfit: Outfit) -> str:
        """Build summary of outfit scores."""
        return f"""- Overall match: {outfit.overall_score:.0%}
- Style coherence: {outfit.style_coherence_score:.0%}
- Compatibility: {outfit.compatibility_score:.0%}
- Occasion appropriateness: {outfit.occasion_match_score:.0%}"""
    
    def _build_context_notes(self, context: UserContext) -> str:
        """Build context notes for the prompt."""
        notes = ["Context:"]
        
        if context.occasion:
            notes.append(f"- Occasion: {context.occasion.value}")
        if context.weather:
            notes.append(f"- Weather: {context.weather.temperature_celsius}°C, {context.weather.condition}")
        if context.location:
            notes.append(f"- Location: {context.location}")
        
        return "\n".join(notes) if len(notes) > 1 else ""
    
    def _list_outfit_items(self, outfit: Outfit) -> str:
        """Simple list of outfit items."""
        items = []
        for item in outfit.items:
            attrs = item.garment.attributes
            items.append(f"- {attrs.color.primary} {attrs.subcategory or attrs.category.value}")
        return "\n".join(items)
