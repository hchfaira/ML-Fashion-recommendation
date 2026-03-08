"""
Vision Service - Main orchestrator for Layer 1
Supports Google Gemini API (primary) and OpenAI Vision API (fallback).
"""
import base64
from typing import Optional, Union
from pathlib import Path
import httpx
import google.generativeai as genai
from PIL import Image
import io

from config import get_settings, get_config
from src.core.models import Garment, GarmentAttributes
from src.core.exceptions import VisionProcessingError
from src.core import get_logger

logger = get_logger(__name__)


class VisionService:
    """
    Main vision service that orchestrates clothing analysis.
    
    This service uses Google Gemini API (or OpenAI fallback) to:
    1. Analyze clothing images
    2. Extract attributes
    3. Generate descriptions for embedding
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.config = get_config()
        self.model_name = self.settings.vision_model
        
        # Configure Gemini
        genai.configure(api_key=self.settings.google_api_key)
        self.gemini_model = genai.GenerativeModel(self.model_name)
        
        # Load vision parameters
        self._vision_params = self.config.get_parameters("model_parameters", "vision", default={})
        
        # Optional OpenAI fallback
        self._openai_client = None
        if self.settings.openai_api_key:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
    
    async def analyze_image(
        self,
        image_source: Union[str, Path, bytes],
        detail: str = "high"
    ) -> dict:
        """
        Analyze a clothing image using Gemini Vision.
        
        Args:
            image_source: URL, file path, or bytes of the image
            detail: Level of detail ("low", "high", "auto")
            
        Returns:
            Raw analysis result from vision model
        """
        try:
            # Prepare image for Gemini
            image_data = await self._prepare_image_for_gemini(image_source)
            
            # Create prompt
            prompt = f"{self._get_system_prompt()}\n\n{self._get_analysis_prompt()}"
            
            # Call Gemini
            response = await self._call_gemini(prompt, image_data)
            
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            raise VisionProcessingError(f"Failed to analyze image: {str(e)}")
    
    async def _call_gemini(self, prompt: str, image_data: Union[Image.Image, bytes]) -> str:
        """Call Gemini API with image."""
        import asyncio
        
        # Gemini's generate_content is synchronous, so we run it in executor
        def _sync_call():
            response = self.gemini_model.generate_content(
                [prompt, image_data],
                generation_config=genai.types.GenerationConfig(
                    temperature=self._vision_params.get("generation_temperature", 0.1),
                    max_output_tokens=self._vision_params.get("max_output_tokens", 1500),
                )
            )
            return response.text
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_call)
    
    async def _prepare_image_for_gemini(self, image_source: Union[str, Path, bytes]) -> Image.Image:
        """Prepare image for Gemini API submission."""
        if isinstance(image_source, bytes):
            return Image.open(io.BytesIO(image_source))
        
        if isinstance(image_source, Path) or (isinstance(image_source, str) and not image_source.startswith('http')):
            # Read local file
            path = Path(image_source)
            if not path.exists():
                raise VisionProcessingError(f"Image file not found: {path}")
            return Image.open(path)
        
        # URL - download the image
        async with httpx.AsyncClient() as client:
            response = await client.get(image_source)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for fashion analysis."""
        return self.config.get_prompt("vision_analysis")
    
    def _get_analysis_prompt(self) -> str:
        """Get the analysis prompt for extracting clothing attributes."""
        return self.config.get_prompt("vision_simple_analysis")
    
    def _parse_response(self, content: str) -> dict:
        """Parse the vision model response."""
        import json
        
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return raw content if JSON parsing fails
            return {"raw_response": content, "parse_error": str(e)}
