"""
Clothing Segmentation Service
Identifies and segments individual clothing items in images.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import base64

from openai import AsyncOpenAI

from config import get_settings
from src.core.exceptions import VisionProcessingError
from src.core import get_logger

logger = get_logger(__name__)


class ClothingSegmenter:
    """
    Service for segmenting and identifying clothing items in images.
    
    Uses OpenAI Vision API to identify all visible clothing items
    and their approximate positions/regions in the image.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.vision_model
    
    async def segment_outfit(self, image_source: str) -> List[Dict[str, Any]]:
        """
        Identify all clothing items in an outfit image.
        
        Args:
            image_source: URL or path to the image
            
        Returns:
            List of identified clothing items with their details
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at identifying and cataloging 
                        individual clothing items in fashion images. Analyze the image 
                        and list each distinct clothing item visible."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self._get_segmentation_prompt()
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_source,
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            return self._parse_segmentation_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise VisionProcessingError(f"Failed to segment outfit: {str(e)}")
    
    def _get_segmentation_prompt(self) -> str:
        """Get prompt for outfit segmentation."""
        return """Analyze this fashion image and identify ALL distinct clothing items and accessories.

For each item, provide in JSON format:
{
    "items": [
        {
            "item_number": 1,
            "category": "top|bottom|dress|outerwear|shoes|accessory|bag",
            "specific_type": "e.g., blazer, sneakers, handbag",
            "position": "upper_body|lower_body|feet|head|hands|full_body",
            "visibility": "full|partial",
            "is_layered": true/false,
            "layer_order": 1 (1=closest to body),
            "brief_description": "short description"
        }
    ],
    "total_items": number,
    "outfit_type": "casual|business|formal|athletic|etc"
}

Be thorough - identify everything visible including jewelry, watches, belts, etc."""
    
    def _parse_segmentation_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse segmentation response."""
        import json
        
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            return result.get("items", [])
        except json.JSONDecodeError:
            logger.warning("Failed to parse segmentation JSON")
            return []


class OutfitDecomposer:
    """
    Decomposes a complete outfit image into individual garment analyses.
    """
    
    def __init__(self):
        self.segmenter = ClothingSegmenter()
    
    async def decompose_outfit(self, image_source: str) -> Dict[str, Any]:
        """
        Fully decompose an outfit into its components with detailed analysis.
        
        Args:
            image_source: URL or path to the outfit image
            
        Returns:
            Complete decomposition with all items and relationships
        """
        # Get individual items
        items = await self.segmenter.segment_outfit(image_source)
        
        return {
            "items": items,
            "item_count": len(items),
            "categories_present": list(set(item.get("category") for item in items)),
            "has_complete_outfit": self._check_completeness(items)
        }
    
    def _check_completeness(self, items: List[Dict]) -> bool:
        """Check if outfit has basic required items."""
        categories = {item.get("category") for item in items}
        
        # Basic outfit should have top+bottom or dress
        has_top = "top" in categories or "outerwear" in categories
        has_bottom = "bottom" in categories
        has_dress = "dress" in categories
        
        return (has_top and has_bottom) or has_dress
