"""
Attribute Extractor
Extracts detailed fashion attributes from clothing images using Google Gemini.

Supports two extraction modes:
- FULL_OUTFIT: Extracts all garments from an outfit image (default)
- SPECIFIC_GARMENT: Extracts only a specific garment type (e.g., pants, top)
"""
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from enum import Enum
import asyncio
import io

import google.generativeai as genai
from PIL import Image
import httpx

from config import get_settings, get_config
from src.core.models import (
    GarmentAttributes, GarmentCategory, ColorProfile, PatternInfo, 
    FormalityLevel, Season, MaterialProfile, NecklineType, LengthType, 
    WaistRise, ClosureType, TransparencyLevel, SleeveType, DetailTags,
    OutfitRole, ColorTemperature, ColorDepth, ContrastLevel, FabricWeight,
    StretchLevel, StructureType, VolumeLevel, TrendAlignment, StatementLevel,
    TemperatureSuitability, SilhouetteProfile, StyleIdentity, StylingCompatibility,
    OccasionProfile, SeasonalityInfo, VersatilityInfo
)
from src.core.exceptions import VisionProcessingError
from src.core import get_logger
from src.layer1_vision.color_utils import (
    FASHION_COLORS, extract_colors_from_text, extract_hex_codes,
    hex_to_color_name, normalize_color_name
)

logger = get_logger(__name__)


class ExtractionMode(str, Enum):
    """
    Mode for attribute extraction from images.
    
    FULL_OUTFIT: Extract all garments visible in the image (default)
    SPECIFIC_GARMENT: Extract only specific garment type(s)
    """
    FULL_OUTFIT = "full_outfit"
    SPECIFIC_GARMENT = "specific_garment"


# Default extraction mode
DEFAULT_EXTRACTION_MODE = ExtractionMode.FULL_OUTFIT


class AttributeExtractor:
    """
    Extracts detailed fashion attributes from clothing items using Gemini Vision.
    
    Supports two extraction modes:
    - FULL_OUTFIT: Extract all garments visible in the image (default)
    - SPECIFIC_GARMENT: Extract only specific garment type(s) using ClothingSegmenter
      Can extract a single category or multiple categories (e.g., [TOP, BOTTOM])
    
    Attributes extracted:
    - Color (primary, secondary, accent, hex codes)
    - Material and texture
    - Pattern type and scale
    - Style tags
    - Formality level
    - Seasonality
    - Silhouette and fit
    """
    
    def __init__(
        self, 
        extraction_mode: ExtractionMode = DEFAULT_EXTRACTION_MODE,
        target_categories: Optional[Union[GarmentCategory, List[GarmentCategory]]] = None
    ):
        """
        Initialize the AttributeExtractor.
        
        Args:
            extraction_mode: Mode for extraction (FULL_OUTFIT or SPECIFIC_GARMENT)
            target_categories: When mode is SPECIFIC_GARMENT, the category or list of
                             categories to extract (e.g., GarmentCategory.BOTTOM for pants,
                             or [GarmentCategory.TOP, GarmentCategory.BOTTOM] for both)
        """
        self.settings = get_settings()
        self.config = get_config()
        self.extraction_mode = extraction_mode
        
        # Normalize target_categories to always be a list or None
        if target_categories is None:
            self.target_categories = None
        elif isinstance(target_categories, GarmentCategory):
            self.target_categories = [target_categories]
        else:
            self.target_categories = list(target_categories)
        
        # Backward compatibility: single target_category property
        self.target_category = self.target_categories[0] if self.target_categories else None
        
        # Validate configuration
        if extraction_mode == ExtractionMode.SPECIFIC_GARMENT and not self.target_categories:
            raise ValueError(
                "target_categories must be specified when using SPECIFIC_GARMENT mode"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.settings.google_api_key)
        self.model = genai.GenerativeModel(self.settings.vision_model)
        
        # Load vision parameters from configuration
        self._vision_params = self.config.get_parameters("model_parameters", "vision", default={})
        
        # Lazy-load segmenter only when needed
        self._segmenter = None
    
    @property
    def segmenter(self):
        """Lazy-load the ClothingSegmenter when needed."""
        if self._segmenter is None:
            from src.layer1_vision.segmentation import ClothingSegmenter
            self._segmenter = ClothingSegmenter()
        return self._segmenter
    
    async def extract_attributes(
        self, 
        image_source: Union[str, Path, bytes],
        mode: Optional[ExtractionMode] = None,
        target_categories: Optional[Union[GarmentCategory, List[GarmentCategory]]] = None
    ) -> Union[GarmentAttributes, List[GarmentAttributes], None]:
        """
        Extract attributes from a garment image.
        
        Args:
            image_source: URL, path, or bytes of the garment image
            mode: Override the extraction mode for this call (optional)
            target_categories: Override the target category/categories for this call.
                             Can be a single GarmentCategory or a list of categories.
            
        Returns:
            - FULL_OUTFIT mode: List of GarmentAttributes for all detected garments
            - SPECIFIC_GARMENT mode with single category: Single GarmentAttributes or None
            - SPECIFIC_GARMENT mode with multiple categories: List of GarmentAttributes
              (only includes found categories)
        """
        # Use overrides or fall back to instance settings
        effective_mode = mode or self.extraction_mode
        
        # Normalize target_categories
        if target_categories is not None:
            if isinstance(target_categories, GarmentCategory):
                effective_targets = [target_categories]
            else:
                effective_targets = list(target_categories)
        else:
            effective_targets = self.target_categories
        
        # Validate SPECIFIC_GARMENT mode has targets
        if effective_mode == ExtractionMode.SPECIFIC_GARMENT and not effective_targets:
            raise ValueError(
                "target_categories must be specified when using SPECIFIC_GARMENT mode"
            )
        
        try:
            if effective_mode == ExtractionMode.FULL_OUTFIT:
                return await self._extract_full_outfit(image_source)
            else:
                return await self._extract_specific_garments(image_source, effective_targets)
                
        except Exception as e:
            logger.error(f"Attribute extraction failed: {e}")
            raise VisionProcessingError(f"Failed to extract attributes: {str(e)}")
    
    async def _extract_full_outfit(
        self, 
        image_source: Union[str, Path, bytes]
    ) -> List[GarmentAttributes]:
        """
        Extract attributes for all garments in an outfit image.
        
        Uses ClothingSegmenter to identify all items, then extracts
        attributes for each one.
        
        Args:
            image_source: URL, path, or bytes of the outfit image
            
        Returns:
            List of GarmentAttributes for all detected garments
        """
        # First, segment the outfit to identify all items
        image_path = self._get_image_path_for_segmenter(image_source)
        
        try:
            segmented_items = await self.segmenter.segment_outfit(image_path)
        except Exception as e:
            logger.warning(f"Segmentation failed, falling back to single extraction: {e}")
            # Fallback: treat as single garment
            return [await self._extract_single_garment(image_source)]
        
        if not segmented_items:
            logger.warning("No items segmented, falling back to single extraction")
            return [await self._extract_single_garment(image_source)]
        
        # Extract attributes for each identified item
        all_attributes = []
        image = await self._prepare_image(image_source)
        
        for item in segmented_items:
            category_str = item.get("category", "").lower()
            specific_type = item.get("specific_type", "")
            
            # Create targeted prompt for this specific item
            prompt = self._get_targeted_extraction_prompt(category_str, specific_type)
            
            try:
                response_text = await self._call_gemini(prompt, image)
                raw_attrs = self._parse_response(response_text)
                
                # Override category from segmentation if needed
                if "taxonomy" not in raw_attrs:
                    raw_attrs["taxonomy"] = {}
                raw_attrs["taxonomy"]["category"] = category_str
                raw_attrs["taxonomy"]["subcategory"] = specific_type
                
                garment_attrs = self._convert_to_model(raw_attrs)
                all_attributes.append(garment_attrs)
                
            except Exception as e:
                logger.warning(f"Failed to extract attributes for {specific_type}: {e}")
                continue
        
        return all_attributes if all_attributes else [await self._extract_single_garment(image_source)]
    
    async def _extract_specific_garments(
        self, 
        image_source: Union[str, Path, bytes],
        target_categories: List[GarmentCategory]
    ) -> Union[GarmentAttributes, List[GarmentAttributes], None]:
        """
        Extract attributes for specific garment type(s) only.
        
        Uses ClothingSegmenter to identify items and filters to the target categories.
        
        Args:
            image_source: URL, path, or bytes of the outfit image
            target_categories: List of garment categories to extract
            
        Returns:
            - Single category: GarmentAttributes or None if not found
            - Multiple categories: List of GarmentAttributes (only found items)
        """
        # Segment to find target items
        image_path = self._get_image_path_for_segmenter(image_source)
        
        try:
            segmented_items = await self.segmenter.segment_outfit(image_path)
        except Exception as e:
            logger.warning(f"Segmentation failed: {e}")
            # Fallback: extract and hope it matches
            result = await self._extract_single_garment(image_source)
            if result.category in target_categories:
                if len(target_categories) == 1:
                    return result
                return [result]
            return None if len(target_categories) == 1 else []
        
        # Build set of target category values for fast lookup
        target_values = {cat.value for cat in target_categories}
        
        # Find matching items from segmentation
        matching_items = []
        for item in segmented_items:
            item_category = item.get("category", "").lower()
            if item_category in target_values:
                matching_items.append(item)
        
        if not matching_items:
            categories_str = ", ".join(cat.value for cat in target_categories)
            logger.info(f"No target categories found in image: {categories_str}")
            return None if len(target_categories) == 1 else []
        
        # Extract attributes for each matching item
        image = await self._prepare_image(image_source)
        results = []
        
        for item in matching_items:
            item_category = item.get("category", "").lower()
            specific_type = item.get("specific_type", "")
            
            try:
                prompt = self._get_targeted_extraction_prompt(item_category, specific_type)
                response_text = await self._call_gemini(prompt, image)
                raw_attrs = self._parse_response(response_text)
                
                # Ensure correct category
                if "taxonomy" not in raw_attrs:
                    raw_attrs["taxonomy"] = {}
                raw_attrs["taxonomy"]["category"] = item_category
                raw_attrs["taxonomy"]["subcategory"] = specific_type
                
                garment_attrs = self._convert_to_model(raw_attrs)
                results.append(garment_attrs)
                
            except Exception as e:
                logger.warning(f"Failed to extract attributes for {specific_type}: {e}")
                continue
        
        # Return based on request type
        if len(target_categories) == 1:
            # Single category requested - return single result or None
            return results[0] if results else None
        else:
            # Multiple categories requested - return list (may be empty)
            return results
    
    async def _extract_specific_garment(
        self, 
        image_source: Union[str, Path, bytes],
        target_category: GarmentCategory
    ) -> Optional[GarmentAttributes]:
        """
        Extract attributes for a specific garment type only.
        
        Convenience wrapper around _extract_specific_garments for single category.
        
        Args:
            image_source: URL, path, or bytes of the outfit image
            target_category: The specific garment category to extract (e.g., BOTTOM for pants)
            
        Returns:
            GarmentAttributes for the target garment, or None if not found
        """
        return await self._extract_specific_garments(image_source, [target_category])
    
    async def _extract_single_garment(
        self, 
        image_source: Union[str, Path, bytes]
    ) -> GarmentAttributes:
        """
        Extract attributes from a single garment image (legacy behavior).
        
        Args:
            image_source: URL, path, or bytes of the garment image
            
        Returns:
            GarmentAttributes for the garment
        """
        image = await self._prepare_image(image_source)
        prompt = f"{self._get_system_prompt()}\n\n{self._get_extraction_prompt()}"
        response_text = await self._call_gemini(prompt, image)
        raw_attrs = self._parse_response(response_text)
        return self._convert_to_model(raw_attrs)
    
    def _get_image_path_for_segmenter(self, image_source: Union[str, Path, bytes]) -> str:
        """Convert image source to path/URL for segmenter."""
        if isinstance(image_source, bytes):
            # For bytes, we need to save temporarily or use base64
            # For now, raise an error - segmenter needs a path/URL
            raise VisionProcessingError(
                "FULL_OUTFIT mode with bytes input not fully supported. "
                "Use a file path or URL instead."
            )
        return str(image_source)
    
    def _get_targeted_extraction_prompt(self, category: str, specific_type: str) -> str:
        """Get a prompt targeted at a specific garment in the image."""
        return f"""{self._get_system_prompt()}

Focus ONLY on the {specific_type or category} visible in this image.
Ignore all other clothing items.

{self._get_extraction_prompt()}"""
    
    # ========== Convenience Methods ==========
    
    async def extract_pants(
        self, 
        image_source: Union[str, Path, bytes]
    ) -> Optional[GarmentAttributes]:
        """
        Convenience method to extract only pants/bottoms from an image.
        
        Args:
            image_source: URL, path, or bytes of the image
            
        Returns:
            GarmentAttributes for pants, or None if not found
        """
        return await self.extract_attributes(
            image_source,
            mode=ExtractionMode.SPECIFIC_GARMENT,
            target_categories=GarmentCategory.BOTTOM
        )
    
    async def extract_top(
        self, 
        image_source: Union[str, Path, bytes]
    ) -> Optional[GarmentAttributes]:
        """
        Convenience method to extract only top from an image.
        
        Args:
            image_source: URL, path, or bytes of the image
            
        Returns:
            GarmentAttributes for top, or None if not found
        """
        return await self.extract_attributes(
            image_source,
            mode=ExtractionMode.SPECIFIC_GARMENT,
            target_categories=GarmentCategory.TOP
        )
    
    async def extract_dress(
        self, 
        image_source: Union[str, Path, bytes]
    ) -> Optional[GarmentAttributes]:
        """
        Convenience method to extract only dress from an image.
        
        Args:
            image_source: URL, path, or bytes of the image
            
        Returns:
            GarmentAttributes for dress, or None if not found
        """
        return await self.extract_attributes(
            image_source,
            mode=ExtractionMode.SPECIFIC_GARMENT,
            target_categories=GarmentCategory.DRESS
        )
    
    async def extract_shoes(
        self, 
        image_source: Union[str, Path, bytes]
    ) -> Optional[GarmentAttributes]:
        """
        Convenience method to extract only shoes from an image.
        
        Args:
            image_source: URL, path, or bytes of the image
            
        Returns:
            GarmentAttributes for shoes, or None if not found
        """
        return await self.extract_attributes(
            image_source,
            mode=ExtractionMode.SPECIFIC_GARMENT,
            target_categories=GarmentCategory.SHOES
        )
    
    async def extract_outerwear(
        self, 
        image_source: Union[str, Path, bytes]
    ) -> Optional[GarmentAttributes]:
        """
        Convenience method to extract only outerwear from an image.
        
        Args:
            image_source: URL, path, or bytes of the image
            
        Returns:
            GarmentAttributes for outerwear, or None if not found
        """
        return await self.extract_attributes(
            image_source,
            mode=ExtractionMode.SPECIFIC_GARMENT,
            target_categories=GarmentCategory.OUTERWEAR
        )
    
    async def extract_accessory(
        self, 
        image_source: Union[str, Path, bytes]
    ) -> Optional[GarmentAttributes]:
        """
        Convenience method to extract only accessories from an image.
        
        Args:
            image_source: URL, path, or bytes of the image
            
        Returns:
            GarmentAttributes for accessory, or None if not found
        """
        return await self.extract_attributes(
            image_source,
            mode=ExtractionMode.SPECIFIC_GARMENT,
            target_categories=GarmentCategory.ACCESSORY
        )
    
    async def extract_multiple(
        self, 
        image_source: Union[str, Path, bytes],
        categories: List[GarmentCategory]
    ) -> List[GarmentAttributes]:
        """
        Convenience method to extract multiple specific garment types from an image.
        
        Args:
            image_source: URL, path, or bytes of the image
            categories: List of garment categories to extract
                       (e.g., [GarmentCategory.TOP, GarmentCategory.BOTTOM])
            
        Returns:
            List of GarmentAttributes for found categories (may be empty)
        """
        return await self.extract_attributes(
            image_source,
            mode=ExtractionMode.SPECIFIC_GARMENT,
            target_categories=categories
        )
    
    async def extract_top_and_bottom(
        self, 
        image_source: Union[str, Path, bytes]
    ) -> List[GarmentAttributes]:
        """
        Convenience method to extract top and bottom from an image.
        
        Args:
            image_source: URL, path, or bytes of the image
            
        Returns:
            List of GarmentAttributes for top and bottom (may be partial)
        """
        return await self.extract_multiple(
            image_source,
            [GarmentCategory.TOP, GarmentCategory.BOTTOM]
        )
    
    async def extract_full_outfit(
        self, 
        image_source: Union[str, Path, bytes]
    ) -> List[GarmentAttributes]:
        """
        Convenience method to extract all garments from an outfit image.
        
        Args:
            image_source: URL, path, or bytes of the outfit image
            
        Returns:
            List of GarmentAttributes for all detected garments
        """
        return await self.extract_attributes(
            image_source,
            mode=ExtractionMode.FULL_OUTFIT
        )

    async def _call_gemini(self, prompt: str, image: Image.Image) -> str:
        """Call Gemini API with image."""
        def _sync_call():
            response = self.model.generate_content(
                [prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2500,  # Increased from 1500 to avoid truncation
                )
            )
            response_text = response.text
            # Log first 500 chars of response for debugging color extraction
            logger.debug(f"Gemini raw response (first 500 chars): {response_text[:500]}")
            return response_text
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_call)
    
    async def _prepare_image(self, image_source: Union[str, Path, bytes]) -> Image.Image:
        """Prepare image for Gemini API."""
        if isinstance(image_source, bytes):
            return Image.open(io.BytesIO(image_source))
        
        if isinstance(image_source, Path) or (isinstance(image_source, str) and not image_source.startswith('http')):
            path = Path(image_source)
            if not path.exists():
                raise VisionProcessingError(f"Image file not found: {path}")
            return Image.open(path)
        
        # URL - download the image
        async with httpx.AsyncClient() as client:
            response = await client.get(image_source)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
    
    async def extract_color_palette(self, image_source: Union[str, Path, bytes]) -> ColorProfile:
        """Extract detailed color information from an image."""
        try:
            image = await self._prepare_image(image_source)
            
            prompt = """Analyze the colors in this clothing item and provide:
{
    "primary": "main color name",
    "secondary": "second most prominent color or null",
    "color_palette": ["list of all colors"],
    "hex_codes": ["#XXXXXX", "#YYYYYY"],
    "color_temperature": "warm|cool|neutral",
    "color_depth": "light|medium|dark",
    "contrast_level": "low|medium|high"
}
Respond only with valid JSON."""
            
            response_text = await self._call_gemini(prompt, image)
            color_data = self._parse_response(response_text)
            
            # Parse color temperature
            temp_str = color_data.get("color_temperature")
            color_temp = None
            if temp_str:
                try:
                    color_temp = ColorTemperature(temp_str.lower())
                except ValueError:
                    pass
            
            # Parse color depth
            depth_str = color_data.get("color_depth")
            color_depth = None
            if depth_str:
                try:
                    color_depth = ColorDepth(depth_str.lower())
                except ValueError:
                    pass
            
            # Parse contrast level
            contrast_str = color_data.get("contrast_level")
            contrast = None
            if contrast_str:
                try:
                    contrast = ContrastLevel(contrast_str.lower())
                except ValueError:
                    pass
            
            return ColorProfile(
                primary=color_data.get("primary", "unknown"),
                secondary=color_data.get("secondary"),
                color_palette=color_data.get("color_palette", []),
                hex_codes=color_data.get("hex_codes", []),
                color_temperature=color_temp,
                color_depth=color_depth,
                contrast_level=contrast
            )
            
        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            raise VisionProcessingError(f"Failed to extract colors: {str(e)}")
    
    def _get_system_prompt(self) -> str:
        """System prompt for attribute extraction."""
        return self.config.get_prompt("vision_system")
    
    def _get_extraction_prompt(self) -> str:
        """Prompt for comprehensive attribute extraction optimized for outfit compatibility."""
        return self.config.get_prompt("attribute_extraction")
    
    def _extract_color_from_text(self, text: str) -> Optional[str]:
        """
        Extract the primary color from text using color_utils.
        
        Args:
            text: Raw text that may contain color information
            
        Returns:
            Normalized color name or None if not found
        """
        import re
        
        # Strategy 1: Look for explicit color field values
        color_patterns = [
            r'"primary_color"\s*:\s*"([^"]+)"',
            r'"primary"\s*:\s*"([^"]+)"',
            r'"color"\s*:\s*"([^"]+)"',
            r'"main_color"\s*:\s*"([^"]+)"',
            r'"dominant_color"\s*:\s*"([^"]+)"',
            r'"base_color"\s*:\s*"([^"]+)"',
        ]
        
        for pattern in color_patterns:
            color_match = re.search(pattern, text, re.IGNORECASE)
            if color_match:
                raw_color = color_match.group(1)
                # Skip null/none values
                if raw_color.lower() not in ("null", "none", "n/a", "unknown", ""):
                    normalized = normalize_color_name(raw_color)
                    if normalized and normalized != "unknown":
                        logger.debug(f"Color extracted from pattern {pattern}: {normalized}")
                        return normalized
        
        # Strategy 2: Extract hex codes and convert
        hex_codes = extract_hex_codes(text)
        if hex_codes:
            color_name = hex_to_color_name(hex_codes[0])
            if color_name and color_name != "unknown":
                logger.debug(f"Color extracted from hex {hex_codes[0]}: {color_name}")
                return color_name
        
        # Strategy 3: Find color names in the text (using fashion color vocabulary)
        found_colors = extract_colors_from_text(text)
        if found_colors:
            logger.debug(f"Color extracted from text search: {found_colors[0]}")
            return found_colors[0]
        
        # Strategy 4: Look for color in description or notes fields
        desc_patterns = [
            r'"description"\s*:\s*"([^"]+)"',
            r'"notes"\s*:\s*"([^"]+)"',
            r'"details"\s*:\s*"([^"]+)"',
        ]
        for pattern in desc_patterns:
            desc_match = re.search(pattern, text, re.IGNORECASE)
            if desc_match:
                desc_text = desc_match.group(1)
                colors_in_desc = extract_colors_from_text(desc_text)
                if colors_in_desc:
                    logger.debug(f"Color extracted from description: {colors_in_desc[0]}")
                    return colors_in_desc[0]
        
        return None
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from vision model with robust error handling."""
        import json
        import re
        
        original_content = content  # Keep original for color extraction fallback
        
        try:
            # Extract JSON from markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            content = content.strip()
            
            # Try direct parsing first
            parsed_result = None
            try:
                parsed_result = json.loads(content)
            except json.JSONDecodeError:
                pass
            
            # If direct parsing failed, try to fix common JSON issues
            if parsed_result is None:
                # Fix common JSON issues from LLM responses
                # 1. Remove trailing commas before } or ]
                content = re.sub(r',\s*([}\]])', r'\1', content)
                
                # 2. Fix unquoted property names
                content = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', content)
                
                # 3. Fix single quotes to double quotes
                content = content.replace("'", '"')
                
                # 4. Remove newlines inside strings
                content = re.sub(r'(?<!\\)\n(?=[^"]*"[^"]*$)', ' ', content)
                
                # 5. Fix truncated strings - add missing closing quote
                lines = content.split('\n')
                fixed_lines = []
                for line in lines:
                    # Count quotes - if odd, add closing quote
                    quote_count = line.count('"') - line.count('\\"')
                    if quote_count % 2 == 1:
                        line = line.rstrip() + '"'
                    fixed_lines.append(line)
                content = '\n'.join(fixed_lines)
                
                try:
                    parsed_result = json.loads(content)
                except json.JSONDecodeError:
                    pass
            
            # If still no result, try to find valid JSON object
            if parsed_result is None:
                match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                if match:
                    try:
                        parsed_result = json.loads(match.group())
                    except json.JSONDecodeError:
                        pass
            
            # ===== ENHANCED COLOR VALIDATION =====
            # If we have a parsed result, check if colors need improvement
            if parsed_result is not None:
                color_data = parsed_result.get("color_profile", parsed_result.get("color", {}))
                primary_color = color_data.get("primary_color", color_data.get("primary", ""))
                
                # If color is unknown, empty, or not a recognized fashion color, try to extract from original
                if not primary_color or primary_color.lower() in ("unknown", "n/a", "null", "none", ""):
                    logger.info(f"Color missing or unknown in parsed JSON, enhancing with color_utils")
                    enhanced_color = self._extract_color_from_text(original_content)
                    if enhanced_color and enhanced_color != "unknown":
                        if "color_profile" not in parsed_result:
                            parsed_result["color_profile"] = {}
                        if "color" not in parsed_result:
                            parsed_result["color"] = {}
                        parsed_result["color_profile"]["primary_color"] = enhanced_color
                        parsed_result["color"]["primary"] = enhanced_color
                        logger.info(f"Enhanced color extraction: {enhanced_color}")
                else:
                    # Normalize the existing color name
                    normalized = normalize_color_name(primary_color)
                    if normalized != primary_color.lower():
                        if "color_profile" in parsed_result:
                            parsed_result["color_profile"]["primary_color"] = normalized
                        if "color" in parsed_result:
                            parsed_result["color"]["primary"] = normalized
                
                return parsed_result
            
            # 7. Last resort - manual extraction when all JSON parsing fails
            logger.warning(f"JSON parse error, attempting manual extraction with color_utils")
            result = {}
            
            # Extract category
            cat_match = re.search(r'"category"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
            if cat_match:
                result["category"] = cat_match.group(1).lower()
            
            # Extract taxonomy category
            tax_cat_match = re.search(r'taxonomy[^}]*"category"\s*:\s*"([^"]+)"', content, re.IGNORECASE | re.DOTALL)
            if tax_cat_match:
                result["taxonomy"] = result.get("taxonomy", {})
                result["taxonomy"]["category"] = tax_cat_match.group(1).lower()
            
            # ===== ENHANCED COLOR EXTRACTION using color_utils =====
            color_extracted = None
            secondary_color = None
            
            # Pattern 1: Extract from "primary_color" or "primary" field
            color_match = re.search(r'"primary_color"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
            if not color_match:
                color_match = re.search(r'"primary"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
            if color_match:
                raw_color = color_match.group(1)
                color_extracted = normalize_color_name(raw_color)
            
            # Pattern 2: Extract hex codes and convert to color names
            if not color_extracted or color_extracted == "unknown":
                hex_codes = extract_hex_codes(content)
                if hex_codes:
                    color_extracted = hex_to_color_name(hex_codes[0])
                    if len(hex_codes) > 1:
                        secondary_color = hex_to_color_name(hex_codes[1])
            
            # Pattern 3: Search for color names in the entire content
            if not color_extracted or color_extracted == "unknown":
                found_colors = extract_colors_from_text(content)
                if found_colors:
                    color_extracted = found_colors[0]
                    if len(found_colors) > 1:
                        secondary_color = found_colors[1]
            
            # Pattern 4: Look specifically in color_profile block
            if not color_extracted or color_extracted == "unknown":
                color_block = re.search(r'color_profile[^}]+}', content, re.IGNORECASE | re.DOTALL)
                if color_block:
                    block_colors = extract_colors_from_text(color_block.group())
                    if block_colors:
                        color_extracted = block_colors[0]
                        if len(block_colors) > 1:
                            secondary_color = block_colors[1]
            
            # Pattern 5: Use _extract_color_from_text as final fallback on original content
            if not color_extracted or color_extracted == "unknown":
                color_extracted = self._extract_color_from_text(original_content)
            
            # Set colors in result if found
            if color_extracted and color_extracted != "unknown":
                result["color_profile"] = {"primary_color": color_extracted}
                result["color"] = {"primary": color_extracted}
                logger.info(f"Color extracted using color_utils: {color_extracted}")
            
            # Extract secondary color from explicit fields if not found
            if not secondary_color:
                sec_color_match = re.search(r'"secondary_color"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
                if not sec_color_match:
                    sec_color_match = re.search(r'"secondary"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
                if sec_color_match:
                    raw_sec = sec_color_match.group(1)
                    if raw_sec.lower() not in ("null", "none", "n/a"):
                        secondary_color = normalize_color_name(raw_sec)
            
            # Add secondary color to result
            if secondary_color and secondary_color != "unknown":
                if "color_profile" in result:
                    result["color_profile"]["secondary_color"] = secondary_color
                if "color" in result:
                    result["color"]["secondary"] = secondary_color
            
            # Extract subcategory
            sub_match = re.search(r'"subcategory"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
            if sub_match:
                result["subcategory"] = sub_match.group(1)
                if "taxonomy" not in result:
                    result["taxonomy"] = {}
                result["taxonomy"]["subcategory"] = sub_match.group(1)
            
            # Extract pattern
            pattern_match = re.search(r'"pattern"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
            if not pattern_match:
                pattern_match = re.search(r'"pattern_type"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
            if pattern_match:
                result["pattern"] = {"type": pattern_match.group(1)}
            
            # Extract material
            material_match = re.search(r'"material"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
            if material_match:
                result["material_profile"] = {"material": material_match.group(1)}
            
            # Extract formality
            formality_match = re.search(r'"formality_level"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
            if formality_match:
                result["occasion_profile"] = {"formality_level": formality_match.group(1)}
            
            if result:
                extracted_keys = list(result.keys())
                color_info = result.get("color_profile", {}).get("primary_color", "not found")
                logger.info(f"Manual extraction recovered: {extracted_keys}, color: {color_info}")
                return result
            
            return {}
            
        except Exception as e:
            logger.warning(f"JSON parse error: {e}")
            return {}
    
    def _convert_to_model(self, raw: Dict[str, Any]) -> GarmentAttributes:
        """Convert raw extracted data to GarmentAttributes model."""
        
        # ========== Parse Taxonomy ==========
        taxonomy = raw.get("taxonomy", {})
        category_str = taxonomy.get("category", raw.get("category", "top")).lower()
        try:
            category = GarmentCategory(category_str)
        except ValueError:
            category = GarmentCategory.TOP
        
        subcategory = taxonomy.get("subcategory", raw.get("subcategory"))
        product_type = taxonomy.get("product_type")
        
        outfit_role = None
        role_str = taxonomy.get("outfit_role")
        if role_str:
            try:
                outfit_role = OutfitRole(role_str.lower())
            except ValueError:
                pass
        
        # ========== Parse Color Profile ==========
        color_data = raw.get("color_profile", raw.get("color", {}))
        
        # Parse color temperature
        color_temp = None
        temp_str = color_data.get("color_temperature")
        if temp_str:
            try:
                color_temp = ColorTemperature(temp_str.lower())
            except ValueError:
                pass
        
        # Parse color depth
        color_depth = None
        depth_str = color_data.get("color_depth")
        if depth_str:
            try:
                color_depth = ColorDepth(depth_str.lower())
            except ValueError:
                pass
        
        # Parse contrast level
        contrast = None
        contrast_str = color_data.get("contrast_level")
        if contrast_str:
            try:
                contrast = ContrastLevel(contrast_str.lower())
            except ValueError:
                pass
        
        color_profile = ColorProfile(
            primary=color_data.get("primary_color", color_data.get("primary", "unknown")),
            secondary=color_data.get("secondary_color", color_data.get("secondary")),
            color_palette=color_data.get("color_palette", []),
            hex_codes=color_data.get("hex_codes", []),
            color_temperature=color_temp,
            color_depth=color_depth,
            contrast_level=contrast
        )
        
        # ========== Parse Pattern ==========
        pattern_str = color_data.get("pattern", "solid")
        pattern_data = raw.get("pattern", {})
        pattern_info = PatternInfo(
            type=pattern_data.get("type", pattern_str) if isinstance(pattern_data, dict) else pattern_str,
            scale=pattern_data.get("scale") if isinstance(pattern_data, dict) else None,
            description=pattern_data.get("description") if isinstance(pattern_data, dict) else None
        )
        
        # ========== Parse Material Profile ==========
        material_data = raw.get("material_profile", raw.get("material", {}))
        material_profile = None
        if material_data:
            # Parse fabric weight
            fabric_weight = None
            weight_str = material_data.get("fabric_weight")
            if weight_str:
                try:
                    fabric_weight = FabricWeight(weight_str.lower())
                except ValueError:
                    pass
            
            # Parse stretch
            stretch = None
            stretch_str = material_data.get("stretch")
            if stretch_str:
                try:
                    stretch = StretchLevel(stretch_str.lower())
                except ValueError:
                    pass
            
            material_profile = MaterialProfile(
                primary=material_data.get("material", material_data.get("primary", "unknown")),
                secondary=material_data.get("secondary_material", material_data.get("secondary")),
                texture=material_data.get("texture"),
                fabric_weight=fabric_weight,
                stretch=stretch
            )
        
        # ========== Parse Silhouette Profile ==========
        silhouette_data = raw.get("silhouette_profile", {})
        silhouette_profile = None
        if silhouette_data:
            # Parse structure
            structure = None
            struct_str = silhouette_data.get("structure")
            if struct_str:
                try:
                    structure = StructureType(struct_str.lower())
                except ValueError:
                    pass
            
            # Parse volume
            volume = None
            vol_str = silhouette_data.get("volume")
            if vol_str:
                try:
                    volume = VolumeLevel(vol_str.lower())
                except ValueError:
                    pass
            
            silhouette_profile = SilhouetteProfile(
                fit=silhouette_data.get("fit"),
                structure=structure,
                length=silhouette_data.get("length"),
                volume=volume
            )
        
        # Legacy silhouette/fit fields
        silhouette = silhouette_data.get("fit", raw.get("silhouette"))
        fit = silhouette_data.get("fit", raw.get("fit"))
        
        # ========== Parse Style Identity ==========
        style_data = raw.get("style_identity", {})
        style_identity = None
        if style_data:
            # Parse trend alignment
            trend = None
            trend_str = style_data.get("trend_alignment")
            if trend_str:
                try:
                    trend = TrendAlignment(trend_str.lower())
                except ValueError:
                    pass
            
            # Parse statement level
            statement = None
            stmt_str = style_data.get("statement_level")
            if stmt_str:
                try:
                    statement = StatementLevel(stmt_str.lower())
                except ValueError:
                    pass
            
            style_identity = StyleIdentity(
                aesthetic_styles=style_data.get("aesthetic_styles", []),
                trend_alignment=trend,
                statement_level=statement
            )
        
        # Legacy style_tags
        style_tags = style_data.get("aesthetic_styles", raw.get("style_tags", []))
        
        # ========== Parse Styling Compatibility ==========
        compat_data = raw.get("styling_compatibility", {})
        styling_compatibility = None
        if compat_data:
            styling_compatibility = StylingCompatibility(
                layering_compatibility=compat_data.get("layering_compatibility", []),
                matching_bottoms=compat_data.get("matching_bottoms", []),
                matching_tops=compat_data.get("matching_tops", []),
                matching_outerwear=compat_data.get("matching_outerwear", []),
                matching_shoes=compat_data.get("matching_shoes", [])
            )
        
        # ========== Parse Occasion Profile ==========
        occasion_data = raw.get("occasion_profile", {})
        occasion_profile = None
        formality = FormalityLevel.CASUAL
        
        if occasion_data:
            formality_str = occasion_data.get("formality_level", "casual").lower()
            try:
                formality = FormalityLevel(formality_str)
            except ValueError:
                formality = FormalityLevel.CASUAL
            
            occasion_profile = OccasionProfile(
                formality_level=formality,
                occasions=occasion_data.get("occasions", [])
            )
        else:
            # Legacy formality parsing
            formality_str = raw.get("formality_level", "casual").lower()
            try:
                formality = FormalityLevel(formality_str)
            except ValueError:
                formality = FormalityLevel.CASUAL
        
        # ========== Parse Seasonality ==========
        season_data = raw.get("seasonality", {})
        seasonality = None
        seasons = []
        
        if season_data:
            seasons_raw = season_data.get("seasons", [])
            for s in seasons_raw:
                try:
                    seasons.append(Season(s.lower()))
                except ValueError:
                    pass
            
            # Parse temperature suitability
            temp_suit = None
            temp_str = season_data.get("temperature_suitability")
            if temp_str:
                try:
                    temp_suit = TemperatureSuitability(temp_str.lower())
                except ValueError:
                    pass
            
            seasonality = SeasonalityInfo(
                seasons=seasons,
                temperature_suitability=temp_suit
            )
        else:
            # Legacy season parsing
            seasons_raw = raw.get("season_suitable", [])
            for s in seasons_raw:
                try:
                    seasons.append(Season(s.lower()))
                except ValueError:
                    pass
        
        # ========== Parse Versatility ==========
        vers_data = raw.get("versatility", {})
        versatility = None
        if vers_data:
            versatility = VersatilityInfo(
                versatility_score=float(vers_data.get("versatility_score", 0.5)),
                capsule_wardrobe_friendly=vers_data.get("capsule_wardrobe_friendly", False)
            )
        
        # ========== Parse Garment Details ==========
        details_data = raw.get("garment_details", raw.get("details", {}))
        
        # Parse neckline
        neckline = None
        neckline_str = details_data.get("neckline", raw.get("neckline"))
        if neckline_str and neckline_str != "null":
            try:
                neckline = NecklineType(neckline_str.lower())
            except ValueError:
                pass
        
        # Parse sleeves
        sleeves = None
        sleeves_data = details_data.get("sleeves", raw.get("sleeves", {}))
        if sleeves_data and sleeves_data.get("length") and sleeves_data.get("length") != "null":
            sleeves = SleeveType(
                length=sleeves_data.get("length", "short"),
                style=sleeves_data.get("style") if sleeves_data.get("style") != "null" else None
            )
        
        # Parse length type
        length_type = None
        length_str = details_data.get("length_type", raw.get("length_type"))
        if length_str and length_str != "null":
            try:
                length_type = LengthType(length_str.lower())
            except ValueError:
                pass
        
        # Parse waist rise
        waist_rise = None
        waist_str = details_data.get("waist_rise", raw.get("waist_rise"))
        if waist_str and waist_str != "null":
            try:
                waist_rise = WaistRise(waist_str.lower())
            except ValueError:
                pass
        
        # Parse closure
        closure = None
        closure_str = details_data.get("closure", raw.get("closure"))
        if closure_str and closure_str != "null":
            try:
                closure = ClosureType(closure_str.lower())
            except ValueError:
                pass
        
        # Parse transparency
        transparency_str = details_data.get("transparency", "opaque")
        try:
            transparency = TransparencyLevel(transparency_str.lower())
        except ValueError:
            transparency = TransparencyLevel.OPAQUE
        
        details = DetailTags(
            has_pockets=details_data.get("has_pockets", False),
            distressed=details_data.get("distressed", False),
            embellishments=details_data.get("embellishments", []),
            transparency=transparency,
            lined=details_data.get("lined", False),
            reversible=details_data.get("reversible", False)
        )
        
        # ========== Parse Confidence Score ==========
        confidence = float(raw.get("confidence_score", 0.0))
        
        # ========== Build Final Model ==========
        return GarmentAttributes(
            # Core taxonomy
            category=category,
            subcategory=subcategory,
            product_type=product_type,
            outfit_role=outfit_role,
            
            # Color profile
            color=color_profile,
            
            # Material profile
            material=material_profile,
            
            # Pattern
            pattern=pattern_info,
            
            # Silhouette profile
            silhouette_profile=silhouette_profile,
            silhouette=silhouette,
            fit=fit,
            
            # Style identity
            style_identity=style_identity,
            style_tags=style_tags,
            
            # Styling compatibility
            styling_compatibility=styling_compatibility,
            
            # Occasion profile
            occasion_profile=occasion_profile,
            formality_level=formality,
            
            # Seasonality
            seasonality=seasonality,
            season_suitable=seasons,
            
            # Versatility
            versatility=versatility,
            
            # Garment details
            neckline=neckline,
            sleeves=sleeves,
            length_type=length_type,
            waist_rise=waist_rise,
            closure=closure,
            details=details,
            
            # Confidence
            confidence_score=confidence
        )
