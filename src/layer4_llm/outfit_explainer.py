"""
Outfit Explainer
Generates detailed explanations of outfit recommendations.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass

from config import get_config
from src.core.models import Outfit, Garment, UserContext
from src.core import get_logger
from .llm_service import LLMService

logger = get_logger(__name__)


@dataclass
class OutfitExplanation:
    """Structured outfit explanation."""
    summary: str
    style_notes: List[str]
    color_harmony_note: str
    occasion_fit: str
    styling_tips: List[str]
    confidence: float


class OutfitExplainer:
    """
    Generates comprehensive outfit explanations.
    
    Combines technical style analysis with LLM-generated
    natural language to create helpful explanations.
    """
    
    def __init__(self):
        self.llm_service = LLMService()
        self.config = get_config()
        self._llm_params = self.config.get_parameters("model_parameters", "llm", default={})
    
    async def explain(
        self,
        outfit: Outfit,
        context: Optional[UserContext] = None,
        detail_level: str = "standard"
    ) -> OutfitExplanation:
        """
        Generate complete explanation for an outfit.
        
        Args:
            outfit: Outfit to explain
            context: User context
            detail_level: "brief", "standard", or "detailed"
            
        Returns:
            Structured explanation
        """
        # Generate summary
        summary = await self.llm_service.explain_outfit(outfit, context)
        
        # Generate specific notes
        style_notes = self._generate_style_notes(outfit)
        color_note = self._generate_color_note(outfit)
        occasion_note = self._generate_occasion_note(outfit, context)
        
        # Generate tips if detailed
        tips = []
        if detail_level == "detailed":
            tips = await self._generate_styling_tips(outfit)
        
        return OutfitExplanation(
            summary=summary,
            style_notes=style_notes,
            color_harmony_note=color_note,
            occasion_fit=occasion_note,
            styling_tips=tips,
            confidence=outfit.overall_score
        )
    
    async def explain_comparison(
        self,
        outfit1: Outfit,
        outfit2: Outfit,
        context: Optional[UserContext] = None
    ) -> str:
        """
        Explain comparison between two outfits.
        
        Args:
            outfit1: First outfit
            outfit2: Second outfit
            context: User context
            
        Returns:
            Comparison explanation
        """
        prompt = self.config.get_prompt(
            "llm_outfit_comparison",
            outfit1_score=f"{outfit1.overall_score:.0%}",
            outfit1_description=self._describe_outfit(outfit1),
            outfit2_score=f"{outfit2.overall_score:.0%}",
            outfit2_description=self._describe_outfit(outfit2)
        )
        
        return await self.llm_service.generate_completion(
            prompt=prompt,
            max_tokens=self._llm_params.get("comparison_max_tokens", 200)
        )
    
    async def explain_rejection(
        self,
        outfit: Outfit,
        reason: str
    ) -> str:
        """
        Explain why an outfit combination doesn't work well.
        
        Args:
            outfit: The problematic outfit
            reason: Technical reason for rejection
            
        Returns:
            User-friendly explanation
        """
        prompt = self.config.get_prompt(
            "llm_outfit_rejection",
            outfit_description=self._describe_outfit(outfit),
            reason=reason
        )
        
        return await self.llm_service.generate_completion(
            prompt=prompt,
            max_tokens=self._llm_params.get("rejection_max_tokens", 150)
        )
    
    def _generate_style_notes(self, outfit: Outfit) -> List[str]:
        """Generate style analysis notes."""
        notes = []
        
        # Analyze silhouette balance
        silhouettes = []
        for item in outfit.items:
            if item.garment.attributes.silhouette:
                silhouettes.append(item.garment.attributes.silhouette)
        
        if silhouettes:
            if len(set(silhouettes)) == 1:
                notes.append(f"Cohesive {silhouettes[0]} silhouette throughout")
            else:
                notes.append(f"Balanced mix of {' and '.join(set(silhouettes))} fits")
        
        # Analyze formality
        formalities = [item.garment.attributes.formality_level for item in outfit.items]
        if len(set(formalities)) == 1:
            notes.append(f"Consistent {formalities[0].value.replace('_', ' ')} level")
        
        # Analyze style tags
        all_tags = []
        for item in outfit.items:
            all_tags.extend(item.garment.attributes.style_tags)
        
        if all_tags:
            common_tags = set(all_tags)
            if len(common_tags) <= 3:
                notes.append(f"Style direction: {', '.join(common_tags)}")
        
        # Analyze necklines
        necklines = []
        for item in outfit.items:
            if item.garment.attributes.neckline:
                necklines.append(item.garment.attributes.neckline.value.replace('_', ' '))
        if necklines:
            notes.append(f"Neckline: {', '.join(set(necklines))}")
        
        # Analyze lengths for proportion
        lengths = []
        for item in outfit.items:
            if item.garment.attributes.length_type:
                lengths.append(item.garment.attributes.length_type.value)
        if lengths:
            notes.append(f"Length proportions: {', '.join(lengths)}")
        
        # Note special details
        special_details = []
        for item in outfit.items:
            details = item.garment.attributes.details
            if details.distressed:
                special_details.append("distressed elements")
            if details.embellishments:
                special_details.extend(details.embellishments[:2])
        if special_details:
            notes.append(f"Notable details: {', '.join(set(special_details))}")
        
        return notes
    
    def _generate_color_note(self, outfit: Outfit) -> str:
        """Generate color harmony note."""
        colors = []
        for item in outfit.items:
            colors.append(item.garment.attributes.color.primary)
        
        unique_colors = list(set(colors))
        
        if len(unique_colors) == 1:
            return f"Monochromatic {unique_colors[0]} palette for a sleek look"
        elif len(unique_colors) == 2:
            return f"Clean {unique_colors[0]} and {unique_colors[1]} combination"
        else:
            return f"Multi-color palette: {', '.join(unique_colors[:3])}"
    
    def _generate_occasion_note(
        self,
        outfit: Outfit,
        context: Optional[UserContext]
    ) -> str:
        """Generate occasion appropriateness note."""
        score = outfit.occasion_match_score
        
        if context and context.occasion:
            occasion = context.occasion.value
            if score >= 0.8:
                return f"Excellent fit for {occasion} occasions"
            elif score >= 0.6:
                return f"Suitable for {occasion} settings"
            else:
                return f"May need adjustments for {occasion}"
        
        # Default based on formality
        avg_formality = sum(
            item.garment.attributes.formality_level.value 
            for item in outfit.items
        ) / len(outfit.items) if outfit.items else "casual"
        
        return f"Versatile for everyday wear"
    
    async def _generate_styling_tips(self, outfit: Outfit) -> List[str]:
        """Generate styling tips for the outfit."""
        tips = []
        
        # Tip based on lowest score aspect
        if outfit.compatibility_score < 0.7:
            tip = await self.llm_service.generate_styling_tip(
                outfit, "item compatibility"
            )
            tips.append(tip)
        
        if outfit.style_coherence_score < 0.7:
            tip = await self.llm_service.generate_styling_tip(
                outfit, "style coherence"
            )
            tips.append(tip)
        
        # General accessory tip
        has_accessories = any(
            item.garment.attributes.category.value == "accessory"
            for item in outfit.items
        )
        
        if not has_accessories:
            tips.append("Consider adding an accessory to complete the look")
        
        return tips
    
    def _describe_outfit(self, outfit: Outfit) -> str:
        """Create simple outfit description."""
        items = []
        for item in outfit.items:
            attrs = item.garment.attributes
            description_parts = [attrs.color.primary]
            
            # Add neckline for tops/dresses
            if attrs.neckline:
                description_parts.append(attrs.neckline.value.replace('_', ' '))
            
            # Add length for dresses/bottoms
            if attrs.length_type:
                description_parts.append(attrs.length_type.value)
            
            # Add sleeve info
            if attrs.sleeves:
                description_parts.append(f"{attrs.sleeves.length} sleeve")
            
            # Add category/subcategory
            description_parts.append(attrs.subcategory or attrs.category.value)
            
            # Add waist rise for bottoms
            if attrs.waist_rise:
                description_parts.insert(-1, attrs.waist_rise.value.replace('_', ' '))
            
            items.append(" ".join(description_parts))
        return ", ".join(items)
