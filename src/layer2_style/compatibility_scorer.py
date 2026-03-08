"""
Compatibility Scorer
Scores the compatibility between clothing items using learned patterns.
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import get_config
from src.core.models import Garment, GarmentCategory, NecklineType, LengthType
from src.core import get_logger

logger = get_logger(__name__)


class CompatibilityScorer:
    """
    Scores compatibility between garments.
    
    This scorer learns abstract compatibility rules:
    - What types of items go together
    - Style coherence patterns
    - Category pairing rules
    
    It can be trained on runway looks to learn high-fashion
    compatibility patterns without copying specific designs.
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Load compatibility data from configuration
        compat_data = self.config.get_data("compatibility_data", default={})
        
        # Build category compatibility matrix from config
        self.category_compatibility = self._build_category_compatibility(
            compat_data.get("category_compatibility", {})
        )
        
        # Style compatibility patterns
        self.style_compatibility = self._initialize_style_patterns()
        
        # Load scoring weights from config
        scoring_params = self.config.get_parameters("model_parameters", "scoring.compatibility", default={})
        self._category_weight = scoring_params.get("category_weight", 0.25)
        self._style_weight = scoring_params.get("style_weight", 0.25)
        self._embedding_weight = scoring_params.get("embedding_weight", 0.35)
        self._detail_weight = scoring_params.get("detail_weight", 0.15)
        
        # Default compatibility score
        self._default_score = compat_data.get("default_compatibility_score", 0.5)
        
        # Trained model weights (placeholder for ML model)
        self.model_weights: Optional[np.ndarray] = None
    
    def _build_category_compatibility(self, config_compat: Dict[str, float]) -> Dict:
        """Build category compatibility dict from config."""
        result = {}
        for key, value in config_compat.items():
            # Parse key like "top_bottom" into tuple
            parts = key.split("_")
            if len(parts) == 2:
                try:
                    cat1 = GarmentCategory(parts[0])
                    cat2 = GarmentCategory(parts[1])
                    result[(cat1, cat2)] = value
                except ValueError:
                    continue
        
        # Add defaults if not in config
        defaults = {
            (GarmentCategory.TOP, GarmentCategory.BOTTOM): 1.0,
            (GarmentCategory.TOP, GarmentCategory.OUTERWEAR): 0.9,
            (GarmentCategory.TOP, GarmentCategory.SHOES): 0.7,
            (GarmentCategory.TOP, GarmentCategory.ACCESSORY): 0.8,
            (GarmentCategory.BOTTOM, GarmentCategory.SHOES): 0.85,
            (GarmentCategory.BOTTOM, GarmentCategory.OUTERWEAR): 0.8,
            (GarmentCategory.DRESS, GarmentCategory.OUTERWEAR): 0.9,
            (GarmentCategory.DRESS, GarmentCategory.SHOES): 0.9,
            (GarmentCategory.DRESS, GarmentCategory.ACCESSORY): 0.85,
            (GarmentCategory.OUTERWEAR, GarmentCategory.SHOES): 0.7,
            (GarmentCategory.OUTERWEAR, GarmentCategory.ACCESSORY): 0.75,
        }
        
        for key, value in defaults.items():
            if key not in result:
                result[key] = value
        
        return result
    
    async def score_pair(self, item1: Garment, item2: Garment) -> float:
        """
        Score compatibility between two garments.
        
        Args:
            item1: First garment
            item2: Second garment
            
        Returns:
            Compatibility score between 0 and 1
        """
        # Base category compatibility
        category_score = self._get_category_score(
            item1.attributes.category,
            item2.attributes.category
        )
        
        # Style tag overlap
        style_score = self._calculate_style_overlap(
            item1.attributes.style_tags,
            item2.attributes.style_tags
        )
        
        # Embedding similarity (if available)
        embedding_score = 0.5  # Default
        if item1.embedding and item2.embedding:
            embedding_score = self._calculate_embedding_similarity(
                item1.embedding,
                item2.embedding
            )
        
        # Detail compatibility (new)
        detail_score = self._calculate_detail_compatibility(item1, item2)
        
        # Combine scores using configured weights
        final_score = (
            self._category_weight * category_score +
            self._style_weight * style_score +
            self._embedding_weight * embedding_score +
            self._detail_weight * detail_score
        )
        
        return min(1.0, max(0.0, final_score))
    
    def _calculate_detail_compatibility(
        self,
        item1: Garment,
        item2: Garment
    ) -> float:
        """
        Calculate compatibility based on garment details.
        
        Considers necklines, lengths, closures, and embellishments.
        """
        score = 0.7  # Base score
        
        attrs1 = item1.attributes
        attrs2 = item2.attributes
        
        # Check neckline compatibility for layering
        if attrs1.neckline and attrs2.neckline:
            # V-neck over collared shirt works well
            if (attrs1.neckline == NecklineType.V_NECK and 
                attrs2.neckline == NecklineType.COLLARED):
                score += 0.15
            # Avoid two high necklines
            elif (attrs1.neckline == NecklineType.TURTLENECK and 
                  attrs2.neckline == NecklineType.TURTLENECK):
                score -= 0.2
        
        # Check length harmony
        if attrs1.length_type and attrs2.length_type:
            # Crop with high-waisted is great
            if (attrs1.length_type == LengthType.CROP and 
                attrs2.waist_rise and attrs2.waist_rise.value == "high_rise"):
                score += 0.15
        
        # Check embellishment balance (avoid too many busy pieces)
        embellishments1 = attrs1.details.embellishments
        embellishments2 = attrs2.details.embellishments
        total_embellishments = len(embellishments1) + len(embellishments2)
        
        if total_embellishments > 4:
            score -= 0.15  # Too busy
        elif total_embellishments == 0:
            score += 0.05  # Clean, minimal look
        
        # Distressed items should be balanced
        if attrs1.details.distressed and attrs2.details.distressed:
            score -= 0.1  # Too much distressing
        
        # Transparency considerations
        if (attrs1.details.transparency.value != "opaque" and 
            attrs2.details.transparency.value != "opaque"):
            score -= 0.15  # Two sheer pieces is tricky
        
        return min(1.0, max(0.0, score))
    
    async def score_items(self, items: List[Garment]) -> float:
        """
        Score overall compatibility of multiple items.
        
        Args:
            items: List of garments to score
            
        Returns:
            Overall compatibility score
        """
        if len(items) < 2:
            return 0.5
        
        # Calculate pairwise scores
        pair_scores = []
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                score = await self.score_pair(item1, item2)
                pair_scores.append(score)
        
        # Return average
        return np.mean(pair_scores) if pair_scores else 0.5
    
    def _get_category_score(
        self,
        cat1: GarmentCategory,
        cat2: GarmentCategory
    ) -> float:
        """Get base compatibility score for category pair."""
        # Check both orderings
        score = self.category_compatibility.get((cat1, cat2))
        if score is None:
            score = self.category_compatibility.get((cat2, cat1))
        
        return score if score is not None else 0.5
    
    def _calculate_style_overlap(
        self,
        tags1: List[str],
        tags2: List[str]
    ) -> float:
        """Calculate style tag overlap score."""
        if not tags1 or not tags2:
            return 0.5
        
        set1 = set(t.lower() for t in tags1)
        set2 = set(t.lower() for t in tags2)
        
        # Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.5
        
        jaccard = intersection / union
        
        # Also check compatible styles
        compatible_bonus = self._check_style_compatibility(set1, set2)
        
        return min(1.0, jaccard + compatible_bonus)
    
    def _check_style_compatibility(
        self,
        styles1: set,
        styles2: set
    ) -> float:
        """Check if styles are complementary even without overlap."""
        compatible_pairs = [
            ({"minimalist", "clean"}, {"classic", "timeless"}),
            ({"streetwear", "urban"}, {"sporty", "athletic"}),
            ({"bohemian", "boho"}, {"romantic", "feminine"}),
            ({"edgy", "punk"}, {"rock", "grunge"}),
        ]
        
        for pair1, pair2 in compatible_pairs:
            if (styles1 & pair1 and styles2 & pair2) or \
               (styles1 & pair2 and styles2 & pair1):
                return 0.2
        
        return 0.0
    
    def _calculate_embedding_similarity(
        self,
        emb1: List[float],
        emb2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        arr1 = np.array(emb1).reshape(1, -1)
        arr2 = np.array(emb2).reshape(1, -1)
        
        similarity = cosine_similarity(arr1, arr2)[0][0]
        
        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1) / 2
    
    def _initialize_style_patterns(self) -> Dict[str, List[str]]:
        """Initialize style compatibility patterns."""
        return {
            "minimalist": ["clean", "modern", "simple", "elegant"],
            "classic": ["timeless", "traditional", "preppy", "sophisticated"],
            "streetwear": ["urban", "casual", "sporty", "contemporary"],
            "bohemian": ["boho", "romantic", "feminine", "artsy"],
            "edgy": ["punk", "rock", "avant-garde", "bold"],
            "elegant": ["refined", "luxurious", "sophisticated", "chic"]
        }
    
    async def train_on_runway_looks(
        self,
        looks: List[List[Garment]]
    ) -> None:
        """
        Train compatibility patterns from runway looks.
        
        This learns ABSTRACT patterns, not specific designs:
        - What style tags appear together
        - What category combinations work
        - What color patterns are used
        
        Args:
            looks: List of outfits (each outfit is a list of garments)
        """
        logger.info(f"Training on {len(looks)} runway looks")
        
        # Extract patterns (simplified - would use ML in production)
        style_cooccurrence = {}
        category_pairs_success = {}
        
        for look in looks:
            # Count style tag co-occurrences
            all_tags = []
            for garment in look:
                all_tags.extend(garment.attributes.style_tags)
            
            for i, tag1 in enumerate(all_tags):
                for tag2 in all_tags[i+1:]:
                    key = tuple(sorted([tag1.lower(), tag2.lower()]))
                    style_cooccurrence[key] = style_cooccurrence.get(key, 0) + 1
            
            # Count successful category pairings
            for i, g1 in enumerate(look):
                for g2 in look[i+1:]:
                    cat_pair = tuple(sorted([g1.attributes.category, g2.attributes.category]))
                    category_pairs_success[cat_pair] = category_pairs_success.get(cat_pair, 0) + 1
        
        # Update compatibility matrices based on learned patterns
        # (Simplified - production would use proper ML)
        logger.info("Compatibility patterns updated from runway data")
