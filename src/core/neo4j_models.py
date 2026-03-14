"""
Neo4j Graph Models
==================

Defines the graph schema for Neo4j, including node types, relationships,
and their properties. This module provides the logical structure for
storing and querying wardrobe data in Neo4j.

Graph Structure:
    ├─ Garment (nodes)
    ├─ Category (nodes)
    ├─ Season (nodes)
    ├─ Color (nodes)
    ├─ Occasion (nodes)
    ├─ FormalityLevel (nodes)
    ├─ COMPATIBLE_WITH (relationships)
    ├─ BELONGS_TO (relationships)
    ├─ HAS_COLOR (relationships)
    ├─ SUITABLE_FOR (relationships)
    └─ ... (many others)
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime


# ============================================================================
# NODE SCHEMAS
# ============================================================================


@dataclass
class GarmentNode:
    """Garment node representation in Neo4j."""
    id: str
    title: str
    category: str
    subcategory: Optional[str] = None
    
    # Color properties
    color_primary: Optional[str] = None
    color_secondary: Optional[str] = None
    
    # Style properties
    pattern_type: Optional[str] = None
    material: Optional[str] = None
    formality_level: int = 2  # 0-6 scale
    
    # Metadata
    rating: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_worn_at: Optional[str] = None
    total_wears: int = 0
    
    # Computed properties (updated weekly)
    versatility_score: float = 0.5
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    pagerank: float = 0.0
    
    # Status
    is_retired: bool = False
    user_id: str = ""
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j node properties."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class CategoryNode:
    """Category node (TOP, BOTTOM, SHOES, etc.)."""
    name: str
    value: str
    priority: int = 0
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SeasonNode:
    """Season node (SPRING, SUMMER, FALL, WINTER)."""
    name: str
    value: str
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ColorNode:
    """Color node representation."""
    name: str
    hex: str
    rgb: Optional[str] = None
    role: str = "neutral"  # neutral, accent, warm, cool
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OccasionNode:
    """Occasion node (CASUAL, FORMAL, BUSINESS, etc.)."""
    name: str
    value: str
    importance: float = 0.5  # 0-1 scale
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FormalityLevelNode:
    """Formality level node (0=Very Casual to 6=Black Tie)."""
    name: str
    value: int
    order: int
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutfitNode:
    """Outfit node representation."""
    id: str
    user_id: str
    occasion: Optional[str] = None
    formality: int = 2
    season: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_worn_at: Optional[str] = None
    total_wears: int = 0
    overall_score: float = 0.0
    description: Optional[str] = None
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserNode:
    """User node representation."""
    id: str
    profile: str = "DEFAULT"  # MINIMALIST, MAXIMIZE_OPTIONS, STYLE_UPGRADE, etc.
    wardrobe_size: int = 0
    total_outfits: int = 0
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# RELATIONSHIP SCHEMAS
# ============================================================================


@dataclass
class CompatibleWithRelationship:
    """COMPATIBLE_WITH relationship between garments."""
    weight: float  # 0-1 score
    occasions: List[str] = field(default_factory=list)
    seasons: List[str] = field(default_factory=list)
    formality_delta: int = 0
    color_harmony: float = 0.5
    pattern_compatible: bool = True
    volume_balance_score: float = 0.5
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ColorHarmonizesRelationship:
    """Color harmony relationship (COMPLEMENTS, ANALOGOUS, TRIADIC)."""
    weight: float  # 0-1 harmony score
    harmony_type: str = "COMPLEMENTS"  # COMPLEMENTS, ANALOGOUS, TRIADIC
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SuitableForRelationship:
    """SUITABLE_FOR relationship (garment to season/occasion)."""
    weight: float = 0.8  # How suitable (0.6=ok, 0.9=primary)
    is_primary: bool = False
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContainsRelationship:
    """CONTAINS relationship (outfit to garment)."""
    role: str = "main"  # main, accent, outer, accessory
    position: int = 0
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WoreRelationship:
    """WORE relationship (user wore garment)."""
    worn_at: str
    outfit_id: Optional[str] = None
    rating: float = 0.5
    occasions: List[str] = field(default_factory=list)
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# GRAPH SCHEMA DEFINITION
# ============================================================================


class GraphSchema:
    """Defines the complete Neo4j graph schema."""
    
    # Node types
    NODE_TYPES = [
        'Garment',
        'Category',
        'Season',
        'Color',
        'Occasion',
        'FormalityLevel',
        'Outfit',
        'User'
    ]
    
    # Relationship types
    RELATIONSHIP_TYPES = [
        'COMPATIBLE_WITH',
        'COMPLEMENTS',
        'ANALOGOUS',
        'TRIADIC',
        'BELONGS_TO',
        'HAS_COLOR',
        'SUITABLE_FOR',
        'SUITABLE_FOR_OCCASION',
        'FORMALITY_LEVEL',
        'CONTAINS',
        'OWNS',
        'WORE',
        'CREATED',
        'WORE_OUTFIT',
        'SIMILAR_TO',
        'CATEGORY_PAIRS_WITH',
        'PATTERN_COMPATIBLE'
    ]
    
    # Indexes to create
    INDEXES = [
        ("garment_id", "CREATE INDEX garment_id IF NOT EXISTS FOR (g:Garment) ON (g.id)"),
        ("garment_category", "CREATE INDEX garment_category IF NOT EXISTS FOR (g:Garment) ON (g.category)"),
        ("garment_formality", "CREATE INDEX garment_formality IF NOT EXISTS FOR (g:Garment) ON (g.formality_level)"),
        ("garment_user", "CREATE INDEX garment_user IF NOT EXISTS FOR (g:Garment) ON (g.user_id)"),
        ("season_name", "CREATE INDEX season_name IF NOT EXISTS FOR (s:Season) ON (s.name)"),
        ("occasion_name", "CREATE INDEX occasion_name IF NOT EXISTS FOR (o:Occasion) ON (o.name)"),
        ("color_name", "CREATE INDEX color_name IF NOT EXISTS FOR (c:Color) ON (c.name)"),
        ("outfit_user", "CREATE INDEX outfit_user IF NOT EXISTS FOR (o:Outfit) ON (o.user_id)"),
        ("user_id", "CREATE INDEX user_id IF NOT EXISTS FOR (u:User) ON (u.id)"),
    ]
    
    @staticmethod
    def get_node_properties(node_type: str) -> Dict[str, str]:
        """Get property schema for a node type."""
        schemas = {
            'Garment': {
                'id': 'STRING',
                'title': 'STRING',
                'category': 'STRING',
                'color_primary': 'STRING',
                'formality_level': 'INTEGER',
                'rating': 'FLOAT',
                'versatility_score': 'FLOAT',
            },
            'Category': {
                'name': 'STRING',
                'value': 'STRING',
            },
            'Season': {
                'name': 'STRING',
                'value': 'STRING',
            },
            'Color': {
                'name': 'STRING',
                'hex': 'STRING',
                'role': 'STRING',
            },
            'Occasion': {
                'name': 'STRING',
                'value': 'STRING',
                'importance': 'FLOAT',
            },
        }
        return schemas.get(node_type, {})
    
    @staticmethod
    def get_relationship_properties(rel_type: str) -> Dict[str, str]:
        """Get property schema for a relationship type."""
        schemas = {
            'COMPATIBLE_WITH': {
                'weight': 'FLOAT',
                'formality_delta': 'INTEGER',
                'color_harmony': 'FLOAT',
            },
            'SUITABLE_FOR': {
                'weight': 'FLOAT',
                'is_primary': 'BOOLEAN',
            },
            'CONTAINS': {
                'role': 'STRING',
                'position': 'INTEGER',
            },
            'WORE': {
                'worn_at': 'STRING',
                'rating': 'FLOAT',
            },
        }
        return schemas.get(rel_type, {})


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_garment_node_from_dict(data: Dict[str, Any]) -> GarmentNode:
    """Create a GarmentNode from a dictionary."""
    return GarmentNode(**{k: v for k, v in data.items() if k in GarmentNode.__dataclass_fields__})


def create_compatibility_from_scores(
    score_dict: Dict[str, float]
) -> CompatibleWithRelationship:
    """
    Create a CompatibleWithRelationship from a scoring dictionary.
    
    Expected keys:
    - weight: overall compatibility
    - color_harmony: color score
    - pattern_compatible: boolean
    - volume_balance: silhouette score
    """
    return CompatibleWithRelationship(
        weight=score_dict.get('weight', 0.0),
        color_harmony=score_dict.get('color_harmony', 0.5),
        pattern_compatible=score_dict.get('pattern_compatible', True),
        volume_balance_score=score_dict.get('volume_balance', 0.5),
    )
