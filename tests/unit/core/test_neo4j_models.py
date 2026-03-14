"""
Tests for Neo4j Models

Tests for Neo4j graph models and schema definitions:
- Node dataclasses
- Relationship dataclasses
- Conversion to Neo4j properties
- Schema definitions
- GraphSchema utilities

Markers:
    @pytest.mark.unit: Fast unit tests
"""

import pytest
from datetime import datetime

from src.core.neo4j_models import (
    GarmentNode,
    CategoryNode,
    SeasonNode,
    ColorNode,
    OccasionNode,
    FormalityLevelNode,
    OutfitNode,
    UserNode,
    CompatibleWithRelationship,
    ColorHarmonizesRelationship,
    SuitableForRelationship,
    ContainsRelationship,
    WoreRelationship,
    GraphSchema,
    create_garment_node_from_dict,
    create_compatibility_from_scores,
)


# ============================================================================
# GARMENT NODE TESTS
# ============================================================================


class TestGarmentNode:
    """Tests for GarmentNode dataclass."""
    
    def test_garment_node_creation(self):
        """Test GarmentNode can be created with required fields."""
        garment = GarmentNode(
            id="g1",
            title="White T-Shirt",
            category="TOP"
        )
        
        assert garment.id == "g1"
        assert garment.title == "White T-Shirt"
        assert garment.category == "TOP"
    
    def test_garment_node_with_all_fields(self):
        """Test GarmentNode with all fields."""
        garment = GarmentNode(
            id="g1",
            title="White T-Shirt",
            category="TOP",
            subcategory="t-shirt",
            color_primary="white",
            color_secondary="gray",
            pattern_type="solid",
            material="cotton",
            formality_level=1,
            rating=0.9,
            total_wears=10,
            versatility_score=0.8,
            pagerank=0.05,
            user_id="user1"
        )
        
        assert garment.formality_level == 1
        assert garment.rating == 0.9
        assert garment.total_wears == 10
    
    def test_garment_node_to_neo4j_properties(self):
        """Test GarmentNode.to_neo4j_properties()."""
        garment = GarmentNode(
            id="g1",
            title="White T-Shirt",
            category="TOP",
            color_primary="white",
            rating=0.9
        )
        
        props = garment.to_neo4j_properties()
        
        assert props["id"] == "g1"
        assert props["title"] == "White T-Shirt"
        assert props["category"] == "TOP"
        assert props["color_primary"] == "white"
        assert props["rating"] == 0.9
    
    def test_garment_node_default_values(self):
        """Test GarmentNode default values."""
        garment = GarmentNode(
            id="g1",
            title="White T-Shirt",
            category="TOP"
        )
        
        assert garment.formality_level == 2
        assert garment.rating == 0.5
        assert garment.total_wears == 0
        assert garment.is_retired is False


# ============================================================================
# CATEGORY NODE TESTS
# ============================================================================


class TestCategoryNode:
    """Tests for CategoryNode dataclass."""
    
    def test_category_node_creation(self):
        """Test CategoryNode creation."""
        category = CategoryNode(
            name="Top",
            value="TOP",
            priority=1
        )
        
        assert category.name == "Top"
        assert category.value == "TOP"
        assert category.priority == 1
    
    def test_category_node_to_neo4j_properties(self):
        """Test CategoryNode.to_neo4j_properties()."""
        category = CategoryNode(
            name="Top",
            value="TOP"
        )
        
        props = category.to_neo4j_properties()
        
        assert props["name"] == "Top"
        assert props["value"] == "TOP"


# ============================================================================
# SEASON NODE TESTS
# ============================================================================


class TestSeasonNode:
    """Tests for SeasonNode dataclass."""
    
    def test_season_node_creation(self):
        """Test SeasonNode creation."""
        season = SeasonNode(
            name="Spring",
            value="SPRING"
        )
        
        assert season.name == "Spring"
        assert season.value == "SPRING"
    
    def test_season_node_to_neo4j_properties(self):
        """Test SeasonNode.to_neo4j_properties()."""
        season = SeasonNode(
            name="Summer",
            value="SUMMER"
        )
        
        props = season.to_neo4j_properties()
        
        assert props["name"] == "Summer"
        assert props["value"] == "SUMMER"


# ============================================================================
# COLOR NODE TESTS
# ============================================================================


class TestColorNode:
    """Tests for ColorNode dataclass."""
    
    def test_color_node_creation(self):
        """Test ColorNode creation."""
        color = ColorNode(
            name="Blue",
            hex="#0000FF"
        )
        
        assert color.name == "Blue"
        assert color.hex == "#0000FF"
    
    def test_color_node_with_role(self):
        """Test ColorNode with color role."""
        color = ColorNode(
            name="Red",
            hex="#FF0000",
            role="warm"
        )
        
        assert color.role == "warm"
    
    def test_color_node_to_neo4j_properties(self):
        """Test ColorNode.to_neo4j_properties()."""
        color = ColorNode(
            name="Green",
            hex="#00FF00",
            role="cool"
        )
        
        props = color.to_neo4j_properties()
        
        assert props["name"] == "Green"
        assert props["hex"] == "#00FF00"
        assert props["role"] == "cool"
    
    def test_color_node_default_role(self):
        """Test ColorNode default role."""
        color = ColorNode(
            name="Gray",
            hex="#808080"
        )
        
        assert color.role == "neutral"


# ============================================================================
# OCCASION NODE TESTS
# ============================================================================


class TestOccasionNode:
    """Tests for OccasionNode dataclass."""
    
    def test_occasion_node_creation(self):
        """Test OccasionNode creation."""
        occasion = OccasionNode(
            name="Business",
            value="BUSINESS"
        )
        
        assert occasion.name == "Business"
        assert occasion.value == "BUSINESS"
    
    def test_occasion_node_with_importance(self):
        """Test OccasionNode with importance."""
        occasion = OccasionNode(
            name="Wedding",
            value="WEDDING",
            importance=0.9
        )
        
        assert occasion.importance == 0.9


# ============================================================================
# FORMALITY LEVEL NODE TESTS
# ============================================================================


class TestFormalityLevelNode:
    """Tests for FormalityLevelNode dataclass."""
    
    def test_formality_level_node_creation(self):
        """Test FormalityLevelNode creation."""
        level = FormalityLevelNode(
            name="Business Casual",
            value=3,
            order=3
        )
        
        assert level.name == "Business Casual"
        assert level.value == 3


# ============================================================================
# OUTFIT NODE TESTS
# ============================================================================


class TestOutfitNode:
    """Tests for OutfitNode dataclass."""
    
    def test_outfit_node_creation(self):
        """Test OutfitNode creation."""
        outfit = OutfitNode(
            id="o1",
            user_id="user1"
        )
        
        assert outfit.id == "o1"
        assert outfit.user_id == "user1"
    
    def test_outfit_node_with_details(self):
        """Test OutfitNode with details."""
        outfit = OutfitNode(
            id="o1",
            user_id="user1",
            occasion="CASUAL",
            season="SUMMER",
            overall_score=0.85
        )
        
        assert outfit.occasion == "CASUAL"
        assert outfit.season == "SUMMER"
        assert outfit.overall_score == 0.85


# ============================================================================
# USER NODE TESTS
# ============================================================================


class TestUserNode:
    """Tests for UserNode dataclass."""
    
    def test_user_node_creation(self):
        """Test UserNode creation."""
        user = UserNode(id="user1")
        
        assert user.id == "user1"
        assert user.profile == "DEFAULT"
    
    def test_user_node_with_profile(self):
        """Test UserNode with profile."""
        user = UserNode(
            id="user1",
            profile="MINIMALIST",
            wardrobe_size=50
        )
        
        assert user.profile == "MINIMALIST"
        assert user.wardrobe_size == 50


# ============================================================================
# RELATIONSHIP TESTS
# ============================================================================


class TestCompatibleWithRelationship:
    """Tests for CompatibleWithRelationship."""
    
    def test_compatible_with_creation(self):
        """Test CompatibleWithRelationship creation."""
        rel = CompatibleWithRelationship(weight=0.85)
        
        assert rel.weight == 0.85
        assert rel.pattern_compatible is True
    
    def test_compatible_with_full_details(self):
        """Test CompatibleWithRelationship with all details."""
        rel = CompatibleWithRelationship(
            weight=0.9,
            occasions=["CASUAL", "BUSINESS"],
            seasons=["SPRING", "SUMMER"],
            color_harmony=0.8,
            volume_balance_score=0.75
        )
        
        assert rel.weight == 0.9
        assert "CASUAL" in rel.occasions
        assert rel.color_harmony == 0.8
    
    def test_compatible_with_to_neo4j_properties(self):
        """Test CompatibleWithRelationship.to_neo4j_properties()."""
        rel = CompatibleWithRelationship(
            weight=0.85,
            color_harmony=0.8
        )
        
        props = rel.to_neo4j_properties()
        
        assert props["weight"] == 0.85
        assert props["color_harmony"] == 0.8


class TestColorHarmonizesRelationship:
    """Tests for ColorHarmonizesRelationship."""
    
    def test_color_harmonizes_creation(self):
        """Test ColorHarmonizesRelationship creation."""
        rel = ColorHarmonizesRelationship(
            weight=0.9,
            harmony_type="COMPLEMENTS"
        )
        
        assert rel.weight == 0.9
        assert rel.harmony_type == "COMPLEMENTS"
    
    def test_color_harmonizes_types(self):
        """Test different harmony types."""
        types = ["COMPLEMENTS", "ANALOGOUS", "TRIADIC"]
        
        for harmony_type in types:
            rel = ColorHarmonizesRelationship(
                weight=0.8,
                harmony_type=harmony_type
            )
            assert rel.harmony_type == harmony_type


class TestSuitableForRelationship:
    """Tests for SuitableForRelationship."""
    
    def test_suitable_for_creation(self):
        """Test SuitableForRelationship creation."""
        rel = SuitableForRelationship(weight=0.85)
        
        assert rel.weight == 0.85
        assert rel.is_primary is False
    
    def test_suitable_for_primary(self):
        """Test SuitableForRelationship as primary."""
        rel = SuitableForRelationship(
            weight=0.95,
            is_primary=True
        )
        
        assert rel.is_primary is True


class TestContainsRelationship:
    """Tests for ContainsRelationship."""
    
    def test_contains_creation(self):
        """Test ContainsRelationship creation."""
        rel = ContainsRelationship(role="main", position=0)
        
        assert rel.role == "main"
        assert rel.position == 0


class TestWoreRelationship:
    """Tests for WoreRelationship."""
    
    def test_wore_creation(self):
        """Test WoreRelationship creation."""
        rel = WoreRelationship(worn_at="2024-01-15T10:00:00")
        
        assert rel.worn_at == "2024-01-15T10:00:00"
    
    def test_wore_with_outfit(self):
        """Test WoreRelationship with outfit reference."""
        rel = WoreRelationship(
            worn_at="2024-01-15T10:00:00",
            outfit_id="o1",
            rating=0.9
        )
        
        assert rel.outfit_id == "o1"
        assert rel.rating == 0.9


# ============================================================================
# GRAPH SCHEMA TESTS
# ============================================================================


class TestGraphSchema:
    """Tests for GraphSchema class."""
    
    def test_node_types(self):
        """Test NODE_TYPES list."""
        assert "Garment" in GraphSchema.NODE_TYPES
        assert "Season" in GraphSchema.NODE_TYPES
        assert "Color" in GraphSchema.NODE_TYPES
        assert "Outfit" in GraphSchema.NODE_TYPES
    
    def test_relationship_types(self):
        """Test RELATIONSHIP_TYPES list."""
        assert "COMPATIBLE_WITH" in GraphSchema.RELATIONSHIP_TYPES
        assert "SUITABLE_FOR" in GraphSchema.RELATIONSHIP_TYPES
        assert "HAS_COLOR" in GraphSchema.RELATIONSHIP_TYPES
    
    def test_indexes(self):
        """Test INDEXES list."""
        assert len(GraphSchema.INDEXES) > 0
        
        # Check some key indexes
        index_names = [idx[0] for idx in GraphSchema.INDEXES]
        assert "garment_id" in index_names
        assert "user_id" in index_names
    
    def test_get_node_properties(self):
        """Test get_node_properties method."""
        garment_props = GraphSchema.get_node_properties("Garment")
        
        assert "id" in garment_props
        assert "title" in garment_props
        assert "category" in garment_props
    
    def test_get_relationship_properties(self):
        """Test get_relationship_properties method."""
        compatible_props = GraphSchema.get_relationship_properties("COMPATIBLE_WITH")
        
        assert "weight" in compatible_props
        assert compatible_props["weight"] == "FLOAT"


# ============================================================================
# HELPER FUNCTIONS TESTS
# ============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_create_garment_node_from_dict(self):
        """Test create_garment_node_from_dict function."""
        data = {
            "id": "g1",
            "title": "Blue Jeans",
            "category": "BOTTOM",
            "color_primary": "blue",
            "rating": 0.9,
            "extra_field": "ignored"
        }
        
        garment = create_garment_node_from_dict(data)
        
        assert garment.id == "g1"
        assert garment.title == "Blue Jeans"
        assert garment.color_primary == "blue"
        assert garment.rating == 0.9
    
    def test_create_compatibility_from_scores(self):
        """Test create_compatibility_from_scores function."""
        scores = {
            "weight": 0.85,
            "color_harmony": 0.8,
            "pattern_compatible": True,
            "volume_balance": 0.75
        }
        
        rel = create_compatibility_from_scores(scores)
        
        assert rel.weight == 0.85
        assert rel.color_harmony == 0.8
        assert rel.pattern_compatible is True
        assert rel.volume_balance_score == 0.75
    
    def test_create_compatibility_from_scores_partial(self):
        """Test create_compatibility_from_scores with partial scores."""
        scores = {"weight": 0.9}
        
        rel = create_compatibility_from_scores(scores)
        
        assert rel.weight == 0.9
        assert rel.color_harmony == 0.5  # default
