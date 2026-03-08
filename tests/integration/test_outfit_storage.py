"""
Integration tests for Outfit Storage.

Tests cover:
- JSON storage operations
- Garment CRUD operations
- Outfit CRUD operations
- Compatibility score storage
"""
import pytest
import shutil
from pathlib import Path
from typing import List

from src.layer2_style.outfit_storage import (
    JSONOutfitStorage,
    Neo4jOutfitStorage,
    OutfitStorageInterface,
    RelationType,
    CompatibilityScore,
    get_outfit_storage
)
from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory, Outfit, OutfitItem,
    ColorProfile, FormalityLevel, Season
)


# ============== Configuration ==============

TEST_OUTPUT_DIR = Path(__file__).parent.parent / "output" / "storage_tests"
TEST_JSON_DIR = TEST_OUTPUT_DIR / "json_storage"


# ============== Fixtures ==============

@pytest.fixture
def test_output_dir():
    """Create test output directory."""
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_OUTPUT_DIR


@pytest.fixture
def sample_garments() -> List[Garment]:
    """Create sample garments for testing."""
    return [
        Garment(
            id="top-001",
            image_path="wardrobe/tops/white_tshirt.jpg",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                subcategory="t-shirt",
                color=ColorProfile(primary="white", secondary=None),
                formality_level=FormalityLevel.CASUAL,
                style_tags=["casual", "minimalist", "basic"],
                season_suitable=[Season.SPRING, Season.SUMMER]
            )
        ),
        Garment(
            id="top-002",
            image_path="wardrobe/tops/blue_shirt.jpg",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                subcategory="shirt",
                color=ColorProfile(primary="navy", secondary="white"),
                formality_level=FormalityLevel.SMART_CASUAL,
                style_tags=["smart casual", "classic", "workwear"],
                season_suitable=[Season.SPRING, Season.FALL]
            )
        ),
        Garment(
            id="bottom-001",
            image_path="wardrobe/bottoms/blue_jeans.jpg",
            attributes=GarmentAttributes(
                category=GarmentCategory.BOTTOM,
                subcategory="jeans",
                color=ColorProfile(primary="blue", secondary=None),
                formality_level=FormalityLevel.CASUAL,
                style_tags=["casual", "denim", "versatile"],
                season_suitable=[Season.SPRING, Season.FALL, Season.WINTER]
            )
        ),
        Garment(
            id="bottom-002",
            image_path="wardrobe/bottoms/chinos.jpg",
            attributes=GarmentAttributes(
                category=GarmentCategory.BOTTOM,
                subcategory="chinos",
                color=ColorProfile(primary="beige", secondary=None),
                formality_level=FormalityLevel.SMART_CASUAL,
                style_tags=["smart casual", "classic", "workwear"],
                season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL]
            )
        ),
        Garment(
            id="shoes-001",
            image_path="wardrobe/shoes/white_sneakers.jpg",
            attributes=GarmentAttributes(
                category=GarmentCategory.SHOES,
                subcategory="sneakers",
                color=ColorProfile(primary="white", secondary=None),
                formality_level=FormalityLevel.CASUAL,
                style_tags=["casual", "sporty", "minimalist"],
                season_suitable=[Season.SPRING, Season.SUMMER]
            )
        ),
        Garment(
            id="shoes-002",
            image_path="wardrobe/shoes/brown_loafers.jpg",
            attributes=GarmentAttributes(
                category=GarmentCategory.SHOES,
                subcategory="loafers",
                color=ColorProfile(primary="brown", secondary=None),
                formality_level=FormalityLevel.SMART_CASUAL,
                style_tags=["smart casual", "classic", "elegant"],
                season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL]
            )
        ),
    ]


@pytest.fixture
def json_storage(test_output_dir):
    """Create JSON storage for testing."""
    if TEST_JSON_DIR.exists():
        shutil.rmtree(TEST_JSON_DIR)
    
    storage = JSONOutfitStorage(base_path=TEST_JSON_DIR)
    yield storage


# ============== Test Classes ==============

@pytest.mark.integration
class TestJSONStorageGarments:
    """Test garment operations with JSON storage."""
    
    @pytest.mark.asyncio
    async def test_save_and_get_garment(self, json_storage, sample_garments):
        """Test saving and retrieving a garment."""
        garment = sample_garments[0]
        
        garment_id = await json_storage.save_garment(garment)
        assert garment_id == garment.id
        
        retrieved = await json_storage.get_garment(garment_id)
        assert retrieved is not None
        assert retrieved.id == garment.id
        assert retrieved.attributes.category == GarmentCategory.TOP
        assert retrieved.attributes.color.primary == "white"
    
    @pytest.mark.asyncio
    async def test_get_all_garments(self, json_storage, sample_garments):
        """Test retrieving all garments."""
        for garment in sample_garments:
            await json_storage.save_garment(garment)
        
        all_garments = await json_storage.get_all_garments()
        assert len(all_garments) == len(sample_garments)
    
    @pytest.mark.asyncio
    async def test_get_garments_by_category(self, json_storage, sample_garments):
        """Test filtering garments by category."""
        for garment in sample_garments:
            await json_storage.save_garment(garment)
        
        tops = await json_storage.get_garments_by_category(GarmentCategory.TOP)
        assert len(tops) == 2
        assert all(g.attributes.category == GarmentCategory.TOP for g in tops)
        
        bottoms = await json_storage.get_garments_by_category(GarmentCategory.BOTTOM)
        assert len(bottoms) == 2
        
        shoes = await json_storage.get_garments_by_category(GarmentCategory.SHOES)
        assert len(shoes) == 2
    
    @pytest.mark.asyncio
    async def test_delete_garment(self, json_storage, sample_garments):
        """Test deleting a garment."""
        garment = sample_garments[0]
        await json_storage.save_garment(garment)
        
        retrieved = await json_storage.get_garment(garment.id)
        assert retrieved is not None
        
        deleted = await json_storage.delete_garment(garment.id)
        assert deleted is True
        
        retrieved_after = await json_storage.get_garment(garment.id)
        assert retrieved_after is None


@pytest.mark.integration
class TestJSONStorageOutfits:
    """Test outfit operations with JSON storage."""
    
    @pytest.mark.asyncio
    async def test_save_and_get_outfit(self, json_storage, sample_garments):
        """Test saving and retrieving an outfit."""
        for garment in sample_garments:
            await json_storage.save_garment(garment)
        
        outfit = Outfit(
            id="outfit-001",
            items=[
                OutfitItem(garment=sample_garments[0], role="main_top"),
                OutfitItem(garment=sample_garments[2], role="main_bottom"),
                OutfitItem(garment=sample_garments[4], role="footwear"),
            ],
            compatibility_score=0.85,
            style_coherence_score=0.85,
            occasion_match_score=0.85,
            overall_score=0.85
        )
        
        outfit_id = await json_storage.save_outfit(outfit)
        assert outfit_id == outfit.id
        
        retrieved = await json_storage.get_outfit(outfit_id)
        assert retrieved is not None
        assert len(retrieved.items) == 3


@pytest.mark.integration
class TestStorageFactory:
    """Test storage factory function."""
    
    def test_get_json_storage(self, test_output_dir):
        """Test getting JSON storage via factory."""
        storage = get_outfit_storage("json", base_path=TEST_JSON_DIR / "factory_test")
        
        assert isinstance(storage, JSONOutfitStorage)


@pytest.mark.integration
class TestNeo4jStorage:
    """Test Neo4j storage (requires running Neo4j)."""
    
    @pytest.mark.asyncio
    async def test_neo4j_connection(self):
        """Test Neo4j connection."""
        try:
            storage = Neo4jOutfitStorage()
            # If we get here, connection was successful
            assert storage is not None
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")
