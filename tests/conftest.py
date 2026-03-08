"""
Shared pytest fixtures and configuration.

This module provides reusable fixtures for all test modules:
- Mock data factories
- Common fixtures (garments, outfits, contexts)
- Test utilities
- Pytest markers configuration
"""
import pytest
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
from uuid import uuid4

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory, Outfit, OutfitItem,
    ColorInfo, PatternInfo, FormalityLevel, Season,
    UserContext, WeatherContext, Occasion,
    MaterialInfo, MaterialProfile, ColorProfile,
    NecklineType, LengthType, SleeveType
)


# ============== Paths ==============

TEST_ROOT = Path(__file__).parent
TEST_IMAGES_DIR = TEST_ROOT / "test_images"
TEST_OUTPUT_DIR = TEST_ROOT / "output"


# ============== Pytest Configuration ==============

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (may use real APIs)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (full pipeline)")
    config.addinivalue_line("markers", "slow: Slow tests that should be skipped in quick runs")
    config.addinivalue_line("markers", "api: API route tests")


# ============== Factory Functions ==============

class GarmentFactory:
    """Factory for creating test garments."""
    
    @staticmethod
    def create_basic_top(
        garment_id: Optional[str] = None,
        color: str = "white",
        formality: FormalityLevel = FormalityLevel.CASUAL,
        **overrides
    ) -> Garment:
        """Create a basic top garment."""
        attrs = GarmentAttributes(
            category=GarmentCategory.TOP,
            subcategory="t-shirt",
            color=ColorInfo(primary=color, hex_codes=[]),
            style_tags=["casual", "basic"],
            formality_level=formality,
            season_suitable=[Season.SPRING, Season.SUMMER],
            **{k: v for k, v in overrides.items() if k in GarmentAttributes.__fields__}
        )
        return Garment(
            id=garment_id or f"top_{uuid4().hex[:8]}",
            attributes=attrs
        )
    
    @staticmethod
    def create_basic_bottom(
        garment_id: Optional[str] = None,
        color: str = "blue",
        formality: FormalityLevel = FormalityLevel.CASUAL,
        **overrides
    ) -> Garment:
        """Create a basic bottom garment (jeans)."""
        attrs = GarmentAttributes(
            category=GarmentCategory.BOTTOM,
            subcategory="jeans",
            color=ColorInfo(primary=color, hex_codes=[]),
            style_tags=["casual", "denim"],
            formality_level=formality,
            season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL],
            **{k: v for k, v in overrides.items() if k in GarmentAttributes.__fields__}
        )
        return Garment(
            id=garment_id or f"bottom_{uuid4().hex[:8]}",
            attributes=attrs
        )
    
    @staticmethod
    def create_dress_shirt(
        garment_id: Optional[str] = None,
        color: str = "white",
        **overrides
    ) -> Garment:
        """Create a business dress shirt."""
        attrs = GarmentAttributes(
            category=GarmentCategory.TOP,
            subcategory="dress_shirt",
            color=ColorInfo(primary=color, hex_codes=[]),
            style_tags=["professional", "classic"],
            formality_level=FormalityLevel.BUSINESS,
            season_suitable=[Season.SPRING, Season.FALL, Season.WINTER],
            **{k: v for k, v in overrides.items() if k in GarmentAttributes.__fields__}
        )
        return Garment(
            id=garment_id or f"dress_shirt_{uuid4().hex[:8]}",
            attributes=attrs
        )
    
    @staticmethod
    def create_dress_pants(
        garment_id: Optional[str] = None,
        color: str = "navy",
        **overrides
    ) -> Garment:
        """Create business dress pants."""
        attrs = GarmentAttributes(
            category=GarmentCategory.BOTTOM,
            subcategory="trousers",
            color=ColorInfo(primary=color, hex_codes=[]),
            style_tags=["professional", "tailored"],
            formality_level=FormalityLevel.BUSINESS,
            season_suitable=[Season.SPRING, Season.FALL, Season.WINTER],
            **{k: v for k, v in overrides.items() if k in GarmentAttributes.__fields__}
        )
        return Garment(
            id=garment_id or f"trousers_{uuid4().hex[:8]}",
            attributes=attrs
        )
    
    @staticmethod
    def create_shoes(
        garment_id: Optional[str] = None,
        color: str = "black",
        subcategory: str = "sneakers",
        formality: FormalityLevel = FormalityLevel.CASUAL,
        **overrides
    ) -> Garment:
        """Create shoes."""
        attrs = GarmentAttributes(
            category=GarmentCategory.SHOES,
            subcategory=subcategory,
            color=ColorInfo(primary=color, hex_codes=[]),
            style_tags=["casual" if formality == FormalityLevel.CASUAL else "formal"],
            formality_level=formality,
            season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
            **{k: v for k, v in overrides.items() if k in GarmentAttributes.__fields__}
        )
        return Garment(
            id=garment_id or f"shoes_{uuid4().hex[:8]}",
            attributes=attrs
        )
    
    @staticmethod
    def create_outerwear(
        garment_id: Optional[str] = None,
        color: str = "black",
        subcategory: str = "jacket",
        **overrides
    ) -> Garment:
        """Create outerwear."""
        attrs = GarmentAttributes(
            category=GarmentCategory.OUTERWEAR,
            subcategory=subcategory,
            color=ColorInfo(primary=color, hex_codes=[]),
            style_tags=["layering", "versatile"],
            formality_level=FormalityLevel.SMART_CASUAL,
            season_suitable=[Season.FALL, Season.WINTER],
            **{k: v for k, v in overrides.items() if k in GarmentAttributes.__fields__}
        )
        return Garment(
            id=garment_id or f"outerwear_{uuid4().hex[:8]}",
            attributes=attrs
        )
    
    @staticmethod
    def create_accessory(
        garment_id: Optional[str] = None,
        subcategory: str = "watch",
        color: str = "silver",
        **overrides
    ) -> Garment:
        """Create an accessory."""
        attrs = GarmentAttributes(
            category=GarmentCategory.ACCESSORY,
            subcategory=subcategory,
            color=ColorInfo(primary=color, hex_codes=[]),
            style_tags=["accessory", "detail"],
            formality_level=FormalityLevel.SMART_CASUAL,
            season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
            **{k: v for k, v in overrides.items() if k in GarmentAttributes.__fields__}
        )
        return Garment(
            id=garment_id or f"accessory_{uuid4().hex[:8]}",
            attributes=attrs
        )


class OutfitFactory:
    """Factory for creating test outfits."""
    
    @staticmethod
    def create_casual_outfit(outfit_id: Optional[str] = None) -> Outfit:
        """Create a casual outfit with top, bottom, and shoes."""
        items = [
            OutfitItem(
                garment=GarmentFactory.create_basic_top(color="white"),
                role="main_top"
            ),
            OutfitItem(
                garment=GarmentFactory.create_basic_bottom(color="blue"),
                role="main_bottom"
            ),
            OutfitItem(
                garment=GarmentFactory.create_shoes(color="white", subcategory="sneakers"),
                role="footwear"
            ),
        ]
        return Outfit(
            id=outfit_id or f"outfit_{uuid4().hex[:8]}",
            items=items,
            compatibility_score=0.8,
            style_coherence_score=0.8,
            occasion_match_score=0.8,
            overall_score=0.8
        )
    
    @staticmethod
    def create_business_outfit(outfit_id: Optional[str] = None) -> Outfit:
        """Create a business outfit."""
        items = [
            OutfitItem(
                garment=GarmentFactory.create_dress_shirt(color="white"),
                role="main_top"
            ),
            OutfitItem(
                garment=GarmentFactory.create_dress_pants(color="navy"),
                role="main_bottom"
            ),
            OutfitItem(
                garment=GarmentFactory.create_shoes(
                    color="brown", subcategory="oxford", formality=FormalityLevel.BUSINESS
                ),
                role="footwear"
            ),
        ]
        return Outfit(
            id=outfit_id or f"outfit_{uuid4().hex[:8]}",
            items=items,
            compatibility_score=0.85,
            style_coherence_score=0.85,
            occasion_match_score=0.9,
            overall_score=0.87
        )
    
    @staticmethod
    def create_formal_outfit(outfit_id: Optional[str] = None) -> Outfit:
        """Create a formal outfit."""
        items = [
            OutfitItem(
                garment=GarmentFactory.create_dress_shirt(color="white"),
                role="main_top"
            ),
            OutfitItem(
                garment=GarmentFactory.create_dress_pants(color="black"),
                role="main_bottom"
            ),
            OutfitItem(
                garment=GarmentFactory.create_shoes(
                    color="black", subcategory="dress_shoes", formality=FormalityLevel.FORMAL
                ),
                role="footwear"
            ),
            OutfitItem(
                garment=GarmentFactory.create_outerwear(
                    color="black", subcategory="blazer"
                ),
                role="layering"
            ),
        ]
        return Outfit(
            id=outfit_id or f"outfit_{uuid4().hex[:8]}",
            items=items,
            compatibility_score=0.9,
            style_coherence_score=0.9,
            occasion_match_score=0.95,
            overall_score=0.92
        )


class ContextFactory:
    """Factory for creating test user contexts."""
    
    @staticmethod
    def create_casual_context(user_id: Optional[str] = None) -> UserContext:
        """Create a casual context."""
        return UserContext(
            user_id=user_id or f"user_{uuid4().hex[:8]}",
            occasion=Occasion.CASUAL,
            body_shape="rectangle",
            style_preferences=["minimalist", "casual"],
            color_preferences=["neutral", "blue"],
            skin_undertone="warm",
            color_season="autumn"
        )
    
    @staticmethod
    def create_business_context(user_id: Optional[str] = None) -> UserContext:
        """Create a business context."""
        return UserContext(
            user_id=user_id or f"user_{uuid4().hex[:8]}",
            occasion=Occasion.BUSINESS,
            body_shape="rectangle",
            style_preferences=["classic", "professional"],
            color_preferences=["navy", "white", "gray"],
            skin_undertone="cool",
            color_season="summer"
        )
    
    @staticmethod
    def create_formal_context(user_id: Optional[str] = None) -> UserContext:
        """Create a formal context."""
        return UserContext(
            user_id=user_id or f"user_{uuid4().hex[:8]}",
            occasion=Occasion.FORMAL,
            body_shape="triangle",
            style_preferences=["elegant", "refined"],
            color_preferences=["black", "white", "silver"],
            skin_undertone="neutral",
            color_season="winter"
        )
    
    @staticmethod
    def create_weather_context(
        temperature: float = 20.0,
        condition: str = "clear"
    ) -> WeatherContext:
        """Create a weather context."""
        return WeatherContext(
            temperature=temperature,
            condition=condition,
            humidity=50.0,
            wind_speed=10.0
        )


# ============== Common Fixtures ==============

@pytest.fixture
def garment_factory():
    """Provide garment factory for tests."""
    return GarmentFactory


@pytest.fixture
def outfit_factory():
    """Provide outfit factory for tests."""
    return OutfitFactory


@pytest.fixture
def context_factory():
    """Provide context factory for tests."""
    return ContextFactory


@pytest.fixture
def casual_top():
    """Create a casual white t-shirt."""
    return GarmentFactory.create_basic_top(color="white")


@pytest.fixture
def casual_bottom():
    """Create casual blue jeans."""
    return GarmentFactory.create_basic_bottom(color="blue")


@pytest.fixture
def casual_shoes():
    """Create casual white sneakers."""
    return GarmentFactory.create_shoes(color="white", subcategory="sneakers")


@pytest.fixture
def casual_outfit():
    """Create a complete casual outfit."""
    return [
        GarmentFactory.create_basic_top(color="white"),
        GarmentFactory.create_basic_bottom(color="blue"),
        GarmentFactory.create_shoes(color="white", subcategory="sneakers"),
    ]


@pytest.fixture
def business_outfit():
    """Create a complete business outfit."""
    return [
        GarmentFactory.create_dress_shirt(color="white"),
        GarmentFactory.create_dress_pants(color="navy"),
        GarmentFactory.create_shoes(
            color="brown", subcategory="oxford", formality=FormalityLevel.BUSINESS
        ),
    ]


@pytest.fixture
def casual_context():
    """Create a casual user context."""
    return ContextFactory.create_casual_context()


@pytest.fixture
def business_context():
    """Create a business user context."""
    return ContextFactory.create_business_context()


@pytest.fixture
def test_images_dir():
    """Get test images directory."""
    return TEST_IMAGES_DIR


@pytest.fixture
def output_dir():
    """Get and ensure output directory exists."""
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_OUTPUT_DIR


# ============== Mock Fixtures ==============

@pytest.fixture
def mock_gemini_response():
    """Create a mock Gemini API response."""
    def _create_response(content: str):
        mock = MagicMock()
        mock.text = content
        return mock
    return _create_response


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    def _create_response(content: str):
        mock = MagicMock()
        mock.choices = [MagicMock(message=MagicMock(content=content))]
        return mock
    return _create_response


@pytest.fixture
def mock_segmentation_result():
    """Create a standard mock segmentation result."""
    return [
        {
            "item_number": 1,
            "category": "top",
            "specific_type": "t-shirt",
            "position": "upper_body",
            "visibility": "full"
        },
        {
            "item_number": 2,
            "category": "bottom",
            "specific_type": "jeans",
            "position": "lower_body",
            "visibility": "full"
        },
        {
            "item_number": 3,
            "category": "shoes",
            "specific_type": "sneakers",
            "position": "feet",
            "visibility": "partial"
        }
    ]


# ============== Utility Functions ==============

def assert_score_in_range(score: float, min_val: float = 0.0, max_val: float = 1.0):
    """Assert that a score is within a valid range."""
    assert min_val <= score <= max_val, f"Score {score} not in range [{min_val}, {max_val}]"


def assert_valid_garment(garment: Garment):
    """Assert that a garment has all required fields."""
    assert garment.id is not None
    assert garment.attributes is not None
    assert garment.attributes.category is not None
    assert garment.attributes.color is not None


def assert_valid_outfit(outfit: Outfit):
    """Assert that an outfit is valid."""
    assert outfit.id is not None
    assert outfit.items is not None
    assert len(outfit.items) > 0
    for item in outfit.items:
        assert_valid_garment(item.garment)


# Export utilities
__all__ = [
    'GarmentFactory',
    'OutfitFactory', 
    'ContextFactory',
    'TEST_ROOT',
    'TEST_IMAGES_DIR',
    'TEST_OUTPUT_DIR',
    'assert_score_in_range',
    'assert_valid_garment',
    'assert_valid_outfit',
]
