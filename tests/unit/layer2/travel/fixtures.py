"""Shared garment fixtures for travel test suite."""
from uuid import uuid4
from typing import List

from src.core.models import (
    Garment,
    GarmentAttributes,
    GarmentCategory,
    ColorProfile,
    FormalityLevel,
    Season,
    PatternInfo,
    MaterialProfile,
    SeasonalityInfo,
)


def _g(
    garment_id=None,
    category=GarmentCategory.TOP,
    subcategory="t-shirt",
    color="white",
    formality=FormalityLevel.CASUAL,
    material="cotton",
    seasons=None,
    style_tags=None,
) -> Garment:
    season_list = seasons or [Season.SPRING, Season.SUMMER]
    return Garment(
        id=garment_id or f"g_{uuid4().hex[:8]}",
        attributes=GarmentAttributes(
            category=category,
            subcategory=subcategory,
            color=ColorProfile(primary=color, hex_codes=[]),
            formality_level=formality,
            season_suitable=season_list,
            seasonality=SeasonalityInfo(seasons=season_list),
            pattern=PatternInfo(type="solid"),
            material=MaterialProfile(primary=material),
            style_tags=style_tags or ["casual"],
        ),
    )


def make_travel_wardrobe() -> List[Garment]:
    """A balanced 12-piece wardrobe for travel tests."""
    return [
        # Tops
        _g("t1", GarmentCategory.TOP, "t-shirt", "white", FormalityLevel.CASUAL, "cotton",
           [Season.SPRING, Season.SUMMER], ["casual", "relaxed"]),
        _g("t2", GarmentCategory.TOP, "shirt", "navy", FormalityLevel.SMART_CASUAL, "cotton",
           [Season.SPRING, Season.FALL], ["smart_casual", "professional"]),
        _g("t3", GarmentCategory.TOP, "blouse", "black", FormalityLevel.BUSINESS, "silk",
           [Season.FALL, Season.WINTER], ["professional", "elegant"]),
        # Bottoms
        _g("b1", GarmentCategory.BOTTOM, "jeans", "blue", FormalityLevel.CASUAL, "denim",
           [Season.SPRING, Season.FALL], ["casual", "relaxed"]),
        _g("b2", GarmentCategory.BOTTOM, "trousers", "grey", FormalityLevel.BUSINESS, "wool",
           [Season.FALL, Season.WINTER], ["professional", "formal"]),
        _g("b3", GarmentCategory.BOTTOM, "shorts", "beige", FormalityLevel.CASUAL, "cotton",
           [Season.SPRING, Season.SUMMER], ["casual", "beach"]),
        # Dress
        _g("d1", GarmentCategory.DRESS, "midi dress", "rust", FormalityLevel.SMART_CASUAL, "linen",
           [Season.SPRING, Season.SUMMER], ["casual", "evening"]),
        # Outerwear
        _g("o1", GarmentCategory.OUTERWEAR, "jacket", "camel", FormalityLevel.SMART_CASUAL, "wool",
           [Season.FALL, Season.WINTER], ["casual", "layering"]),
        _g("o2", GarmentCategory.OUTERWEAR, "blazer", "black", FormalityLevel.BUSINESS, "cotton",
           [Season.FALL, Season.WINTER], ["professional", "formal"]),
        # Shoes
        _g("s1", GarmentCategory.SHOES, "sneakers", "white", FormalityLevel.CASUAL, "leather",
           [Season.SPRING, Season.SUMMER], ["casual"]),
        _g("s2", GarmentCategory.SHOES, "loafers", "brown", FormalityLevel.SMART_CASUAL, "leather",
           [Season.FALL, Season.WINTER], ["professional"]),
        # Accessory
        _g("a1", GarmentCategory.ACCESSORY, "scarf", "olive", FormalityLevel.CASUAL, "wool",
           [Season.FALL, Season.WINTER], ["casual"]),
    ]
