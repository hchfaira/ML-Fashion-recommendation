"""
Garment Taxonomy for Layer 0 Pipeline
======================================

Defines the standardized garment taxonomy used throughout the pipeline:
- GroundingDINO text prompt generation
- SCHP label mapping
- Output file naming
- Garment classification
"""

from typing import Dict, List, Set
from enum import Enum


# ============================================================================
# GARMENT TAXONOMY - The source of truth for all garment categories
# ============================================================================

GARMENT_TAXONOMY: Dict[str, List[str]] = {
    "tops": [
        "t-shirt", "shirt", "blouse", "sweater", "hoodie", "tank top",
        "crop top", "cardigan", "blazer", "vest"
    ],
    "bottoms": [
        "pants", "jeans", "shorts", "skirt", "leggings"
    ],
    "full_body": [
        "dress", "jumpsuit", "romper"
    ],
    "outerwear": [
        "jacket", "coat", "parka", "trench coat"
    ],
    "footwear": [
        "sneakers", "boots", "sandals", "heels", "loafers"
    ],
    "accessories": [
        "bag", "handbag", "backpack", "belt", "scarf", "hat", "cap", "sunglasses"
    ]
}


class GarmentCategory(str, Enum):
    """Enum for garment categories from taxonomy."""
    TOPS = "tops"
    BOTTOMS = "bottoms"
    FULL_BODY = "full_body"
    OUTERWEAR = "outerwear"
    FOOTWEAR = "footwear"
    ACCESSORIES = "accessories"
    UNKNOWN = "unknown"


# ============================================================================
# SCHP Label Mapping (LIP dataset labels)
# ============================================================================

SCHP_LABELS: Dict[int, str] = {
    0: "background",
    1: "hat",
    2: "hair",
    3: "glove",
    4: "sunglasses",
    5: "upper_clothes",
    6: "dress",
    7: "coat",
    8: "socks",
    9: "pants",
    10: "jumpsuits",
    11: "scarf",
    12: "skirt",
    13: "face",
    14: "left_arm",
    15: "right_arm",
    16: "left_leg",
    17: "right_leg",
    18: "left_shoe",
    19: "right_shoe",
}


# Mapping SCHP labels to taxonomy categories
SCHP_TO_TAXONOMY: Dict[int, GarmentCategory] = {
    1: GarmentCategory.ACCESSORIES,    # hat
    3: GarmentCategory.ACCESSORIES,    # glove
    4: GarmentCategory.ACCESSORIES,    # sunglasses
    5: GarmentCategory.TOPS,           # upper_clothes
    6: GarmentCategory.FULL_BODY,      # dress
    7: GarmentCategory.OUTERWEAR,      # coat
    8: GarmentCategory.ACCESSORIES,    # socks
    9: GarmentCategory.BOTTOMS,        # pants
    10: GarmentCategory.FULL_BODY,     # jumpsuits
    11: GarmentCategory.ACCESSORIES,   # scarf
    12: GarmentCategory.BOTTOMS,       # skirt
    18: GarmentCategory.FOOTWEAR,      # left_shoe
    19: GarmentCategory.FOOTWEAR,      # right_shoe
}

# SCHP labels that represent garments (not background/body parts)
GARMENT_SCHP_LABELS: Set[int] = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19}


# ============================================================================
# Prompt Generation for GroundingDINO
# ============================================================================

def generate_detection_prompt(categories: List[str] = None) -> str:
    """
    Generate a GroundingDINO text prompt from taxonomy.
    
    Args:
        categories: List of category names to include. If None, includes all.
        
    Returns:
        Formatted prompt string for GroundingDINO detection
    """
    if categories is None:
        categories = list(GARMENT_TAXONOMY.keys())
    
    all_items = []
    for category in categories:
        if category in GARMENT_TAXONOMY:
            all_items.extend(GARMENT_TAXONOMY[category])
    
    # GroundingDINO expects period-separated prompts
    return " . ".join(all_items)


def generate_category_prompt(category: str) -> str:
    """
    Generate a GroundingDINO prompt for a specific category.
    
    Args:
        category: Category name from taxonomy
        
    Returns:
        Formatted prompt string for the category
    """
    if category not in GARMENT_TAXONOMY:
        raise ValueError(f"Unknown category: {category}. Valid: {list(GARMENT_TAXONOMY.keys())}")
    
    return " . ".join(GARMENT_TAXONOMY[category])


def get_category_from_label(label: str) -> GarmentCategory:
    """
    Determine the taxonomy category from a detection label.
    
    Args:
        label: Detection label (e.g., "t-shirt", "jeans")
        
    Returns:
        GarmentCategory enum value
    """
    label_lower = label.lower().strip()
    
    for category, items in GARMENT_TAXONOMY.items():
        for item in items:
            if item in label_lower or label_lower in item:
                return GarmentCategory(category)
    
    return GarmentCategory.UNKNOWN


def get_all_garment_labels() -> List[str]:
    """
    Get a flat list of all garment labels from taxonomy.
    
    Returns:
        List of all garment label strings
    """
    all_labels = []
    for items in GARMENT_TAXONOMY.values():
        all_labels.extend(items)
    return all_labels


def get_schp_category(label_id: int) -> GarmentCategory:
    """
    Get taxonomy category from SCHP label ID.
    
    Args:
        label_id: SCHP label ID (0-19)
        
    Returns:
        GarmentCategory enum value
    """
    return SCHP_TO_TAXONOMY.get(label_id, GarmentCategory.UNKNOWN)


def get_schp_label_name(label_id: int) -> str:
    """
    Get human-readable name for SCHP label ID.
    
    Args:
        label_id: SCHP label ID (0-19)
        
    Returns:
        Human-readable label name
    """
    return SCHP_LABELS.get(label_id, "unknown")


def is_garment_schp_label(label_id: int) -> bool:
    """
    Check if SCHP label ID represents a garment (not body part/background).
    
    Args:
        label_id: SCHP label ID
        
    Returns:
        True if label represents a garment
    """
    return label_id in GARMENT_SCHP_LABELS
