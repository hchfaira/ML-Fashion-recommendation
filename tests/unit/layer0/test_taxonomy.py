"""
Tests for Layer 0: Garment Taxonomy
"""

import pytest


class TestGarmentTaxonomy:
    """Tests for the GARMENT_TAXONOMY and related utilities."""
    
    def test_taxonomy_contains_all_categories(self):
        """Test that taxonomy has all expected categories."""
        from src.layer0_segmentation import GARMENT_TAXONOMY
        
        expected_categories = [
            "tops", "bottoms", "full_body", 
            "outerwear", "footwear", "accessories"
        ]
        
        for category in expected_categories:
            assert category in GARMENT_TAXONOMY
    
    def test_taxonomy_tops_contains_expected_items(self):
        """Test that tops category has expected items."""
        from src.layer0_segmentation import GARMENT_TAXONOMY
        
        tops = GARMENT_TAXONOMY["tops"]
        assert "t-shirt" in tops
        assert "shirt" in tops
        assert "sweater" in tops
        assert "hoodie" in tops
    
    def test_taxonomy_bottoms_contains_expected_items(self):
        """Test that bottoms category has expected items."""
        from src.layer0_segmentation import GARMENT_TAXONOMY
        
        bottoms = GARMENT_TAXONOMY["bottoms"]
        assert "pants" in bottoms
        assert "jeans" in bottoms
        assert "shorts" in bottoms
        assert "skirt" in bottoms
    
    def test_taxonomy_full_body_contains_expected_items(self):
        """Test that full_body category has expected items."""
        from src.layer0_segmentation import GARMENT_TAXONOMY
        
        full_body = GARMENT_TAXONOMY["full_body"]
        assert "dress" in full_body
        assert "jumpsuit" in full_body
        assert "romper" in full_body


class TestGarmentCategory:
    """Tests for GarmentCategory enum."""
    
    def test_all_categories_exist(self):
        """Test that all expected categories exist."""
        from src.layer0_segmentation import GarmentCategory
        
        assert GarmentCategory.TOPS.value == "tops"
        assert GarmentCategory.BOTTOMS.value == "bottoms"
        assert GarmentCategory.FULL_BODY.value == "full_body"
        assert GarmentCategory.OUTERWEAR.value == "outerwear"
        assert GarmentCategory.FOOTWEAR.value == "footwear"
        assert GarmentCategory.ACCESSORIES.value == "accessories"
        assert GarmentCategory.UNKNOWN.value == "unknown"


class TestPromptGeneration:
    """Tests for detection prompt generation."""
    
    def test_generate_detection_prompt_all(self):
        """Test generating prompt for all categories."""
        from src.layer0_segmentation import generate_detection_prompt
        
        prompt = generate_detection_prompt()
        
        # Should contain items from taxonomy
        assert "t-shirt" in prompt
        assert "pants" in prompt
        assert "dress" in prompt
        assert "jacket" in prompt
        assert "sneakers" in prompt
        assert "bag" in prompt
        
        # Items should be separated by " . "
        assert " . " in prompt
    
    def test_generate_detection_prompt_specific_categories(self):
        """Test generating prompt for specific categories."""
        from src.layer0_segmentation import generate_detection_prompt
        
        prompt = generate_detection_prompt(["tops", "bottoms"])
        
        assert "t-shirt" in prompt
        assert "pants" in prompt
        assert "dress" not in prompt  # full_body not included
        assert "jacket" not in prompt  # outerwear not included
    
    def test_generate_category_prompt(self):
        """Test generating prompt for single category."""
        from src.layer0_segmentation import generate_category_prompt
        
        tops_prompt = generate_category_prompt("tops")
        
        assert "t-shirt" in tops_prompt
        assert "sweater" in tops_prompt
        assert "pants" not in tops_prompt
    
    def test_generate_category_prompt_invalid_raises(self):
        """Test that invalid category raises error."""
        from src.layer0_segmentation import generate_category_prompt
        
        with pytest.raises(ValueError, match="Unknown category"):
            generate_category_prompt("invalid_category")


class TestCategoryDetection:
    """Tests for category detection from labels."""
    
    def test_get_category_from_label_tops(self):
        """Test detecting tops category."""
        from src.layer0_segmentation import get_category_from_label, GarmentCategory
        
        assert get_category_from_label("t-shirt") == GarmentCategory.TOPS
        assert get_category_from_label("sweater") == GarmentCategory.TOPS
        assert get_category_from_label("hoodie") == GarmentCategory.TOPS
    
    def test_get_category_from_label_bottoms(self):
        """Test detecting bottoms category."""
        from src.layer0_segmentation import get_category_from_label, GarmentCategory
        
        assert get_category_from_label("pants") == GarmentCategory.BOTTOMS
        assert get_category_from_label("jeans") == GarmentCategory.BOTTOMS
        assert get_category_from_label("skirt") == GarmentCategory.BOTTOMS
    
    def test_get_category_from_label_full_body(self):
        """Test detecting full_body category."""
        from src.layer0_segmentation import get_category_from_label, GarmentCategory
        
        assert get_category_from_label("dress") == GarmentCategory.FULL_BODY
        assert get_category_from_label("jumpsuit") == GarmentCategory.FULL_BODY
    
    def test_get_category_from_label_unknown(self):
        """Test unknown labels return UNKNOWN."""
        from src.layer0_segmentation import get_category_from_label, GarmentCategory
        
        assert get_category_from_label("unknown_item") == GarmentCategory.UNKNOWN
        assert get_category_from_label("random") == GarmentCategory.UNKNOWN


class TestSCHPMapping:
    """Tests for SCHP label mapping."""
    
    def test_schp_labels_defined(self):
        """Test that SCHP labels are defined."""
        from src.layer0_segmentation import SCHP_LABELS
        
        assert 0 in SCHP_LABELS  # background
        assert 5 in SCHP_LABELS  # upper_clothes
        assert 9 in SCHP_LABELS  # pants
        assert SCHP_LABELS[5] == "upper_clothes"
        assert SCHP_LABELS[9] == "pants"
    
    def test_get_schp_category(self):
        """Test getting category from SCHP label ID."""
        from src.layer0_segmentation import get_schp_category, GarmentCategory
        
        assert get_schp_category(5) == GarmentCategory.TOPS  # upper_clothes
        assert get_schp_category(9) == GarmentCategory.BOTTOMS  # pants
        assert get_schp_category(6) == GarmentCategory.FULL_BODY  # dress
        assert get_schp_category(0) == GarmentCategory.UNKNOWN  # background
    
    def test_is_garment_schp_label(self):
        """Test checking if SCHP label represents garment."""
        from src.layer0_segmentation import is_garment_schp_label
        
        assert is_garment_schp_label(5) is True  # upper_clothes
        assert is_garment_schp_label(9) is True  # pants
        assert is_garment_schp_label(0) is False  # background
        assert is_garment_schp_label(13) is False  # face


class TestGetAllGarmentLabels:
    """Tests for get_all_garment_labels function."""
    
    def test_returns_flat_list(self):
        """Test that function returns flat list of all labels."""
        from src.layer0_segmentation import get_all_garment_labels
        
        labels = get_all_garment_labels()
        
        assert isinstance(labels, list)
        assert "t-shirt" in labels
        assert "pants" in labels
        assert "dress" in labels
        assert "jacket" in labels
        assert "sneakers" in labels
        assert "bag" in labels
    
    def test_no_duplicates(self):
        """Test that there are no duplicate labels."""
        from src.layer0_segmentation import get_all_garment_labels
        
        labels = get_all_garment_labels()
        assert len(labels) == len(set(labels))
