"""
Tests for Fit Predictor
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.layer3_context.fit_predictor import (
    FitPredictor,
    FitPrediction,
    OutfitFitPrediction
)


class TestFitPrediction:
    """Tests for FitPrediction dataclass."""
    
    def test_fit_prediction_creation(self):
        """Test creating a FitPrediction."""
        prediction = FitPrediction(
            score=0.85,
            fit_type="good",
            size_difference=0,
            notes=["Size matches well"]
        )
        
        assert prediction.score == 0.85
        assert prediction.fit_type == "good"
        assert prediction.size_difference == 0
        assert "Size matches well" in prediction.notes
    
    def test_fit_prediction_defaults(self):
        """Test FitPrediction default values."""
        prediction = FitPrediction(
            score=0.5,
            fit_type="acceptable",
            size_difference=1
        )
        
        assert prediction.notes == []
        assert prediction.adjustments_needed == []


class TestOutfitFitPrediction:
    """Tests for OutfitFitPrediction dataclass."""
    
    def test_outfit_fit_prediction_creation(self):
        """Test creating an OutfitFitPrediction."""
        garment_scores = {
            "garment1": FitPrediction(score=0.9, fit_type="perfect", size_difference=0),
            "garment2": FitPrediction(score=0.7, fit_type="good", size_difference=1)
        }
        
        prediction = OutfitFitPrediction(
            overall_score=0.8,
            garment_scores=garment_scores,
            average_fit_type="good",
            fit_notes=["Overall good fit"]
        )
        
        assert prediction.overall_score == 0.8
        assert len(prediction.garment_scores) == 2
        assert prediction.average_fit_type == "good"


class TestFitPredictor:
    """Tests for FitPredictor class."""
    
    @pytest.fixture
    def predictor(self):
        """Create a FitPredictor instance."""
        with patch('src.layer3_context.fit_predictor.get_config') as mock_config:
            mock_config.return_value.get_data.return_value = {}
            return FitPredictor()
    
    @pytest.fixture
    def mock_garment(self):
        """Create a mock garment."""
        garment = MagicMock()
        garment.id = "test_garment_1"
        garment.category = "top"
        garment.attributes = {"size": "M", "fit": "regular"}
        return garment
    
    def test_init(self, predictor):
        """Test FitPredictor initialization."""
        assert predictor is not None
        assert predictor.size_match_scores is not None
        assert predictor.size_order is not None
    
    def test_normalize_size_standard(self, predictor):
        """Test normalizing standard sizes."""
        assert predictor._normalize_size("m") == "M"
        assert predictor._normalize_size("  L  ") == "L"
        assert predictor._normalize_size("xl") == "XL"
    
    def test_normalize_size_variations(self, predictor):
        """Test normalizing size variations."""
        assert predictor._normalize_size("EXTRA SMALL") == "XS"
        assert predictor._normalize_size("SMALL") == "S"
        assert predictor._normalize_size("MEDIUM") == "M"
        assert predictor._normalize_size("LARGE") == "L"
        assert predictor._normalize_size("EXTRA LARGE") == "XL"
        assert predictor._normalize_size("2XL") == "XXL"
    
    def test_calculate_size_difference_exact(self, predictor):
        """Test size difference calculation for exact match."""
        diff = predictor._calculate_size_difference("M", "M")
        assert diff == 0
    
    def test_calculate_size_difference_larger(self, predictor):
        """Test size difference when garment is larger."""
        diff = predictor._calculate_size_difference("L", "M")
        assert diff == 1
        
        diff = predictor._calculate_size_difference("XL", "M")
        assert diff == 2
    
    def test_calculate_size_difference_smaller(self, predictor):
        """Test size difference when garment is smaller."""
        diff = predictor._calculate_size_difference("S", "M")
        assert diff == -1
        
        diff = predictor._calculate_size_difference("XS", "M")
        assert diff == -2
    
    def test_get_size_match_score_exact(self, predictor):
        """Test score for exact size match."""
        score = predictor._get_size_match_score(0)
        assert score == 1.0
    
    def test_get_size_match_score_one_off(self, predictor):
        """Test score for one size off."""
        score = predictor._get_size_match_score(1)
        assert score == 0.7
    
    def test_get_size_match_score_two_off(self, predictor):
        """Test score for two sizes off."""
        score = predictor._get_size_match_score(2)
        assert score == 0.4
    
    def test_get_size_match_score_three_plus_off(self, predictor):
        """Test score for three or more sizes off."""
        score = predictor._get_size_match_score(3)
        assert score == 0.1
        
        score = predictor._get_size_match_score(5)
        assert score == 0.1
    
    def test_classify_fit_perfect(self, predictor):
        """Test fit classification for perfect score."""
        assert predictor._classify_fit(0.95) == "perfect"
        assert predictor._classify_fit(0.90) == "perfect"
    
    def test_classify_fit_good(self, predictor):
        """Test fit classification for good score."""
        assert predictor._classify_fit(0.85) == "good"
        assert predictor._classify_fit(0.70) == "good"
    
    def test_classify_fit_acceptable(self, predictor):
        """Test fit classification for acceptable score."""
        assert predictor._classify_fit(0.65) == "acceptable"
        assert predictor._classify_fit(0.50) == "acceptable"
    
    def test_classify_fit_poor(self, predictor):
        """Test fit classification for poor score."""
        assert predictor._classify_fit(0.45) == "poor"
        assert predictor._classify_fit(0.10) == "poor"
    
    def test_get_garment_fit_type_slim(self, predictor):
        """Test detecting slim fit garments."""
        garment = MagicMock()
        garment.attributes = {"fit": "slim fit"}
        assert predictor._get_garment_fit_type(garment) == "slim"
        
        garment.attributes = {"fit": "skinny"}
        assert predictor._get_garment_fit_type(garment) == "slim"
    
    def test_get_garment_fit_type_oversized(self, predictor):
        """Test detecting oversized garments."""
        garment = MagicMock()
        garment.attributes = {"fit": "oversized"}
        assert predictor._get_garment_fit_type(garment) == "oversized"
        
        garment.attributes = {"fit": "relaxed fit"}
        assert predictor._get_garment_fit_type(garment) == "oversized"
    
    def test_get_garment_fit_type_regular(self, predictor):
        """Test detecting regular fit garments."""
        garment = MagicMock()
        garment.attributes = {"fit": "regular"}
        assert predictor._get_garment_fit_type(garment) == "regular"
        
        garment.attributes = {}
        assert predictor._get_garment_fit_type(garment) == "regular"
    
    def test_predict_fit_no_garment_size(self, predictor):
        """Test prediction when garment has no size info."""
        # Create a new mock that properly returns None for size-related fields
        garment = MagicMock()
        garment.id = "test_garment_no_size"
        garment.category = "top"
        garment.attributes = {}  # Empty attributes - no size
        # Explicitly set size-related fields to None to override MagicMock default
        garment.size = None
        garment.garment_size = None
        garment.clothing_size = None
        
        prediction = predictor.predict_fit(
            garment,
            user_top_size="M"
        )
        
        # When garment size is not available, returns neutral score
        assert prediction.score == 0.7
        assert prediction.fit_type == "unknown"
        assert "not available" in prediction.notes[0].lower()
    
    def test_predict_fit_no_user_size(self, predictor, mock_garment):
        """Test prediction when user has no size info."""
        prediction = predictor.predict_fit(
            mock_garment,
            user_top_size=None,
            user_bottom_size=None
        )
        
        assert prediction.score == 0.7
        assert prediction.fit_type == "unknown"
    
    def test_predict_fit_exact_match(self, predictor, mock_garment):
        """Test prediction for exact size match."""
        mock_garment.attributes = {"size": "M"}
        mock_garment.category = "top"
        
        prediction = predictor.predict_fit(
            mock_garment,
            user_top_size="M"
        )
        
        assert prediction.score >= 0.9
        assert prediction.fit_type == "perfect"
        assert prediction.size_difference == 0
    
    def test_predict_fit_one_size_off(self, predictor, mock_garment):
        """Test prediction for one size difference."""
        mock_garment.attributes = {"size": "L"}
        mock_garment.category = "top"
        
        prediction = predictor.predict_fit(
            mock_garment,
            user_top_size="M"
        )
        
        assert prediction.score == 0.7
        assert prediction.fit_type == "good"
        assert prediction.size_difference == 1
    
    def test_predict_fit_bottom_garment(self, predictor, mock_garment):
        """Test prediction uses bottom size for pants."""
        mock_garment.attributes = {"size": "L"}
        mock_garment.category = "pants"
        
        prediction = predictor.predict_fit(
            mock_garment,
            user_top_size="M",
            user_bottom_size="L"
        )
        
        assert prediction.score >= 0.9
        assert prediction.size_difference == 0
    
    def test_predict_outfit_fit(self, predictor):
        """Test prediction for entire outfit."""
        garment1 = MagicMock()
        garment1.id = "top1"
        garment1.category = "top"
        garment1.attributes = {"size": "M"}
        
        garment2 = MagicMock()
        garment2.id = "pants1"
        garment2.category = "pants"
        garment2.attributes = {"size": "M"}
        
        prediction = predictor.predict_outfit_fit(
            [garment1, garment2],
            user_top_size="M",
            user_bottom_size="M"
        )
        
        assert isinstance(prediction, OutfitFitPrediction)
        assert prediction.overall_score >= 0.9
        assert len(prediction.garment_scores) == 2
    
    def test_predict_outfit_fit_mixed(self, predictor):
        """Test prediction for outfit with mixed fits."""
        garment1 = MagicMock()
        garment1.id = "top1"
        garment1.category = "top"
        garment1.attributes = {"size": "M"}  # Matches
        
        garment2 = MagicMock()
        garment2.id = "pants1"
        garment2.category = "pants"
        garment2.attributes = {"size": "XL"}  # Too big
        
        prediction = predictor.predict_outfit_fit(
            [garment1, garment2],
            user_top_size="M",
            user_bottom_size="M"
        )
        
        assert prediction.overall_score < 0.9
        # Should have penalty for poor-fitting item
    
    def test_predict_outfit_fit_empty(self, predictor):
        """Test prediction for empty outfit."""
        prediction = predictor.predict_outfit_fit(
            [],
            user_top_size="M"
        )
        
        assert prediction.overall_score == 0.7
        assert len(prediction.garment_scores) == 0
    
    def test_generate_fit_notes_exact(self, predictor):
        """Test note generation for exact size."""
        notes = predictor._generate_fit_notes(0, "regular", None)
        assert "Size matches well" in notes
    
    def test_generate_fit_notes_larger(self, predictor):
        """Test note generation for larger garment."""
        notes = predictor._generate_fit_notes(2, "regular", None)
        assert any("larger" in note for note in notes)
    
    def test_generate_fit_notes_smaller(self, predictor):
        """Test note generation for smaller garment."""
        notes = predictor._generate_fit_notes(-1, "regular", None)
        assert any("smaller" in note for note in notes)
    
    def test_generate_fit_notes_slim_overweight(self, predictor):
        """Test note for slim fit with overweight BMI."""
        notes = predictor._generate_fit_notes(0, "slim", "overweight")
        assert any("slim" in note.lower() and "may" in note.lower() for note in notes)


class TestFitPredictorWithConfig:
    """Test FitPredictor with configuration."""
    
    def test_load_config_from_file(self, tmp_path):
        """Test loading configuration from file."""
        config_file = tmp_path / "test_config.json"
        config_content = {
            "fit_predictor": {
                "size_match_scores": {
                    "exact": 1.0,
                    "one_off": 0.8,
                    "two_off": 0.5,
                    "three_plus_off": 0.2
                },
                "size_order": ["XS", "S", "M", "L", "XL"]
            }
        }
        
        import json
        config_file.write_text(json.dumps(config_content))
        
        with patch('src.layer3_context.fit_predictor.get_config') as mock_config:
            mock_config.return_value.get_data.return_value = {}
            predictor = FitPredictor(config_path=config_file)
        
        assert predictor.size_match_scores["one_off"] == 0.8
        assert predictor.size_match_scores["two_off"] == 0.5
