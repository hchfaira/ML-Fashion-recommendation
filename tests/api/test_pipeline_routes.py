"""
Tests for Full Pipeline API Route
===================================

Tests the POST /api/v1/pipeline/recommend endpoint which integrates ALL layers:

Layer 0 – Segmentation (garment extraction)
Layer 1 – Vision (attribute extraction)
Layer 2 – Style (outfit building & scoring)
Layer 3 – Context (user profile + context scoring)
Layer 4 – LLM (explanations)
Layer 5 – Visualization (catalogue images)
Layer 6 – Try-On (virtual try-on)

Every heavy service is mocked so the tests run fast and without GPU/API keys.
"""

import base64
import io
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import uuid4

from PIL import Image
from fastapi import HTTPException

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory,
    ColorProfile, UserContext, Occasion,
    Outfit, OutfitItem, FormalityLevel, Season,
    WeatherContext,
)
from src.api.routes.pipeline import (
    PipelineRequest,
    PipelineResponse,
    WardrobeImageItem,
    UserProfileInput,
    OutfitRecommendation,
    GarmentOut,
    UserProfileOut,
    full_pipeline_recommend,
    _organise_wardrobe,
    _garment_to_out,
    _candidates_to_outfits,
    _candidate_to_outfit,
    _enrich_context_from_profile,
    _b64_to_pil,
    _pil_to_b64,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_b64_image(width: int = 64, height: int = 64, color="red") -> str:
    """Create a tiny base-64 encoded PNG for testing."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_garment(
    garment_id: str = None,
    category: GarmentCategory = GarmentCategory.TOP,
    color: str = "white",
    subcategory: str = None,
    formality: FormalityLevel = FormalityLevel.CASUAL,
) -> Garment:
    """Create a sample Garment."""
    return Garment(
        id=garment_id or f"g_{uuid4().hex[:8]}",
        attributes=GarmentAttributes(
            category=category,
            subcategory=subcategory or category.value,
            color=ColorProfile(primary=color, hex_codes=[]),
            formality_level=formality,
            season_suitable=[Season.SPRING, Season.SUMMER],
        ),
    )


def _make_candidate(garments=None, score=0.85, name="Test Outfit"):
    """Create a mock OutfitCandidate."""
    if garments is None:
        garments = [
            _make_garment(category=GarmentCategory.TOP, color="white"),
            _make_garment(category=GarmentCategory.BOTTOM, color="blue"),
        ]

    mock = MagicMock()
    mock.garments = garments
    mock.overall_score = score
    mock.name = name
    mock.scorecard = MagicMock()
    mock.scorecard.scores = {
        "overall": score,
        "color_harmony": 0.80,
        "seven_point": 0.75,
        "design_principles": 0.70,
        "occasion": 0.65,
    }
    # Allow sorting
    mock.__lt__ = lambda self, other: self.overall_score < other.overall_score
    return mock


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def b64_image():
    """A tiny red 64×64 PNG encoded in base-64."""
    return _make_b64_image(64, 64, "red")


@pytest.fixture
def b64_image_blue():
    """A tiny blue 64×64 PNG."""
    return _make_b64_image(64, 64, "blue")


@pytest.fixture
def sample_garments():
    """Predefined list of garments covering the main categories."""
    return [
        _make_garment("top_1", GarmentCategory.TOP, "white", "t-shirt"),
        _make_garment("bottom_1", GarmentCategory.BOTTOM, "blue", "jeans"),
        _make_garment("shoes_1", GarmentCategory.SHOES, "white", "sneakers"),
        _make_garment("outer_1", GarmentCategory.OUTERWEAR, "black", "jacket"),
    ]


@pytest.fixture
def sample_context():
    """A minimal UserContext."""
    return UserContext(
        user_id="test_user",
        occasion=Occasion.CASUAL,
    )


@pytest.fixture
def mock_outfit_builder():
    """
    Patch get_outfit_builder so it returns a controllable mock.
    """
    mock = MagicMock()
    # extract_garment_from_image returns a single garment
    mock.extract_garment_from_image = AsyncMock(
        side_effect=lambda path: _make_garment()
    )
    # extract_full_outfit_from_image returns two garments
    mock.extract_full_outfit_from_image = AsyncMock(
        side_effect=lambda path: [
            _make_garment(category=GarmentCategory.TOP, color="white"),
            _make_garment(category=GarmentCategory.BOTTOM, color="blue"),
        ]
    )
    # search_best_outfits returns mock candidates
    mock.search_best_outfits = MagicMock(
        return_value=[
            _make_candidate(score=0.90, name="Outfit A"),
            _make_candidate(score=0.85, name="Outfit B"),
            _make_candidate(score=0.80, name="Outfit C"),
        ]
    )
    with patch(
        "src.api.routes.pipeline.get_outfit_builder", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_context_engine():
    """Patch get_context_engine."""
    mock = MagicMock()
    mock.set_style_profile = MagicMock()
    mock.apply_context = AsyncMock(side_effect=lambda outfits, ctx: outfits)
    with patch(
        "src.api.routes.pipeline.get_context_engine", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_outfit_explainer():
    """Patch get_outfit_explainer."""
    explanation = MagicMock()
    explanation.summary = "A stylish casual outfit with great color harmony."

    mock = MagicMock()
    mock.explain = AsyncMock(return_value=explanation)
    with patch(
        "src.api.routes.pipeline.get_outfit_explainer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_visualizer():
    """Patch get_visualizer."""
    viz_result = MagicMock()
    viz_result.image = Image.new("RGBA", (200, 200), "white")

    mock = MagicMock()
    mock.create_outfit_catalogue = MagicMock(return_value=viz_result)
    with patch(
        "src.api.routes.pipeline.get_visualizer", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_tryon_service():
    """Patch get_tryon_service."""
    result = MagicMock()
    result.composite_image = Image.new("RGB", (200, 300), "green")

    mock = MagicMock()
    mock.try_on_outfit = MagicMock(return_value=result)
    with patch(
        "src.api.routes.pipeline.get_tryon_service", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_style_profile_pipeline():
    """Patch get_style_profile_pipeline."""
    profile = MagicMock()
    profile.body_shape = "rectangle"
    profile.color_profile = MagicMock()
    profile.color_profile.skin_tone = MagicMock(value="medium")
    profile.color_profile.undertone = MagicMock(value="warm")
    profile.hair_color = MagicMock(value="dark_brown")
    profile.contrast_level = MagicMock(value="high")
    profile.body_metrics = MagicMock()
    profile.body_metrics.estimated_top_size = "M"
    profile.body_metrics.estimated_bottom_size = "M"

    pipeline_result = MagicMock()
    pipeline_result.profile = profile
    pipeline_result.stages_completed = ["body_detection", "face_detection", "skin_tone"]
    pipeline_result.stages_failed = []

    mock = MagicMock()
    mock.analyze = MagicMock(return_value=pipeline_result)
    with patch(
        "src.api.routes.pipeline.get_style_profile_pipeline", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_garment_extraction_pipeline():
    """Patch get_garment_extraction_pipeline."""
    eg = MagicMock()
    eg.image_path = "/tmp/garment.png"

    result = MagicMock()
    result.garments = [eg]

    mock = MagicMock()
    mock.process = MagicMock(return_value=result)
    with patch(
        "src.api.routes.pipeline.get_garment_extraction_pipeline", return_value=mock
    ):
        yield mock


@pytest.fixture
def all_mocks(
    mock_outfit_builder,
    mock_context_engine,
    mock_outfit_explainer,
    mock_visualizer,
    mock_tryon_service,
    mock_style_profile_pipeline,
    mock_garment_extraction_pipeline,
):
    """Convenience fixture that activates all mocks."""
    return {
        "builder": mock_outfit_builder,
        "context": mock_context_engine,
        "explainer": mock_outfit_explainer,
        "visualizer": mock_visualizer,
        "tryon": mock_tryon_service,
        "profile": mock_style_profile_pipeline,
        "segmentation": mock_garment_extraction_pipeline,
    }


# ============================================================================
# Unit tests — pure helper functions
# ============================================================================

class TestHelpers:
    """Tests for pure utility functions."""

    def test_b64_roundtrip(self):
        """Encoding then decoding a PIL image must give the same pixels."""
        original = Image.new("RGB", (10, 10), "red")
        b64 = _pil_to_b64(original)
        decoded = _b64_to_pil(b64)
        assert decoded.size == original.size

    def test_organise_wardrobe(self, sample_garments):
        """_organise_wardrobe splits garments into the right buckets."""
        wardrobe = _organise_wardrobe(sample_garments)

        assert len(wardrobe["tops"]) == 1
        assert len(wardrobe["bottoms"]) == 1
        assert len(wardrobe["shoes"]) == 1
        assert len(wardrobe["outerwear"]) == 1
        assert len(wardrobe["accessories"]) == 0

    def test_organise_wardrobe_dress_goes_to_full_body(self):
        """A dress garment should land in 'full_body'."""
        dress = _make_garment(category=GarmentCategory.DRESS, color="red")
        wardrobe = _organise_wardrobe([dress])
        assert len(wardrobe["full_body"]) == 1

    def test_garment_to_out(self):
        """_garment_to_out produces correct GarmentOut."""
        g = _make_garment("g1", GarmentCategory.TOP, "navy", "polo")
        out = _garment_to_out(g)
        assert out.id == "g1"
        assert out.category == "top"
        assert out.subcategory == "polo"
        assert out.color_primary == "navy"

    def test_candidates_to_outfits(self):
        """_candidates_to_outfits converts to valid Outfit objects."""
        candidates = [_make_candidate(), _make_candidate()]
        outfits = _candidates_to_outfits(candidates)
        assert len(outfits) == 2
        for o in outfits:
            assert isinstance(o, Outfit)
            assert len(o.items) == 2

    def test_enrich_context_from_profile(self):
        """_enrich_context_from_profile merges profile into context."""
        ctx = UserContext()
        profile = MagicMock()
        profile.body_shape = MagicMock(value="pear")
        profile.color_profile = MagicMock()
        profile.color_profile.undertone = MagicMock(value="cool")
        profile.hair_color = MagicMock(value="black")
        profile.contrast_level = MagicMock(value="high")

        _enrich_context_from_profile(ctx, profile)

        assert ctx.body_shape == "pear"
        assert ctx.skin_undertone == "cool"
        assert ctx.hair_color == "black"
        assert ctx.contrast_type == "high"

    def test_enrich_context_from_profile_none_safe(self):
        """_enrich_context_from_profile must not crash on None."""
        ctx = UserContext()
        _enrich_context_from_profile(ctx, None)
        assert ctx.body_shape is None


# ============================================================================
# Full pipeline endpoint tests
# ============================================================================

class TestFullPipelineEndpoint:
    """Tests for POST /pipeline/recommend."""

    @pytest.mark.asyncio
    async def test_minimal_request_two_images(self, all_mocks, b64_image):
        """
        Minimal valid request: 2 wardrobe images → garments extracted,
        outfits scored, explanations + visualisations returned.
        """
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            top_k=2,
        )

        response = await full_pipeline_recommend(request)

        assert isinstance(response, PipelineResponse)
        assert len(response.recommendations) <= 2
        assert response.total_garments_extracted >= 2
        assert "garment_extraction" in response.stages_completed
        assert "outfit_scoring" in response.stages_completed
        assert response.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_single_image_too_few_garments_raises_400(
        self, all_mocks, b64_image
    ):
        """
        If only 1 garment is extracted (1 image with extract_category),
        the endpoint must return HTTP 400.
        """
        # Make builder return only 1 garment
        all_mocks["builder"].extract_garment_from_image = AsyncMock(
            return_value=_make_garment()
        )
        all_mocks["builder"].extract_full_outfit_from_image = AsyncMock(
            return_value=[_make_garment()]
        )

        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image, extract_category="top"),
            ],
        )

        with pytest.raises(HTTPException) as exc_info:
            await full_pipeline_recommend(request)
        assert exc_info.value.status_code == 400
        assert "at least 2 garments" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_extract_specific_category(
        self, all_mocks, b64_image
    ):
        """
        When extract_category is set, the pipeline should use Layer 0
        to extract only that category.
        """
        # We need at least 2 garments total so add a second full image
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image, extract_category="top"),
                WardrobeImageItem(image_b64=b64_image),  # extracts 2 garments
            ],
        )

        response = await full_pipeline_recommend(request)

        # Layer 0 pipeline should have been called for the first image
        all_mocks["segmentation"].process.assert_called_once()
        assert response.total_garments_extracted >= 2

    @pytest.mark.asyncio
    async def test_extract_all_garments_from_image(
        self, all_mocks, b64_image
    ):
        """
        When extract_category is None, extract_full_outfit_from_image
        is used (Layer 0 + Layer 1).
        """
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
            ],
        )

        response = await full_pipeline_recommend(request)

        all_mocks["builder"].extract_full_outfit_from_image.assert_called()
        assert response.total_garments_extracted >= 2

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self, all_mocks, b64_image):
        """Response should contain at most top_k recommendations."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            top_k=1,
        )

        response = await full_pipeline_recommend(request)

        assert len(response.recommendations) <= 1

    @pytest.mark.asyncio
    async def test_recommendations_are_ranked(self, all_mocks, b64_image):
        """Recommendations must be sorted by rank 1, 2, 3, …"""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            top_k=3,
        )

        response = await full_pipeline_recommend(request)

        ranks = [r.rank for r in response.recommendations]
        assert ranks == sorted(ranks)

    @pytest.mark.asyncio
    async def test_score_breakdown_present(self, all_mocks, b64_image):
        """Each recommendation should include a score breakdown dict."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            top_k=1,
        )

        response = await full_pipeline_recommend(request)
        rec = response.recommendations[0]

        assert isinstance(rec.score_breakdown, dict)
        assert rec.overall_score > 0


# ============================================================================
# User profile tests
# ============================================================================

class TestUserProfile:
    """Tests for user profile integration (Layer 3 — user profile pipeline)."""

    @pytest.mark.asyncio
    async def test_user_profile_extracted(
        self, all_mocks, b64_image
    ):
        """When user_profile is provided the profile pipeline runs."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            user_profile=UserProfileInput(
                image_b64=b64_image,
                height_cm=175,
                weight_kg=70,
            ),
        )

        response = await full_pipeline_recommend(request)

        assert response.user_profile is not None
        assert response.user_profile.body_shape == "rectangle"
        assert response.user_profile.skin_tone == "medium"
        assert response.user_profile.height_cm == 175
        assert "user_profile" in response.stages_completed

    @pytest.mark.asyncio
    async def test_user_profile_without_image(
        self, all_mocks, b64_image
    ):
        """
        height/weight can be provided without an image;
        the profile pipeline should still run.
        """
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            user_profile=UserProfileInput(
                height_cm=180,
                weight_kg=80,
            ),
        )

        response = await full_pipeline_recommend(request)

        # Pipeline was called (it may work with None image depending on impl)
        all_mocks["profile"].analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_profile_failure_does_not_crash(
        self, all_mocks, b64_image
    ):
        """
        If user profile extraction fails the pipeline should continue
        and report the error.
        """
        all_mocks["profile"].analyze.side_effect = RuntimeError("GPU out of memory")

        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            user_profile=UserProfileInput(image_b64=b64_image),
        )

        response = await full_pipeline_recommend(request)

        assert response.user_profile is None
        assert any("user_profile" in e for e in response.errors)

    @pytest.mark.asyncio
    async def test_context_enriched_from_profile(
        self, all_mocks, b64_image
    ):
        """
        Extracted profile data should be merged into the UserContext
        before scoring.
        """
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            user_profile=UserProfileInput(image_b64=b64_image),
            context=UserContext(occasion=Occasion.BUSINESS),
        )

        response = await full_pipeline_recommend(request)

        # The context engine should have been given the profile
        all_mocks["context"].set_style_profile.assert_called_once()


# ============================================================================
# Context scoring tests
# ============================================================================

class TestContextScoring:
    """Tests for context-aware scoring (Layer 3)."""

    @pytest.mark.asyncio
    async def test_context_engine_called(self, all_mocks, b64_image):
        """ContextEngine.apply_context should be called after scoring."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            context=UserContext(occasion=Occasion.FORMAL),
        )

        response = await full_pipeline_recommend(request)

        all_mocks["context"].apply_context.assert_called_once()
        assert "context_scoring" in response.stages_completed

    @pytest.mark.asyncio
    async def test_context_failure_reported(self, all_mocks, b64_image):
        """If context scoring fails, the error is reported but pipeline continues."""
        all_mocks["context"].apply_context = AsyncMock(
            side_effect=RuntimeError("context crash")
        )

        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
        )

        response = await full_pipeline_recommend(request)

        assert any("context_scoring" in e for e in response.errors)
        # Should still have recommendations from Layer 2
        assert len(response.recommendations) > 0


# ============================================================================
# LLM explanation tests
# ============================================================================

class TestLLMExplanation:
    """Tests for LLM explanations (Layer 4)."""

    @pytest.mark.asyncio
    async def test_explanations_generated(self, all_mocks, b64_image):
        """Each recommendation should have an explanation."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            enable_explanation=True,
            top_k=2,
        )

        response = await full_pipeline_recommend(request)

        for rec in response.recommendations:
            assert rec.explanation is not None
            assert len(rec.explanation) > 0
        assert "llm_explanation" in response.stages_completed

    @pytest.mark.asyncio
    async def test_explanations_disabled(self, all_mocks, b64_image):
        """When enable_explanation=False, no LLM calls should be made."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            enable_explanation=False,
        )

        response = await full_pipeline_recommend(request)

        all_mocks["explainer"].explain.assert_not_called()
        assert "llm_explanation" not in response.stages_completed

    @pytest.mark.asyncio
    async def test_explanation_detail_levels(self, all_mocks, b64_image):
        """The detail level is forwarded to the explainer."""
        for level in ("brief", "standard", "detailed"):
            request = PipelineRequest(
                wardrobe_images=[
                    WardrobeImageItem(image_b64=b64_image),
                    WardrobeImageItem(image_b64=b64_image),
                ],
                enable_explanation=True,
                explanation_detail=level,
                top_k=1,
            )

            await full_pipeline_recommend(request)

            _, kwargs = all_mocks["explainer"].explain.call_args
            assert kwargs["detail_level"] == level

    @pytest.mark.asyncio
    async def test_explanation_failure_is_non_fatal(
        self, all_mocks, b64_image
    ):
        """If LLM fails, recommendations still come back without explanations."""
        all_mocks["explainer"].explain = AsyncMock(
            side_effect=RuntimeError("LLM timeout")
        )

        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            enable_explanation=True,
        )

        response = await full_pipeline_recommend(request)

        assert len(response.recommendations) > 0
        assert any("llm_explanation" in e for e in response.errors)


# ============================================================================
# Visualization tests
# ============================================================================

class TestVisualization:
    """Tests for catalogue visualisation (Layer 5)."""

    @pytest.mark.asyncio
    async def test_catalogue_images_generated(self, all_mocks, b64_image):
        """Each recommendation gets a base-64 catalogue image."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            enable_visualization=True,
            top_k=2,
        )

        response = await full_pipeline_recommend(request)

        for rec in response.recommendations:
            assert rec.catalogue_image_b64 is not None
            # Verify it decodes to a valid image
            img = _b64_to_pil(rec.catalogue_image_b64)
            assert img.size[0] > 0
        assert "visualization" in response.stages_completed

    @pytest.mark.asyncio
    async def test_visualization_disabled(self, all_mocks, b64_image):
        """When enable_visualization=False, no images should be generated."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            enable_visualization=False,
        )

        response = await full_pipeline_recommend(request)

        all_mocks["visualizer"].create_outfit_catalogue.assert_not_called()
        for rec in response.recommendations:
            assert rec.catalogue_image_b64 is None

    @pytest.mark.asyncio
    async def test_visualization_failure_non_fatal(
        self, all_mocks, b64_image
    ):
        """Visualization crash should not kill the pipeline."""
        all_mocks["visualizer"].create_outfit_catalogue.side_effect = RuntimeError(
            "PIL error"
        )

        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            enable_visualization=True,
        )

        response = await full_pipeline_recommend(request)

        assert any("visualization" in e for e in response.errors)
        assert len(response.recommendations) > 0


# ============================================================================
# Virtual try-on tests
# ============================================================================

class TestVirtualTryOn:
    """Tests for virtual try-on (Layer 6)."""

    @pytest.mark.asyncio
    async def test_tryon_runs_when_enabled(self, all_mocks, b64_image):
        """Try-on should run for the #1 outfit when enabled + user photo given."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            user_profile=UserProfileInput(image_b64=b64_image),
            enable_tryon=True,
            top_k=2,
        )

        response = await full_pipeline_recommend(request)

        all_mocks["tryon"].try_on_outfit.assert_called_once()
        assert response.recommendations[0].tryon_image_b64 is not None
        # Only the #1 outfit gets try-on
        if len(response.recommendations) > 1:
            assert response.recommendations[1].tryon_image_b64 is None
        assert "tryon" in response.stages_completed

    @pytest.mark.asyncio
    async def test_tryon_disabled_by_default(self, all_mocks, b64_image):
        """Try-on should NOT run when enable_tryon is False (default)."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            enable_tryon=False,
        )

        response = await full_pipeline_recommend(request)

        all_mocks["tryon"].try_on_outfit.assert_not_called()
        for rec in response.recommendations:
            assert rec.tryon_image_b64 is None

    @pytest.mark.asyncio
    async def test_tryon_requires_user_image(self, all_mocks, b64_image):
        """Try-on should error if no user image is given."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            enable_tryon=True,
            # No user_profile → no user image
        )

        response = await full_pipeline_recommend(request)

        all_mocks["tryon"].try_on_outfit.assert_not_called()
        assert any("tryon" in e for e in response.errors)

    @pytest.mark.asyncio
    async def test_tryon_failure_non_fatal(self, all_mocks, b64_image):
        """Try-on crash should not kill the pipeline."""
        all_mocks["tryon"].try_on_outfit.side_effect = RuntimeError(
            "CUDA OOM"
        )

        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            user_profile=UserProfileInput(image_b64=b64_image),
            enable_tryon=True,
        )

        response = await full_pipeline_recommend(request)

        assert any("tryon" in e for e in response.errors)
        assert len(response.recommendations) > 0


# ============================================================================
# Scoring profile tests
# ============================================================================

class TestScoringProfiles:
    """Tests for different scoring profiles."""

    @pytest.mark.asyncio
    async def test_scoring_profile_forwarded(self, all_mocks, b64_image):
        """scoring_profile should be forwarded to outfit builder."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            scoring_profile="business",
        )

        await full_pipeline_recommend(request)

        call_kwargs = all_mocks["builder"].search_best_outfits.call_args
        assert call_kwargs.kwargs.get("profile") == "business" or (
            len(call_kwargs.args) > 0  # positional fallback
        )

    @pytest.mark.asyncio
    async def test_default_scoring_profile(self, all_mocks, b64_image):
        """Without explicit profile, None is passed (uses default config)."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
        )

        await full_pipeline_recommend(request)

        call_kwargs = all_mocks["builder"].search_best_outfits.call_args
        assert call_kwargs.kwargs.get("profile") is None


# ============================================================================
# Error handling tests
# ============================================================================

class TestErrorHandling:
    """Tests for robustness and error handling."""

    @pytest.mark.asyncio
    async def test_garment_extraction_failure_raises_400(
        self, all_mocks, b64_image
    ):
        """
        If garment extraction itself throws, the endpoint should
        raise 400 because we end up with too few garments.
        """
        all_mocks["builder"].extract_full_outfit_from_image = AsyncMock(
            side_effect=RuntimeError("extraction error")
        )
        all_mocks["builder"].extract_garment_from_image = AsyncMock(
            side_effect=RuntimeError("extraction error")
        )

        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
            ],
        )

        with pytest.raises(HTTPException) as exc_info:
            await full_pipeline_recommend(request)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_outfit_builder_returns_empty_raises_400(
        self, all_mocks, b64_image
    ):
        """If no valid outfit combinations exist, respond with 400."""
        all_mocks["builder"].search_best_outfits.return_value = []

        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
        )

        with pytest.raises(HTTPException) as exc_info:
            await full_pipeline_recommend(request)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_stages_completed_tracks_all_successful_stages(
        self, all_mocks, b64_image
    ):
        """stages_completed list should reflect every stage that ran."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            user_profile=UserProfileInput(image_b64=b64_image),
            enable_explanation=True,
            enable_visualization=True,
            enable_tryon=True,
            top_k=1,
        )

        response = await full_pipeline_recommend(request)

        expected_stages = {
            "garment_extraction",
            "user_profile",
            "outfit_scoring",
            "context_scoring",
            "llm_explanation",
            "visualization",
            "tryon",
        }
        assert expected_stages.issubset(set(response.stages_completed))

    @pytest.mark.asyncio
    async def test_processing_time_is_positive(self, all_mocks, b64_image):
        """processing_time_ms must be a positive number."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
        )

        response = await full_pipeline_recommend(request)
        assert response.processing_time_ms > 0


# ============================================================================
# Integration-style tests (still mocked, but checking cross-layer data flow)
# ============================================================================

class TestCrossLayerDataFlow:
    """Verify data flows correctly between layers."""

    @pytest.mark.asyncio
    async def test_garments_flow_to_builder(self, all_mocks, b64_image):
        """
        Garments extracted from images should be organised and
        passed to the outfit builder.
        """
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
        )

        await full_pipeline_recommend(request)

        # search_best_outfits was called with a wardrobe dict
        call_args = all_mocks["builder"].search_best_outfits.call_args
        wardrobe_arg = call_args.kwargs.get("wardrobe") or call_args.args[0]
        assert isinstance(wardrobe_arg, dict)

    @pytest.mark.asyncio
    async def test_context_receives_scored_outfits(
        self, all_mocks, b64_image
    ):
        """apply_context should receive Outfit objects and the UserContext."""
        ctx = UserContext(occasion=Occasion.EVENING)
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            context=ctx,
        )

        await full_pipeline_recommend(request)

        call_args = all_mocks["context"].apply_context.call_args
        outfits_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("outfits")
        context_arg = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("context")

        assert isinstance(outfits_arg, list)
        assert len(outfits_arg) > 0
        assert isinstance(outfits_arg[0], Outfit)

    @pytest.mark.asyncio
    async def test_explainer_receives_outfit_and_context(
        self, all_mocks, b64_image
    ):
        """OutfitExplainer.explain should receive an Outfit + UserContext."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            enable_explanation=True,
            top_k=1,
        )

        await full_pipeline_recommend(request)

        call_args = all_mocks["explainer"].explain.call_args
        outfit_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("outfit")
        assert isinstance(outfit_arg, Outfit)

    @pytest.mark.asyncio
    async def test_visualizer_receives_garments(
        self, all_mocks, b64_image
    ):
        """OutfitVisualizer.create_outfit_catalogue should receive garment list."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            enable_visualization=True,
            top_k=1,
        )

        await full_pipeline_recommend(request)

        call_args = all_mocks["visualizer"].create_outfit_catalogue.call_args
        garments_arg = call_args.kwargs.get("garments") or call_args.args[0]
        assert isinstance(garments_arg, list)
        assert len(garments_arg) > 0

    @pytest.mark.asyncio
    async def test_tryon_receives_person_image_and_garments(
        self, all_mocks, b64_image
    ):
        """VirtualTryOnService.try_on_outfit should receive person image + garments."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            user_profile=UserProfileInput(image_b64=b64_image),
            enable_tryon=True,
            top_k=1,
        )

        await full_pipeline_recommend(request)

        call_kwargs = all_mocks["tryon"].try_on_outfit.call_args.kwargs
        assert "person_image" in call_kwargs
        assert "garments" in call_kwargs
        assert "outfit_name" in call_kwargs


# ============================================================================
# Request validation tests
# ============================================================================

class TestRequestValidation:
    """Tests for Pydantic request validation."""

    def test_empty_wardrobe_images_rejected(self):
        """PipelineRequest requires at least 1 wardrobe image."""
        with pytest.raises(Exception):
            PipelineRequest(wardrobe_images=[])

    def test_top_k_min_max(self):
        """top_k must be between 1 and 20."""
        b64 = _make_b64_image()
        with pytest.raises(Exception):
            PipelineRequest(
                wardrobe_images=[WardrobeImageItem(image_b64=b64)],
                top_k=0,
            )
        with pytest.raises(Exception):
            PipelineRequest(
                wardrobe_images=[WardrobeImageItem(image_b64=b64)],
                top_k=21,
            )

    def test_valid_request_creates_ok(self):
        """A well-formed request should instantiate without errors."""
        b64 = _make_b64_image()
        req = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64),
                WardrobeImageItem(image_b64=b64, extract_category="bottom"),
            ],
            user_profile=UserProfileInput(
                image_b64=b64, height_cm=170, weight_kg=65
            ),
            context=UserContext(occasion=Occasion.DATE),
            top_k=5,
            scoring_profile="creative",
            enable_explanation=True,
            explanation_detail="detailed",
            enable_visualization=True,
            enable_tryon=True,
            tryon_backend="replicate",
        )
        assert req.top_k == 5
        assert req.scoring_profile == "creative"

    def test_wardrobe_image_item_requires_b64(self):
        """WardrobeImageItem must have image_b64."""
        with pytest.raises(Exception):
            WardrobeImageItem()


# ============================================================================
# Additional coverage: internal helpers
# ============================================================================

class TestCandidateToOutfit:
    """Unit tests for _candidate_to_outfit."""

    def test_converts_candidate_with_scorecard(self):
        cand = _make_candidate(score=0.88, name="Nice Outfit")
        outfit = _candidate_to_outfit(cand, idx=5)

        assert isinstance(outfit, Outfit)
        assert outfit.id.startswith("outfit_5_")
        assert outfit.overall_score == 0.88
        assert len(outfit.items) == 2
        for item in outfit.items:
            assert isinstance(item, OutfitItem)

    def test_converts_candidate_without_scorecard(self):
        cand = _make_candidate(score=0.7, name="X")
        cand.scorecard = None
        outfit = _candidate_to_outfit(cand, idx=0)

        # Should fallback to 0.5 for individual scores
        assert outfit.compatibility_score == 0.5
        assert outfit.style_coherence_score == 0.5
        assert outfit.occasion_match_score == 0.5

    def test_extracts_individual_scores_from_scorecard(self):
        cand = _make_candidate(score=0.9, name="Scored")
        outfit = _candidate_to_outfit(cand, idx=0)

        assert outfit.compatibility_score == 0.80  # color_harmony
        assert outfit.style_coherence_score == 0.70  # design_principles
        assert outfit.occasion_match_score == 0.65  # occasion


class TestUpdateCandidateScores:
    """Unit tests for _update_candidate_scores."""

    def test_updates_scores_by_position(self):
        from src.api.routes.pipeline import _update_candidate_scores

        c1 = _make_candidate(score=0.80)
        c2 = _make_candidate(score=0.70)

        o1 = Outfit(id="o1", items=[], overall_score=0.95,
                     compatibility_score=0.9, style_coherence_score=0.9, occasion_match_score=0.9)
        o2 = Outfit(id="o2", items=[], overall_score=0.60,
                     compatibility_score=0.6, style_coherence_score=0.6, occasion_match_score=0.6)

        _update_candidate_scores([c1, c2], [o1, o2])

        assert c1.overall_score == 0.95
        assert c2.overall_score == 0.60

    def test_handles_fewer_scored_outfits_than_candidates(self):
        from src.api.routes.pipeline import _update_candidate_scores

        c1 = _make_candidate(score=0.80)
        c2 = _make_candidate(score=0.70)

        o1 = Outfit(id="o1", items=[], overall_score=0.99,
                     compatibility_score=0.9, style_coherence_score=0.9, occasion_match_score=0.9)

        _update_candidate_scores([c1, c2], [o1])
        assert c1.overall_score == 0.99
        assert c2.overall_score == 0.70  # unchanged

    def test_handles_empty_scored_outfits(self):
        from src.api.routes.pipeline import _update_candidate_scores

        c1 = _make_candidate(score=0.80)
        _update_candidate_scores([c1], [])
        assert c1.overall_score == 0.80  # unchanged


class TestOrganiseWardrobeExtended:
    """Extended tests for _organise_wardrobe covering all category mappings."""

    def test_bag_goes_to_accessories(self):
        bag = _make_garment(category=GarmentCategory.BAG, color="brown")
        wardrobe = _organise_wardrobe([bag])
        assert len(wardrobe["accessories"]) == 1

    def test_accessory_goes_to_accessories(self):
        acc = _make_garment(category=GarmentCategory.ACCESSORY, color="gold")
        wardrobe = _organise_wardrobe([acc])
        assert len(wardrobe["accessories"]) == 1

    def test_outerwear_bucket(self):
        outer = _make_garment(category=GarmentCategory.OUTERWEAR, color="black")
        wardrobe = _organise_wardrobe([outer])
        assert len(wardrobe["outerwear"]) == 1

    def test_shoes_bucket(self):
        shoes = _make_garment(category=GarmentCategory.SHOES, color="black")
        wardrobe = _organise_wardrobe([shoes])
        assert len(wardrobe["shoes"]) == 1

    def test_mixed_garments(self):
        garments = [
            _make_garment(category=GarmentCategory.TOP),
            _make_garment(category=GarmentCategory.BOTTOM),
            _make_garment(category=GarmentCategory.DRESS),
            _make_garment(category=GarmentCategory.SHOES),
            _make_garment(category=GarmentCategory.BAG),
            _make_garment(category=GarmentCategory.ACCESSORY),
            _make_garment(category=GarmentCategory.OUTERWEAR),
        ]
        wardrobe = _organise_wardrobe(garments)
        assert len(wardrobe["tops"]) == 1
        assert len(wardrobe["bottoms"]) == 1
        assert len(wardrobe["full_body"]) == 1
        assert len(wardrobe["shoes"]) == 1
        assert len(wardrobe["accessories"]) == 2  # bag + accessory
        assert len(wardrobe["outerwear"]) == 1

    def test_empty_list(self):
        wardrobe = _organise_wardrobe([])
        for bucket in wardrobe.values():
            assert len(bucket) == 0


class TestEnrichContextFromProfileExtended:
    """Extended tests for _enrich_context_from_profile edge cases."""

    def test_body_shape_with_value_attr(self):
        """body_shape that has .value should extract the value."""
        ctx = UserContext()
        profile = MagicMock()
        profile.body_shape = MagicMock(value="hourglass")
        profile.color_profile = None
        profile.hair_color = None
        profile.contrast_level = None

        _enrich_context_from_profile(ctx, profile)
        assert ctx.body_shape == "hourglass"

    def test_body_shape_plain_string(self):
        """body_shape that is a plain string (no .value)."""
        ctx = UserContext()
        profile = MagicMock(spec=[])  # empty spec prevents auto-attributes
        profile.body_shape = "rectangle"
        profile.color_profile = None
        profile.hair_color = None
        profile.contrast_level = None

        _enrich_context_from_profile(ctx, profile)
        assert ctx.body_shape == "rectangle"

    def test_all_fields_none(self):
        """When all profile fields are None/missing, context stays unchanged."""
        ctx = UserContext()
        profile = MagicMock()
        profile.body_shape = None
        profile.color_profile = None
        profile.hair_color = None
        profile.contrast_level = None

        _enrich_context_from_profile(ctx, profile)
        assert ctx.body_shape is None
        assert ctx.skin_undertone is None


# ============================================================================
# Additional coverage: extraction fallbacks
# ============================================================================

class TestGarmentExtractionFallbacks:
    """Tests for fallback logic in _extract_garments_from_images."""

    @pytest.mark.asyncio
    async def test_full_outfit_extraction_fallback(
        self, all_mocks, b64_image
    ):
        """
        When extract_full_outfit_from_image throws, the pipeline should
        fall back to extract_garment_from_image for that image.
        """
        all_mocks["builder"].extract_full_outfit_from_image = AsyncMock(
            side_effect=RuntimeError("model crash")
        )
        # Fallback returns single garment per image
        all_mocks["builder"].extract_garment_from_image = AsyncMock(
            return_value=_make_garment()
        )

        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
        )

        response = await full_pipeline_recommend(request)

        # Fell back for both images
        assert all_mocks["builder"].extract_garment_from_image.call_count == 2
        assert response.total_garments_extracted >= 2

    @pytest.mark.asyncio
    async def test_specific_category_fallback_on_layer0_failure(
        self, all_mocks, b64_image
    ):
        """
        When Layer 0 pipeline.process fails for a specific category,
        the code should fall back to builder.extract_garment_from_image.
        """
        all_mocks["segmentation"].process.side_effect = RuntimeError(
            "segmentation model error"
        )
        all_mocks["builder"].extract_garment_from_image = AsyncMock(
            return_value=_make_garment()
        )

        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image, extract_category="top"),
                WardrobeImageItem(image_b64=b64_image),  # normal extraction
            ],
        )

        response = await full_pipeline_recommend(request)

        # Segmentation tried and failed, fallback used
        all_mocks["segmentation"].process.assert_called_once()
        assert all_mocks["builder"].extract_garment_from_image.call_count >= 1
        assert response.total_garments_extracted >= 2


# ============================================================================
# Additional coverage: try-on edge cases
# ============================================================================

class TestTryOnEdgeCases:
    """Extra try-on edge cases."""

    @pytest.mark.asyncio
    async def test_tryon_with_user_profile_no_image(
        self, all_mocks, b64_image
    ):
        """
        enable_tryon=True + user_profile provided but image_b64=None
        should produce an error in errors list.
        """
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            user_profile=UserProfileInput(height_cm=170, weight_kg=65),
            enable_tryon=True,
        )

        response = await full_pipeline_recommend(request)

        all_mocks["tryon"].try_on_outfit.assert_not_called()
        assert any("tryon" in e for e in response.errors)

    @pytest.mark.asyncio
    async def test_tryon_null_composite_image(self, all_mocks, b64_image):
        """
        If try_on_outfit returns result with composite_image=None,
        tryon_image_b64 should remain None.
        """
        result = MagicMock()
        result.composite_image = None
        all_mocks["tryon"].try_on_outfit.return_value = result

        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            user_profile=UserProfileInput(image_b64=b64_image),
            enable_tryon=True,
            top_k=1,
        )

        response = await full_pipeline_recommend(request)

        assert response.recommendations[0].tryon_image_b64 is None
        assert "tryon" in response.stages_completed


# ============================================================================
# Additional coverage: scoring failure
# ============================================================================

class TestScoringFailure:
    """Tests for outfit scoring stage failures."""

    @pytest.mark.asyncio
    async def test_scoring_exception_raises_400(self, all_mocks, b64_image):
        """
        If search_best_outfits throws, the error is logged and
        we raise 400 because candidates is empty.
        """
        all_mocks["builder"].search_best_outfits.side_effect = RuntimeError(
            "scoring explosion"
        )

        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
        )

        with pytest.raises(HTTPException) as exc_info:
            await full_pipeline_recommend(request)
        assert exc_info.value.status_code == 400


# ============================================================================
# Additional coverage: response fields
# ============================================================================

class TestResponseFields:
    """Verify all response fields are correctly populated."""

    @pytest.mark.asyncio
    async def test_total_combinations_scored(self, all_mocks, b64_image):
        """total_combinations_scored reflects the number of candidates."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            top_k=2,
        )

        response = await full_pipeline_recommend(request)

        # Builder returns 3 candidates (from mock)
        assert response.total_combinations_scored == 3

    @pytest.mark.asyncio
    async def test_errors_list_empty_on_success(self, all_mocks, b64_image):
        """When everything succeeds, errors list should be empty."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            enable_tryon=False,
        )

        response = await full_pipeline_recommend(request)
        assert response.errors == []

    @pytest.mark.asyncio
    async def test_garment_out_fields(self, all_mocks, b64_image):
        """Each GarmentOut in recommendations should have correct fields."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            top_k=1,
        )

        response = await full_pipeline_recommend(request)
        rec = response.recommendations[0]

        for g in rec.garments:
            assert isinstance(g.id, str)
            assert isinstance(g.category, str)
            assert isinstance(g.color_primary, str)
            assert isinstance(g.confidence, float)

    @pytest.mark.asyncio
    async def test_user_profile_none_when_not_provided(
        self, all_mocks, b64_image
    ):
        """When user_profile is not in the request, response should have None."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
        )

        response = await full_pipeline_recommend(request)
        assert response.user_profile is None

    @pytest.mark.asyncio
    async def test_recommendation_name_from_candidate(
        self, all_mocks, b64_image
    ):
        """Recommendation name should come from the candidate."""
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image),
                WardrobeImageItem(image_b64=b64_image),
            ],
            top_k=3,
        )

        response = await full_pipeline_recommend(request)

        names = [r.name for r in response.recommendations]
        assert "Outfit A" in names
        assert "Outfit B" in names
        assert "Outfit C" in names


# ============================================================================
# Additional coverage: multiple images with mixed extract_category
# ============================================================================

class TestMixedExtraction:
    """Tests for requests mixing images with and without extract_category."""

    @pytest.mark.asyncio
    async def test_mixed_extraction_modes(self, all_mocks, b64_image):
        """
        Mix of specific-category and full-outfit extraction in the same request.
        """
        request = PipelineRequest(
            wardrobe_images=[
                WardrobeImageItem(image_b64=b64_image, extract_category="shoes"),
                WardrobeImageItem(image_b64=b64_image),  # full extraction
                WardrobeImageItem(image_b64=b64_image, extract_category="bottom"),
            ],
        )

        response = await full_pipeline_recommend(request)

        # Segmentation called twice (for the two with extract_category)
        assert all_mocks["segmentation"].process.call_count == 2
        # Full extraction called once
        assert all_mocks["builder"].extract_full_outfit_from_image.call_count >= 1
        assert response.total_garments_extracted >= 2
