"""End-to-end tests for the Collaborative Filtering subsystem.

These tests exercise the full pipeline path including the optional
Stage 3c CF boost, verifying:

1. Pipeline with ``enable_cf_boost=True`` adds CF stage.
2. Pipeline with ``enable_cf_boost=False`` skips CF.
3. CF crash is non-blocking — pipeline still completes.
"""
from __future__ import annotations

import atexit
import tempfile
from typing import Any, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app
from src.database import Base, get_db
from src.api.middleware.auth import create_token
from src.layer7_cf.cf_engine import reset_cf_engine
from src.layer7_cf.models import CFScore

# ---------------------------------------------------------------------------
# In-memory test DB
# ---------------------------------------------------------------------------

test_db_dir = tempfile.mkdtemp()
test_db_file = f"{test_db_dir}/test_cf_e2e.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{test_db_file}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def cleanup():
    import shutil
    try:
        shutil.rmtree(test_db_dir)
    except Exception:
        pass


atexit.register(cleanup)


def override_get_db() -> Generator:
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_engine():
    reset_cf_engine()
    yield
    reset_cf_engine()


@pytest.fixture
def user_id() -> str:
    return "cf-e2e-test-user"


@pytest.fixture
def auth_headers(user_id) -> dict:
    return {"Authorization": f"Bearer {create_token(user_id)}"}


# ---------------------------------------------------------------------------
# Helpers — mock the heavy pipeline layers so only CF path is exercised
# ---------------------------------------------------------------------------

def _dummy_image_b64() -> str:
    """Return a minimal 1×1 white PNG as base-64."""
    import base64
    # 1×1 white pixel PNG
    pixel = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return base64.b64encode(pixel).decode()


def _mock_cf_engine(trained: bool = True):
    """Return a mock CFEngine."""
    eng = MagicMock()
    eng.cf.is_trained = trained
    eng.cf.get_boost_score.return_value = CFScore(garment_id="g1", score=0.8, confidence=0.7)
    eng.recommender.rerank.side_effect = lambda cands, uid, cf: [
        {**c, "style_score": c["overall_score"], "cf_score": 0.8,
         "combined_score": 0.7 * c["overall_score"] + 0.3 * 0.8,
         "personalization_active": True}
        for c in cands
    ]
    return eng


def _make_mock_candidate(name: str, score: float, garments):
    """Build a MagicMock candidate with all attributes the pipeline response builder expects."""
    cand = MagicMock()
    cand.name = name
    cand.overall_score = score
    cand.garments = garments
    # scorecard.scores must be a real dict for Pydantic validation
    cand.scorecard = MagicMock()
    cand.scorecard.scores = {"color": 0.8, "style": 0.7}
    cand.__gt__ = lambda self, other: self.overall_score > other.overall_score
    cand.__lt__ = lambda self, other: self.overall_score < other.overall_score
    return cand


# ===========================================================================
# E2E tests
# ===========================================================================

@pytest.mark.e2e
class TestCFPipelineE2E:
    """Test the /pipeline/recommend endpoint with enable_cf_boost."""

    _PIPELINE_URL = "/api/v1/pipeline/recommend"

    def _make_pipeline_request(self, enable_cf: bool = False, user_id: str = "e2e-user"):
        return {
            "wardrobe_images": [
                {"image_b64": _dummy_image_b64()},
                {"image_b64": _dummy_image_b64()},
            ],
            "context": {"occasion": "casual"},
            "top_k": 2,
            "enable_explanation": False,
            "enable_visualization": False,
            "enable_tryon": False,
            "enable_cf_boost": enable_cf,
            "user_id": user_id if enable_cf else None,
        }

    def test_pipeline_with_cf_boost_adds_stage(self, auth_headers):
        """When enable_cf_boost=True and CF trained, 'cf_boost' appears in stages."""
        from src.core.models import (
            Garment, GarmentAttributes, GarmentCategory,
            ColorInfo, FormalityLevel, Season,
        )

        # Create mock garments
        top = Garment(
            id="top_1",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP, subcategory="t-shirt",
                color=ColorInfo(primary="white", hex_codes=[]),
                style_tags=["casual"], formality_level=FormalityLevel.CASUAL,
                season_suitable=[Season.SUMMER],
            ),
        )
        bottom = Garment(
            id="bottom_1",
            attributes=GarmentAttributes(
                category=GarmentCategory.BOTTOM, subcategory="jeans",
                color=ColorInfo(primary="blue", hex_codes=[]),
                style_tags=["casual"], formality_level=FormalityLevel.CASUAL,
                season_suitable=[Season.SUMMER],
            ),
        )

        # Mock all heavy layers
        mock_garments = AsyncMock(return_value=[top, bottom])
        mock_builder = MagicMock()

        mock_candidate = _make_mock_candidate("Casual Outfit", 0.85, [top, bottom])
        mock_builder.search_best_outfits.return_value = [mock_candidate]

        mock_context = MagicMock()
        mock_context.apply_context = AsyncMock(return_value=[])

        cf_engine = _mock_cf_engine(trained=True)

        with patch("src.api.routes.pipeline._extract_garments_from_images", mock_garments), \
             patch("src.api.routes.pipeline.get_outfit_builder", return_value=mock_builder), \
             patch("src.api.routes.pipeline.get_context_engine", return_value=mock_context), \
             patch("src.layer7_cf.cf_engine.get_cf_engine", return_value=cf_engine):

            resp = client.post(
                self._PIPELINE_URL,
                json=self._make_pipeline_request(enable_cf=True, user_id="u0"),
                headers=auth_headers,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "cf_boost" in body["stages_completed"]

    def test_pipeline_without_cf_boost_skips_stage(self, auth_headers):
        """When enable_cf_boost=False, 'cf_boost' does NOT appear in stages."""
        from src.core.models import (
            Garment, GarmentAttributes, GarmentCategory,
            ColorInfo, FormalityLevel, Season,
        )

        top = Garment(
            id="top_2",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP, subcategory="blouse",
                color=ColorInfo(primary="red", hex_codes=[]),
                style_tags=["casual"], formality_level=FormalityLevel.CASUAL,
                season_suitable=[Season.SUMMER],
            ),
        )
        bottom = Garment(
            id="bottom_2",
            attributes=GarmentAttributes(
                category=GarmentCategory.BOTTOM, subcategory="skirt",
                color=ColorInfo(primary="black", hex_codes=[]),
                style_tags=["casual"], formality_level=FormalityLevel.CASUAL,
                season_suitable=[Season.SUMMER],
            ),
        )

        mock_garments = AsyncMock(return_value=[top, bottom])
        mock_builder = MagicMock()
        mock_candidate = _make_mock_candidate("Chic Outfit", 0.80, [top, bottom])
        mock_builder.search_best_outfits.return_value = [mock_candidate]
        mock_context = MagicMock()
        mock_context.apply_context = AsyncMock(return_value=[])

        with patch("src.api.routes.pipeline._extract_garments_from_images", mock_garments), \
             patch("src.api.routes.pipeline.get_outfit_builder", return_value=mock_builder), \
             patch("src.api.routes.pipeline.get_context_engine", return_value=mock_context):

            resp = client.post(
                self._PIPELINE_URL,
                json=self._make_pipeline_request(enable_cf=False),
                headers=auth_headers,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "cf_boost" not in body["stages_completed"]


@pytest.mark.e2e
class TestCFCrashResilience:
    """Verify that CF failures don't break the pipeline."""

    def test_cf_exception_is_non_blocking(self, auth_headers):
        """If CF engine raises, the pipeline still returns 200."""
        from src.core.models import (
            Garment, GarmentAttributes, GarmentCategory,
            ColorInfo, FormalityLevel, Season,
        )

        top = Garment(
            id="top_3",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP, subcategory="shirt",
                color=ColorInfo(primary="green", hex_codes=[]),
                style_tags=["casual"], formality_level=FormalityLevel.CASUAL,
                season_suitable=[Season.SUMMER],
            ),
        )
        bottom = Garment(
            id="bottom_3",
            attributes=GarmentAttributes(
                category=GarmentCategory.BOTTOM, subcategory="trousers",
                color=ColorInfo(primary="khaki", hex_codes=[]),
                style_tags=["casual"], formality_level=FormalityLevel.CASUAL,
                season_suitable=[Season.SUMMER],
            ),
        )

        mock_garments = AsyncMock(return_value=[top, bottom])
        mock_builder = MagicMock()
        mock_candidate = _make_mock_candidate("Crash Test", 0.75, [top, bottom])
        mock_builder.search_best_outfits.return_value = [mock_candidate]
        mock_context = MagicMock()
        mock_context.apply_context = AsyncMock(return_value=[])

        # Make get_cf_engine raise an exception
        def _boom():
            raise RuntimeError("CF engine exploded!")

        with patch("src.api.routes.pipeline._extract_garments_from_images", mock_garments), \
             patch("src.api.routes.pipeline.get_outfit_builder", return_value=mock_builder), \
             patch("src.api.routes.pipeline.get_context_engine", return_value=mock_context), \
             patch("src.layer7_cf.cf_engine.get_cf_engine", side_effect=_boom):

            resp = client.post(
                "/api/v1/pipeline/recommend",
                json={
                    "wardrobe_images": [
                        {"image_b64": _dummy_image_b64()},
                        {"image_b64": _dummy_image_b64()},
                    ],
                    "context": {"occasion": "casual"},
                    "top_k": 1,
                    "enable_explanation": False,
                    "enable_visualization": False,
                    "enable_tryon": False,
                    "enable_cf_boost": True,
                    "user_id": "crash-user",
                },
                headers=auth_headers,
            )

        assert resp.status_code == 200
        body = resp.json()
        # CF stage should NOT appear in stages
        assert "cf_boost" not in body["stages_completed"]
        # But the error should be recorded
        assert any("cf_boost" in e for e in body["errors"])
