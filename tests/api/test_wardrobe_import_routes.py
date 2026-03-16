"""
API tests for Wardrobe Import Routes
======================================

Tests cover all 8 endpoints of ``/wardrobe/import``:

  POST /wardrobe/import/analyse
  GET  /wardrobe/import/{sid}
  POST /wardrobe/import/{sid}/accept
  POST /wardrobe/import/{sid}/reject
  POST /wardrobe/import/{sid}/correct
  POST /wardrobe/import/{sid}/resubmit
  POST /wardrobe/import/{sid}/finalise
  POST /wardrobe/import/manual

Strategy
--------
- The extraction pipeline is fully mocked so tests run without any ML models.
- A real :class:`ImportSessionManager` is used to verify end-to-end session
  logic through the HTTP layer.
- Each test is isolated: the module-level ``_session_manager`` singleton is
  reset via the ``reset_manager`` fixture.
"""

import io
import json
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ---------------------------------------------------------------------------
# App bootstrap (minimal — only register the import router)
# ---------------------------------------------------------------------------
from fastapi import FastAPI
from src.api.routes.wardrobe_import import router, get_session_manager
import src.api.routes.wardrobe_import as _route_module

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _png_bytes(width=64, height=64, color=(120, 100, 80)) -> bytes:
    """Return the raw bytes of a tiny PNG image."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_fake_garment(label="shirt", area=5000, confidence=0.92):
    """Build an ExtractedGarment-like mock."""
    from src.layer0_segmentation.models import ExtractedGarment, BoundingBox
    from src.layer0_segmentation.taxonomy import GarmentCategory

    img = Image.new("RGBA", (64, 64), (200, 180, 160, 255))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:54, 10:54] = 255
    return ExtractedGarment(
        image=img,
        category=GarmentCategory.TOPS,
        label=label,
        confidence=confidence,
        mask=mask,
        area=area,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_manager():
    """Reset the singleton session manager before every test."""
    _route_module._session_manager = None
    yield
    _route_module._session_manager = None


@pytest.fixture
def mock_pipeline():
    """Patch the Layer 0 pipeline so tests never load ML models."""
    fake_garment = _make_fake_garment()

    mock_result = MagicMock()
    mock_result.garments = [fake_garment]

    mock_pipe = MagicMock()
    mock_pipe.process.return_value = mock_result

    with patch(
        "src.api.routes.wardrobe_import.create_pipeline",
        return_value=mock_pipe,
    ) as p:
        yield p


@pytest.fixture
def session_with_garment(mock_pipeline):
    """Submit a photo and return the session_id + first garment ID."""
    response = client.post(
        "/wardrobe/import/analyse",
        files={"image": ("outfit.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 202, response.text
    data = response.json()["data"]
    session_id = data["session_id"]

    garments = data.get("garments", [])
    sgid = garments[0]["session_garment_id"] if garments else None
    return session_id, sgid


# ---------------------------------------------------------------------------
# 1. POST /wardrobe/import/analyse
# ---------------------------------------------------------------------------

class TestAnalyseEndpoint:
    def test_returns_202_with_session(self, mock_pipeline):
        response = client.post(
            "/wardrobe/import/analyse",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["ok"] is True
        assert "data" in body
        assert "session_id" in body["data"]

    def test_response_has_summary_fields(self, mock_pipeline):
        response = client.post(
            "/wardrobe/import/analyse",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        data = response.json()["data"]
        for field in ("total", "ready", "needs_review", "failed"):
            assert field in data, f"Missing field: {field}"

    def test_response_has_garments_list(self, mock_pipeline):
        response = client.post(
            "/wardrobe/import/analyse",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        data = response.json()["data"]
        assert "garments" in data
        assert isinstance(data["garments"], list)

    def test_no_image_returns_422(self):
        response = client.post("/wardrobe/import/analyse")
        assert response.status_code == 422

    def test_empty_garments_adds_warning(self):
        """If extraction returns no garments, a user-friendly warning is returned."""
        mock_result = MagicMock()
        mock_result.garments = []
        mock_pipe = MagicMock()
        mock_pipe.process.return_value = mock_result

        with patch("src.api.routes.wardrobe_import.create_pipeline", return_value=mock_pipe):
            response = client.post(
                "/wardrobe/import/analyse",
                files={"image": ("empty.png", _png_bytes(), "image/png")},
            )
        assert response.status_code == 202
        body = response.json()
        assert len(body["warnings"]) > 0
        assert body["data"]["total"] == 0

    def test_pipeline_failure_still_returns_202(self):
        """If the pipeline crashes, the route degrades gracefully."""
        with patch(
            "src.api.routes.wardrobe_import.create_pipeline",
            side_effect=RuntimeError("GPU exploded"),
        ):
            response = client.post(
                "/wardrobe/import/analyse",
                files={"image": ("test.png", _png_bytes(), "image/png")},
            )
        assert response.status_code == 202
        body = response.json()
        assert body["data"]["total"] == 0

    def test_invalid_file_returns_400(self):
        response = client.post(
            "/wardrobe/import/analyse",
            files={"image": ("notanimage.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# 2. GET /wardrobe/import/{session_id}
# ---------------------------------------------------------------------------

class TestGetSession:
    def test_get_existing_session(self, session_with_garment):
        session_id, _ = session_with_garment
        response = client.get(f"/wardrobe/import/{session_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["data"]["session_id"] == session_id

    def test_get_nonexistent_session_returns_404(self):
        response = client.get("/wardrobe/import/nonexistent-session-id")
        assert response.status_code == 404

    def test_response_contains_garments(self, session_with_garment):
        session_id, _ = session_with_garment
        response = client.get(f"/wardrobe/import/{session_id}")
        data = response.json()["data"]
        assert "garments" in data


# ---------------------------------------------------------------------------
# 3. POST /wardrobe/import/{sid}/accept
# ---------------------------------------------------------------------------

class TestAcceptEndpoint:
    def test_accept_specific_garment(self, session_with_garment):
        session_id, sgid = session_with_garment
        if sgid is None:
            pytest.skip("No garments in session")

        # Force garment to pending state by setting it up with a low-quality garment
        manager = get_session_manager()
        session = manager.get_session(session_id)
        record = session.garment_records.get(sgid)
        if record:
            from src.layer0_segmentation.session_manager import GarmentDecision
            record.decision = GarmentDecision.PENDING

        response = client.post(
            f"/wardrobe/import/{session_id}/accept",
            json={"session_garment_id": sgid},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert sgid in body["data"]["accepted"]

    def test_bulk_accept_without_session_garment_id(self, session_with_garment):
        session_id, _ = session_with_garment
        response = client.post(
            f"/wardrobe/import/{session_id}/accept",
            json={},
        )
        assert response.status_code == 200
        assert "accepted_count" in response.json()["data"]

    def test_accept_nonexistent_garment_returns_422(self, session_with_garment):
        session_id, _ = session_with_garment
        response = client.post(
            f"/wardrobe/import/{session_id}/accept",
            json={"session_garment_id": "ghost-garment-id"},
        )
        assert response.status_code == 422

    def test_accept_nonexistent_session_returns_404(self):
        response = client.post(
            "/wardrobe/import/bad-session/accept",
            json={"session_garment_id": "g1"},
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# 4. POST /wardrobe/import/{sid}/reject
# ---------------------------------------------------------------------------

class TestRejectEndpoint:
    def test_reject_garment(self, session_with_garment):
        session_id, sgid = session_with_garment
        if sgid is None:
            pytest.skip("No garments in session")

        response = client.post(
            f"/wardrobe/import/{session_id}/reject",
            json={"session_garment_id": sgid},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["data"]["rejected"] == sgid

    def test_reject_nonexistent_garment_returns_422(self, session_with_garment):
        session_id, _ = session_with_garment
        response = client.post(
            f"/wardrobe/import/{session_id}/reject",
            json={"session_garment_id": "ghost"},
        )
        assert response.status_code == 422

    def test_reject_nonexistent_session_returns_404(self):
        response = client.post(
            "/wardrobe/import/bad/reject",
            json={"session_garment_id": "g1"},
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# 5. POST /wardrobe/import/{sid}/correct
# ---------------------------------------------------------------------------

class TestCorrectEndpoint:
    def test_correct_garment_attributes(self, session_with_garment):
        session_id, sgid = session_with_garment
        if sgid is None:
            pytest.skip("No garments in session")

        response = client.post(
            f"/wardrobe/import/{session_id}/correct",
            json={
                "session_garment_id": sgid,
                "corrections": {"color": "ivory", "category": "blouse"},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["data"]["applied_corrections"]["color"] == "ivory"

    def test_corrections_reflected_in_session(self, session_with_garment):
        session_id, sgid = session_with_garment
        if sgid is None:
            pytest.skip("No garments in session")

        client.post(
            f"/wardrobe/import/{session_id}/correct",
            json={"session_garment_id": sgid, "corrections": {"material": "silk"}},
        )
        manager = get_session_manager()
        session = manager.get_session(session_id)
        assert session.garment_records[sgid].user_corrections.get("material") == "silk"

    def test_correct_nonexistent_garment_returns_422(self, session_with_garment):
        session_id, _ = session_with_garment
        response = client.post(
            f"/wardrobe/import/{session_id}/correct",
            json={"session_garment_id": "ghost", "corrections": {"color": "red"}},
        )
        assert response.status_code == 422

    def test_correct_nonexistent_session_returns_404(self):
        response = client.post(
            "/wardrobe/import/bad/correct",
            json={"session_garment_id": "g1", "corrections": {"color": "blue"}},
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# 6. POST /wardrobe/import/{sid}/resubmit
# ---------------------------------------------------------------------------

class TestResubmitEndpoint:
    def test_mark_for_resubmit(self, session_with_garment):
        session_id, sgid = session_with_garment
        if sgid is None:
            pytest.skip("No garments in session")

        response = client.post(
            f"/wardrobe/import/{session_id}/resubmit",
            json={"session_garment_id": sgid},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert "message" in body["data"]

    def test_resubmit_response_contains_friendly_message(self, session_with_garment):
        session_id, sgid = session_with_garment
        if sgid is None:
            pytest.skip("No garments in session")

        response = client.post(
            f"/wardrobe/import/{session_id}/resubmit",
            json={"session_garment_id": sgid},
        )
        message = response.json()["data"]["message"]
        assert len(message) > 10
        assert "wardrobe" in message.lower() or "photo" in message.lower()

    def test_resubmit_nonexistent_garment_returns_422(self, session_with_garment):
        session_id, _ = session_with_garment
        response = client.post(
            f"/wardrobe/import/{session_id}/resubmit",
            json={"session_garment_id": "ghost"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 7. POST /wardrobe/import/{sid}/finalise
# ---------------------------------------------------------------------------

class TestFinaliseEndpoint:
    def test_finalise_returns_added_list(self, session_with_garment):
        session_id, _ = session_with_garment
        response = client.post(f"/wardrobe/import/{session_id}/finalise")
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert "added" in body["data"]
        assert "added_count" in body["data"]

    def test_finalise_rejected_not_in_added(self, session_with_garment):
        session_id, sgid = session_with_garment
        if sgid is None:
            pytest.skip("No garments in session")

        client.post(
            f"/wardrobe/import/{session_id}/reject",
            json={"session_garment_id": sgid},
        )
        response = client.post(f"/wardrobe/import/{session_id}/finalise")
        data = response.json()["data"]
        added_labels = [a.get("garment_id") for a in data["added"]]
        # sgid should not appear in added garments
        assert data["added_count"] == 0

    def test_finalise_includes_corrected_garment(self, session_with_garment):
        session_id, sgid = session_with_garment
        if sgid is None:
            pytest.skip("No garments in session")

        client.post(
            f"/wardrobe/import/{session_id}/correct",
            json={"session_garment_id": sgid, "corrections": {"color": "teal"}},
        )
        response = client.post(f"/wardrobe/import/{session_id}/finalise")
        data = response.json()["data"]
        assert data["added_count"] >= 1

    def test_finalise_nonexistent_session_returns_404(self):
        response = client.post("/wardrobe/import/bad-session/finalise")
        assert response.status_code == 404

    def test_finalise_session_state_set(self, session_with_garment):
        session_id, _ = session_with_garment
        client.post(f"/wardrobe/import/{session_id}/finalise")
        manager = get_session_manager()
        session = manager.get_session(session_id)
        from src.layer0_segmentation.session_manager import SessionState
        assert session.state == SessionState.FINALISED


# ---------------------------------------------------------------------------
# 8. POST /wardrobe/import/manual
# ---------------------------------------------------------------------------

class TestManualEndpoint:
    def test_add_garment_manually(self):
        response = client.post(
            "/wardrobe/import/manual",
            json={
                "category": "accessory",
                "label": "leather belt",
                "color": "tan",
                "material": "leather",
                "notes": "Added manually — too small to detect",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["ok"] is True
        assert body["data"]["label"] == "leather belt"
        assert body["data"]["source"] == "manual"

    def test_manual_returns_garment_id(self):
        response = client.post(
            "/wardrobe/import/manual",
            json={"category": "accessory", "label": "silk scarf"},
        )
        data = response.json()["data"]
        assert "garment_id" in data
        assert len(data["garment_id"]) == 36   # UUID

    def test_manual_optional_fields_allowed(self):
        response = client.post(
            "/wardrobe/import/manual",
            json={"category": "shoes", "label": "white sneakers"},
        )
        assert response.status_code == 201

    def test_manual_missing_required_fields_returns_422(self):
        response = client.post(
            "/wardrobe/import/manual",
            json={"category": "accessory"},   # missing "label"
        )
        assert response.status_code == 422

    def test_manual_status_is_added(self):
        response = client.post(
            "/wardrobe/import/manual",
            json={"category": "bag", "label": "tote bag"},
        )
        assert response.json()["data"]["status"] == "added"


# ---------------------------------------------------------------------------
# Full import flow integration test
# ---------------------------------------------------------------------------

class TestFullImportFlow:
    """
    Simulate a complete user journey:
    analyse → review → correct → reject → finalise
    """

    def test_full_flow(self, mock_pipeline):
        # Make pipeline return 2 garments
        g1 = _make_fake_garment("shirt", area=6000, confidence=0.92)
        g2 = _make_fake_garment("jeans", area=5500, confidence=0.88)
        mock_pipeline.return_value.process.return_value.garments = [g1, g2]

        # 1. Analyse
        resp = client.post(
            "/wardrobe/import/analyse",
            files={"image": ("outfit.png", _png_bytes(), "image/png")},
        )
        assert resp.status_code == 202
        session_id = resp.json()["data"]["session_id"]
        garments = resp.json()["data"]["garments"]
        assert len(garments) == 2

        sgid_0 = garments[0]["session_garment_id"]
        sgid_1 = garments[1]["session_garment_id"]

        # 2. Correct first garment colour
        resp = client.post(
            f"/wardrobe/import/{session_id}/correct",
            json={"session_garment_id": sgid_0, "corrections": {"color": "white"}},
        )
        assert resp.status_code == 200

        # 3. Reject second garment
        resp = client.post(
            f"/wardrobe/import/{session_id}/reject",
            json={"session_garment_id": sgid_1},
        )
        assert resp.status_code == 200

        # 4. Finalise
        resp = client.post(f"/wardrobe/import/{session_id}/finalise")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["added_count"] == 1
        assert data["skipped_count"] == 0

        # Verify correction is in the added garment
        added_garment = data["added"][0]
        assert added_garment.get("user_corrections", {}).get("color") == "white"
