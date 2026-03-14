"""Comprehensive tests for custom outfit API and database functionality."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from typing import Generator
import jwt
import os

from src.main import app
from src.database import Base, get_db
from src.database.models import UserSubscription, CustomOutfit, OutfitAnalysis
from src.api.middleware.auth import create_token, SECRET_KEY, ALGORITHM
from src.api.models.outfits import (
    CustomOutfitCreate, CustomOutfitResponse,
    ImprovementExplanation, OutfitAnalysisRequest,
)


# ── Database Setup for Tests ──

# Use file-based SQLite for testing (more stable than :memory:)
import tempfile
import atexit
test_db_dir = tempfile.mkdtemp()
test_db_file = f"{test_db_dir}/test_outfits.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{test_db_file}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all tables immediately
Base.metadata.create_all(bind=engine)

# Clean up database on exit
def cleanup_test_db():
    import os
    import shutil
    try:
        shutil.rmtree(test_db_dir)
    except:
        pass
atexit.register(cleanup_test_db)


def override_get_db() -> Generator:
    """Override get_db dependency for tests."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


# ── Fixtures ──

# ── Fixtures ──

@pytest.fixture(scope="function", autouse=True)
def setup_test_database():
    """Setup and teardown database for each test."""
    # Create tables before test
    Base.metadata.create_all(bind=engine)
    yield
    # Clean up after test
    with TestingSessionLocal() as db:
        try:
            db.query(OutfitAnalysis).delete()
            db.query(CustomOutfit).delete()
            db.query(UserSubscription).delete()
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()


@pytest.fixture
def db():
    """Database session for tests that need direct DB access."""
    db = TestingSessionLocal()
    yield db
    db.close()


@pytest.fixture
def test_user_id() -> str:
    """Generate a test user ID."""
    return "test-user-12345"


@pytest.fixture
def test_token(test_user_id: str) -> str:
    """Create a valid JWT token for testing."""
    return create_token(test_user_id)


@pytest.fixture
def auth_headers(test_token: str) -> dict:
    """Get authorization headers with valid token."""
    return {"Authorization": f"Bearer {test_token}"}


@pytest.fixture
def premium_user(db, test_user_id: str) -> UserSubscription:
    """Create a premium user."""
    user = UserSubscription(
        user_id=test_user_id,
        is_premium=True,
        subscription_tier="premium",
        expires_at=datetime.utcnow() + timedelta(days=30),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def free_user(db) -> UserSubscription:
    """Create a free-tier user."""
    user = UserSubscription(
        user_id="free-user-99999",
        is_premium=False,
        subscription_tier="free",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def sample_garment_ids() -> list:
    """Sample garment IDs for testing."""
    return [
        "garment-top-001",
        "garment-bottom-002",
        "garment-shoes-003",
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  Authentication Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAuthentication:
    """Test JWT authentication and authorization."""

    def test_create_auth_token(self):
        """Test token creation endpoint."""
        response = client.post("/api/v1/auth/token?user_id=test-user-123")
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 604800

    def test_get_subscription_without_auth(self):
        """Test that endpoints require authentication."""
        response = client.get("/api/v1/subscriptions/me")
        assert response.status_code == 401
        assert "Missing authorization header" in response.json()["detail"]

    def test_get_subscription_with_invalid_token(self):
        """Test with invalid token format."""
        response = client.get(
            "/api/v1/subscriptions/me",
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401

    def test_get_subscription_with_valid_token(self, test_token: str, db, test_user_id: str):
        """Test getting subscription with valid token creates user if needed."""
        response = client.get(
            "/api/v1/subscriptions/me",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == test_user_id
        assert data["is_premium"] is False
        assert data["subscription_tier"] == "free"

    def test_token_expiration(self):
        """Test that expired tokens are rejected."""
        # Create an expired token
        past_time = datetime.utcnow() - timedelta(hours=1)
        payload = {"sub": "user-123", "exp": past_time}
        expired_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

        response = client.get(
            "/api/v1/subscriptions/me",
            headers={"Authorization": f"Bearer {expired_token}"}
        )
        assert response.status_code == 401
        assert "expired" in response.json()["detail"].lower()


# ══════════════════════════════════════════════════════════════════════════════
#  Subscription Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSubscriptions:
    """Test subscription management."""

    def test_get_free_user_subscription(self, auth_headers: dict):
        """Test free user subscription details."""
        response = client.get("/api/v1/subscriptions/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["is_premium"] is False
        assert data["subscription_tier"] == "free"
        assert data["is_active"] is False

    def test_get_premium_user_subscription(self, db, test_user_id: str, test_token: str):
        """Test premium user subscription details."""
        # Create premium user
        user = UserSubscription(
            user_id=test_user_id,
            is_premium=True,
            subscription_tier="premium",
            expires_at=datetime.utcnow() + timedelta(days=30),
        )
        db.add(user)
        db.commit()

        response = client.get(
            "/api/v1/subscriptions/me",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_premium"] is True
        assert data["is_active"] is True

    def test_expired_premium_subscription(self, db, test_user_id: str, test_token: str):
        """Test that expired subscriptions are marked inactive."""
        user = UserSubscription(
            user_id=test_user_id,
            is_premium=True,
            subscription_tier="premium",
            expires_at=datetime.utcnow() - timedelta(days=1),  # Expired
        )
        db.add(user)
        db.commit()

        response = client.get(
            "/api/v1/subscriptions/me",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_premium"] is True
        assert data["is_active"] is False


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Outfit CRUD Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCustomOutfitCRUD:
    """Test custom outfit create, read, update, delete operations."""

    def test_create_custom_outfit_success(
        self, auth_headers: dict, sample_garment_ids: list, db, test_user_id: str
    ):
        """Test successful outfit creation."""
        payload = {
            "outfit_name": "Summer Casual",
            "garment_ids": sample_garment_ids,
            "user_season": "summer",
            "intended_occasion": "casual",
            "notes": "Perfect for casual weekends",
        }
        response = client.post("/api/v1/outfits/custom", json=payload, headers=auth_headers)
        assert response.status_code == 201
        data = response.json()
        assert data["outfit_name"] == "Summer Casual"
        assert data["garment_ids"] == sample_garment_ids
        assert data["user_season"] == "summer"
        assert data["intended_occasion"] == "casual"

        # Verify user counter incremented
        user = db.query(UserSubscription).filter_by(user_id=test_user_id).first()
        assert user.custom_outfits_created == 1

    def test_create_outfit_without_auth(self, sample_garment_ids: list):
        """Test outfit creation requires authentication."""
        payload = {
            "outfit_name": "Test Outfit",
            "garment_ids": sample_garment_ids,
        }
        response = client.post("/api/v1/outfits/custom", json=payload)
        assert response.status_code == 401

    def test_create_outfit_with_empty_garments(self, auth_headers: dict):
        """Test that outfit must have at least one garment."""
        payload = {
            "outfit_name": "Empty Outfit",
            "garment_ids": [],
        }
        response = client.post("/api/v1/outfits/custom", json=payload, headers=auth_headers)
        assert response.status_code == 400
        assert "At least one garment" in response.json()["detail"]

    def test_list_custom_outfits(self, auth_headers: dict, sample_garment_ids: list):
        """Test listing user's outfits."""
        # Create 3 outfits
        for i in range(3):
            payload = {
                "outfit_name": f"Outfit {i+1}",
                "garment_ids": sample_garment_ids,
            }
            client.post("/api/v1/outfits/custom", json=payload, headers=auth_headers)

        response = client.get("/api/v1/outfits/custom", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["outfit_name"] == "Outfit 3"  # Most recent first

    def test_list_outfits_pagination(self, auth_headers: dict, sample_garment_ids: list):
        """Test outfit list pagination."""
        # Create 5 outfits
        for i in range(5):
            payload = {
                "outfit_name": f"Outfit {i+1}",
                "garment_ids": sample_garment_ids,
            }
            client.post("/api/v1/outfits/custom", json=payload, headers=auth_headers)

        # Test limit
        response = client.get(
            "/api/v1/outfits/custom?limit=2",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert len(response.json()) == 2

        # Test skip
        response = client.get(
            "/api/v1/outfits/custom?skip=2&limit=2",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert len(response.json()) == 2

    def test_get_single_outfit(
        self, auth_headers: dict, sample_garment_ids: list
    ):
        """Test retrieving a specific outfit."""
        # Create outfit
        create_payload = {
            "outfit_name": "Test Outfit",
            "garment_ids": sample_garment_ids,
            "user_season": "autumn",
        }
        create_response = client.post(
            "/api/v1/outfits/custom",
            json=create_payload,
            headers=auth_headers
        )
        outfit_id = create_response.json()["id"]

        # Retrieve outfit
        get_response = client.get(
            f"/api/v1/outfits/custom/{outfit_id}",
            headers=auth_headers
        )
        assert get_response.status_code == 200
        data = get_response.json()
        assert data["id"] == outfit_id
        assert data["outfit_name"] == "Test Outfit"
        assert data["latest_analysis"] is None

    def test_get_nonexistent_outfit(self, auth_headers: dict):
        """Test retrieving non-existent outfit returns 404."""
        response = client.get(
            "/api/v1/outfits/custom/nonexistent-id",
            headers=auth_headers
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_update_outfit(self, auth_headers: dict, sample_garment_ids: list):
        """Test updating outfit metadata."""
        # Create outfit
        create_payload = {
            "outfit_name": "Original Name",
            "garment_ids": sample_garment_ids,
            "notes": "Original notes",
        }
        create_response = client.post(
            "/api/v1/outfits/custom",
            json=create_payload,
            headers=auth_headers
        )
        outfit_id = create_response.json()["id"]

        # Update outfit
        update_payload = {
            "outfit_name": "Updated Name",
            "notes": "Updated notes",
            "intended_occasion": "business",
        }
        update_response = client.put(
            f"/api/v1/outfits/custom/{outfit_id}",
            json=update_payload,
            headers=auth_headers
        )
        assert update_response.status_code == 200
        data = update_response.json()
        assert data["outfit_name"] == "Updated Name"
        assert data["notes"] == "Updated notes"
        assert data["intended_occasion"] == "business"
        assert data["saved_at"] is not None

    def test_delete_outfit(self, auth_headers: dict, sample_garment_ids: list):
        """Test deleting an outfit."""
        # Create outfit
        create_payload = {
            "outfit_name": "To Delete",
            "garment_ids": sample_garment_ids,
        }
        create_response = client.post(
            "/api/v1/outfits/custom",
            json=create_payload,
            headers=auth_headers
        )
        outfit_id = create_response.json()["id"]

        # Delete outfit
        delete_response = client.delete(
            f"/api/v1/outfits/custom/{outfit_id}",
            headers=auth_headers
        )
        assert delete_response.status_code == 204

        # Verify it's gone
        get_response = client.get(
            f"/api/v1/outfits/custom/{outfit_id}",
            headers=auth_headers
        )
        assert get_response.status_code == 404

    def test_outfit_isolation_between_users(self, db, test_token: str):
        """Test that users can only see their own outfits."""
        # Create outfit for user 1
        user1_headers = {"Authorization": f"Bearer {test_token}"}
        payload = {
            "outfit_name": "User 1 Outfit",
            "garment_ids": ["garment-1", "garment-2"],
        }
        response = client.post(
            "/api/v1/outfits/custom",
            json=payload,
            headers=user1_headers
        )
        outfit_id = response.json()["id"]

        # Create outfit for user 2
        user2_token = create_token("user-2")
        user2_headers = {"Authorization": f"Bearer {user2_token}"}
        payload2 = {
            "outfit_name": "User 2 Outfit",
            "garment_ids": ["garment-3", "garment-4"],
        }
        client.post("/api/v1/outfits/custom", json=payload2, headers=user2_headers)

        # User 2 should not see User 1's outfit
        get_response = client.get(
            f"/api/v1/outfits/custom/{outfit_id}",
            headers=user2_headers
        )
        assert get_response.status_code == 404

        # User 1 should only see 1 outfit
        list_response = client.get("/api/v1/outfits/custom", headers=user1_headers)
        assert len(list_response.json()) == 1


# ══════════════════════════════════════════════════════════════════════════════
#  Outfit Analysis Tests (Premium Feature)
# ══════════════════════════════════════════════════════════════════════════════

class TestOutfitAnalysis:
    """Test premium outfit analysis feature."""

    def test_analyze_outfit_requires_premium(
        self, auth_headers: dict, sample_garment_ids: list
    ):
        """Test that analysis requires premium subscription."""
        # Create outfit (auto-creates free user)
        create_payload = {
            "outfit_name": "Test Outfit",
            "garment_ids": sample_garment_ids,
        }
        create_response = client.post(
            "/api/v1/outfits/custom",
            json=create_payload,
            headers=auth_headers
        )
        outfit_id = create_response.json()["id"]

        # Try to analyze as free user
        analysis_payload = {"user_season": "summer"}
        response = client.post(
            f"/api/v1/outfits/custom/{outfit_id}/analyze",
            json=analysis_payload,
            headers=auth_headers
        )
        assert response.status_code == 403
        assert "premium" in response.json()["detail"].lower()

    def test_analyze_outfit_premium_success(
        self, db, test_user_id: str, sample_garment_ids: list
    ):
        """Test successful outfit analysis for premium user."""
        # Create premium user
        premium_user = UserSubscription(
            user_id=test_user_id,
            is_premium=True,
            subscription_tier="premium",
            expires_at=datetime.utcnow() + timedelta(days=30),
        )
        db.add(premium_user)
        db.commit()

        token = create_token(test_user_id)
        headers = {"Authorization": f"Bearer {token}"}

        # Create outfit
        create_payload = {
            "outfit_name": "Premium Analysis Test",
            "garment_ids": sample_garment_ids,
        }
        create_response = client.post(
            "/api/v1/outfits/custom",
            json=create_payload,
            headers=headers
        )
        outfit_id = create_response.json()["id"]

        # Analyze outfit
        analysis_payload = {
            "user_season": "summer",
            "body_shape": "hourglass",
        }
        response = client.post(
            f"/api/v1/outfits/custom/{outfit_id}/analyze",
            json=analysis_payload,
            headers=headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["outfit_id"] == outfit_id
        assert data["user_season"] == "summer"
        assert data["body_shape"] == "hourglass"
        assert data["improvement_explanation"] is not None

        # Verify analysis counter incremented
        db.refresh(premium_user)
        assert premium_user.analyses_performed == 1

    def test_analyze_empty_outfit(
        self, db, test_user_id: str
    ):
        """Test that analyzing outfit without garments fails gracefully."""
        # Create premium user
        premium_user = UserSubscription(
            user_id=test_user_id,
            is_premium=True,
            subscription_tier="premium",
            expires_at=datetime.utcnow() + timedelta(days=30),
        )
        db.add(premium_user)

        # Create outfit with empty garments
        outfit = CustomOutfit(
            user_id=test_user_id,
            outfit_name="Empty Outfit",
            garment_ids=[],
        )
        db.add(outfit)
        db.commit()

        token = create_token(test_user_id)
        headers = {"Authorization": f"Bearer {token}"}

        response = client.post(
            f"/api/v1/outfits/custom/{outfit.id}/analyze",
            json={"user_season": "summer"},
            headers=headers
        )
        assert response.status_code == 400
        assert "without garments" in response.json()["detail"]

    def test_get_outfit_analyses(
        self, db, test_user_id: str, sample_garment_ids: list
    ):
        """Test retrieving all analyses for an outfit."""
        # Create premium user and outfit
        premium_user = UserSubscription(
            user_id=test_user_id,
            is_premium=True,
            subscription_tier="premium",
            expires_at=datetime.utcnow() + timedelta(days=30),
        )
        db.add(premium_user)
        db.commit()

        outfit = CustomOutfit(
            user_id=test_user_id,
            outfit_name="Multi-Analysis Outfit",
            garment_ids=sample_garment_ids,
        )
        db.add(outfit)
        db.commit()

        # Create multiple analyses
        for i in range(3):
            analysis = OutfitAnalysis(
                outfit_id=outfit.id,
                user_id=test_user_id,
                user_season="summer" if i % 2 == 0 else "autumn",
                improvement_explanation={"additions": [], "replacements": [], "purchases": []},
            )
            db.add(analysis)
        db.commit()

        token = create_token(test_user_id)
        headers = {"Authorization": f"Bearer {token}"}

        response = client.get(
            f"/api/v1/outfits/custom/{outfit.id}/analyses",
            headers=headers
        )
        assert response.status_code == 200
        analyses = response.json()
        assert len(analyses) == 3
        # Most recent should be first
        assert analyses[0]["generated_at"] >= analyses[1]["generated_at"]


# ══════════════════════════════════════════════════════════════════════════════
#  Database Model Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDatabaseModels:
    """Test database models and relationships."""

    def test_user_subscription_is_active(self, db):
        """Test subscription active status checks."""
        # Free user
        free_user = UserSubscription(user_id="free", is_premium=False)
        assert not free_user.is_active()

        # Expired premium
        expired_user = UserSubscription(
            user_id="expired",
            is_premium=True,
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        assert not expired_user.is_active()

        # Active premium
        active_user = UserSubscription(
            user_id="active",
            is_premium=True,
            expires_at=datetime.utcnow() + timedelta(days=30),
        )
        assert active_user.is_active()

        # Premium with no expiry
        unlimited_user = UserSubscription(
            user_id="unlimited",
            is_premium=True,
            expires_at=None,
        )
        assert unlimited_user.is_active()

    def test_outfit_to_dict(self, db):
        """Test CustomOutfit serialization."""
        outfit = CustomOutfit(
            user_id="test-user",
            outfit_name="Test Outfit",
            garment_ids=["g1", "g2", "g3"],
            notes="Test notes",
            overall_score=0.85,
            score_grade="A",
        )
        db.add(outfit)
        db.commit()

        outfit_dict = outfit.to_dict()
        assert outfit_dict["outfit_name"] == "Test Outfit"
        assert outfit_dict["garment_ids"] == ["g1", "g2", "g3"]
        assert outfit_dict["overall_score"] == 0.85
        assert outfit_dict["score_grade"] == "A"
        assert outfit_dict["created_at"] is not None

    def test_analysis_to_dict(self, db):
        """Test OutfitAnalysis serialization."""
        analysis = OutfitAnalysis(
            outfit_id="outfit-123",
            user_id="user-456",
            analysis_type="improvement",
            user_season="summer",
            body_shape="pear",
            improvement_explanation={"additions": [], "replacements": [], "purchases": []},
        )
        db.add(analysis)
        db.commit()

        analysis_dict = analysis.to_dict()
        assert analysis_dict["outfit_id"] == "outfit-123"
        assert analysis_dict["user_id"] == "user-456"
        assert analysis_dict["analysis_type"] == "improvement"
        assert analysis_dict["user_season"] == "summer"
        assert analysis_dict["improvement_explanation"] is not None

    def test_user_outfit_relationship(self, db, test_user_id: str):
        """Test relationship between users and outfits."""
        user = UserSubscription(user_id=test_user_id)
        db.add(user)
        db.commit()

        # Create outfits
        for i in range(3):
            outfit = CustomOutfit(
                user_id=test_user_id,
                outfit_name=f"Outfit {i}",
                garment_ids=[f"g{i}"],
            )
            db.add(outfit)
        db.commit()

        user = db.query(UserSubscription).filter_by(user_id=test_user_id).first()
        assert len(user.custom_outfits) == 3

    def test_outfit_analysis_relationship(self, db, test_user_id: str):
        """Test relationship between outfits and analyses."""
        outfit = CustomOutfit(
            user_id=test_user_id,
            outfit_name="Test Outfit",
            garment_ids=["g1", "g2"],
        )
        db.add(outfit)
        db.commit()

        # Create analyses
        for i in range(2):
            analysis = OutfitAnalysis(
                outfit_id=outfit.id,
                user_id=test_user_id,
                user_season="summer" if i == 0 else "autumn",
            )
            db.add(analysis)
        db.commit()

        outfit = db.query(CustomOutfit).filter_by(id=outfit.id).first()
        assert len(outfit.analyses) == 2


# ══════════════════════════════════════════════════════════════════════════════
#  Integration Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_premium_workflow(self, db):
        """Test complete workflow: signup → create outfit → analyze."""
        user_id = "integration-test-user"

        # 1. Create token
        token = create_token(user_id)
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Automatically created free user
        response = client.get("/api/v1/subscriptions/me", headers=headers)
        assert response.status_code == 200
        assert response.json()["is_premium"] is False

        # 3. Upgrade to premium
        user = db.query(UserSubscription).filter_by(user_id=user_id).first()
        user.is_premium = True
        user.subscription_tier = "premium"
        user.expires_at = datetime.utcnow() + timedelta(days=30)
        db.commit()

        # 4. Create outfit
        create_payload = {
            "outfit_name": "Integration Test Outfit",
            "garment_ids": ["g1", "g2", "g3"],
            "user_season": "summer",
        }
        response = client.post("/api/v1/outfits/custom", json=create_payload, headers=headers)
        assert response.status_code == 201
        outfit_id = response.json()["id"]

        # 5. Analyze outfit (premium)
        response = client.post(
            f"/api/v1/outfits/custom/{outfit_id}/analyze",
            json={"user_season": "summer"},
            headers=headers
        )
        assert response.status_code == 201

        # 6. Verify counters incremented
        db.refresh(user)
        assert user.custom_outfits_created == 1
        assert user.analyses_performed == 1

    def test_multiple_outfits_per_user(self):
        """Test user can create and manage multiple outfits."""
        token = create_token("multi-outfit-user")
        headers = {"Authorization": f"Bearer {token}"}

        outfit_ids = []
        for i in range(5):
            payload = {
                "outfit_name": f"Outfit {i+1}",
                "garment_ids": [f"g{i}-1", f"g{i}-2", f"g{i}-3"],
                "intended_occasion": ["casual", "business", "formal", "sport", "evening"][i],
            }
            response = client.post("/api/v1/outfits/custom", json=payload, headers=headers)
            assert response.status_code == 201
            outfit_ids.append(response.json()["id"])

        # List all
        response = client.get("/api/v1/outfits/custom", headers=headers)
        assert len(response.json()) == 5

        # Update one
        update_response = client.put(
            f"/api/v1/outfits/custom/{outfit_ids[0]}",
            json={"outfit_name": "Outfit 1 - Updated"},
            headers=headers
        )
        assert update_response.json()["outfit_name"] == "Outfit 1 - Updated"

        # Delete one
        delete_response = client.delete(
            f"/api/v1/outfits/custom/{outfit_ids[1]}",
            headers=headers
        )
        assert delete_response.status_code == 204

        # Verify count
        response = client.get("/api/v1/outfits/custom", headers=headers)
        assert len(response.json()) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
