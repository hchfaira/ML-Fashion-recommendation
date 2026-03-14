## Custom Outfit Builder API & Database Implementation

This document describes the newly implemented API routes, database schema, and authentication system for the custom outfit builder with premium user features.

### Overview

The custom outfit builder allows users to:
- **Create custom outfits** from selected garments
- **Manage outfits** (CRUD operations)
- **Analyze outfits** (premium feature) using the Layer 4 LLM for styling tips
- **Track subscription status** and premium access

### Architecture

#### 1. Database Schema (SQLAlchemy ORM)

**Location:** `src/database/models.py`

Three main tables:

##### `user_subscriptions`
Tracks user subscription status and premium access.

```python
- id: String (UUID)
- user_id: String (unique, indexed)
- is_premium: Boolean (default: False)
- subscription_tier: String (free | basic | pro | premium)
- created_at: DateTime
- expires_at: DateTime (optional, None if unlimited)
- last_renewed_at: DateTime (optional)
- custom_outfits_created: Integer (counter)
- analyses_performed: Integer (counter)
```

**Methods:**
- `is_active()`: Check if subscription is currently active

##### `custom_outfits`
Stores user-created outfits.

```python
- id: String (UUID)
- user_id: String (foreign key to user_subscriptions)
- outfit_name: String
- garment_ids: JSON (list of garment IDs)
- overall_score: Float (optional)
- score_grade: String (optional, e.g., "A+", "B")
- user_season: String (optional, e.g., "autumn", "summer")
- intended_occasion: String (optional)
- notes: Text (optional)
- created_at: DateTime
- saved_at: DateTime (optional)
- last_analyzed_at: DateTime (optional)
```

**Relationships:**
- `user`: Reference to UserSubscription
- `analyses`: List of OutfitAnalysis records

##### `outfit_analyses`
Caches LLM-generated outfit improvement explanations.

```python
- id: String (UUID)
- outfit_id: String (foreign key to custom_outfits)
- user_id: String (foreign key to user_subscriptions)
- analysis_type: String (improvement | explanation | styling_tip)
- improvement_explanation: JSON
  {
    "additions": [{"item": str, "reason": str}, ...],
    "replacements": [{"original": str, "replacement": str, "reason": str}, ...],
    "purchases": [{"dimension": str, "description": str, "reason": str}, ...]
  }
- styling_tips: JSON (optional)
- generated_at: DateTime
- user_season: String (optional)
- body_shape: String (optional)
```

### 2. Authentication & Authorization

**Location:** `src/api/middleware/auth.py`

#### JWT Token Management
- `create_token(user_id, expires_delta)`: Creates a JWT token
- Default expiration: 7 days
- Algorithm: HS256
- Secret: `SECRET_KEY` environment variable (default: "dev-secret-key-change-in-production")

#### Dependencies

##### `get_current_user`
FastAPI dependency for protected endpoints.
- Extracts JWT token from `Authorization: Bearer <token>` header
- Auto-creates free user subscription if needed
- Returns: `UserSubscription` model

```python
@app.get("/api/v1/protected")
async def protected_endpoint(user: UserSubscription = Depends(get_current_user)):
    return {"user_id": user.user_id}
```

##### `require_premium`
FastAPI dependency for premium-only features.
- Calls `get_current_user` internally
- Checks `is_premium` flag and subscription expiration
- Returns: `UserSubscription` (with premium verified)
- Raises: `HTTPException(403)` if not premium

```python
@app.post("/api/v1/premium-feature")
async def premium_feature(user: UserSubscription = Depends(require_premium)):
    # User is guaranteed to be premium here
    return {"message": "Premium feature accessed"}
```

### 3. API Routes

**Location:** `src/api/routes/custom_outfits.py`

#### Authentication Endpoints

##### `POST /api/v1/auth/token`
Create authentication token for a user.

**Parameters:**
- `user_id` (query string): User identifier

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 604800
}
```

#### Subscription Management

##### `GET /api/v1/subscriptions/me`
Get current user's subscription details.

**Authentication:** Required (Bearer token)

**Response:**
```json
{
  "id": "uuid",
  "user_id": "user-123",
  "is_premium": false,
  "subscription_tier": "free",
  "is_active": false,
  "created_at": "2024-01-15T10:30:00",
  "expires_at": null,
  "custom_outfits_created": 0,
  "analyses_performed": 0
}
```

#### Custom Outfit CRUD

##### `POST /api/v1/outfits/custom`
Create a new custom outfit.

**Authentication:** Required

**Request Body:**
```json
{
  "outfit_name": "Summer Casual",
  "garment_ids": ["garment-001", "garment-002", "garment-003"],
  "user_season": "summer",
  "intended_occasion": "casual",
  "notes": "Perfect for weekend brunches"
}
```

**Response:** `201 Created` with created outfit

**Validations:**
- Outfit name required, 1-255 characters
- At least one garment required
- Returns 400 if validation fails

##### `GET /api/v1/outfits/custom`
List user's custom outfits.

**Authentication:** Required

**Query Parameters:**
- `skip` (int, ≥0, default 0): Pagination offset
- `limit` (int, 1-100, default 10): Results per page

**Response:** `200 OK` with list of outfits

**Ordering:** Most recent first (by `created_at`)

##### `GET /api/v1/outfits/custom/{outfit_id}`
Get specific outfit with latest analysis.

**Authentication:** Required

**Path Parameters:**
- `outfit_id` (string): Outfit UUID

**Response:** `200 OK`
```json
{
  "id": "outfit-uuid",
  "user_id": "user-123",
  "outfit_name": "Summer Casual",
  "garment_ids": [...],
  "overall_score": 0.82,
  "score_grade": "A",
  "user_season": "summer",
  "intended_occasion": "casual",
  "notes": "...",
  "created_at": "2024-01-15T...",
  "saved_at": null,
  "last_analyzed_at": null,
  "latest_analysis": null  // or OutfitAnalysis if available
}
```

**Errors:**
- `404 Not Found`: Outfit not found or user doesn't own it

##### `PUT /api/v1/outfits/custom/{outfit_id}`
Update outfit metadata.

**Authentication:** Required

**Request Body (all optional):**
```json
{
  "outfit_name": "Updated Name",
  "notes": "Updated notes",
  "intended_occasion": "business"
}
```

**Response:** `200 OK` with updated outfit

**Side Effects:**
- Sets `saved_at` to current timestamp

##### `DELETE /api/v1/outfits/custom/{outfit_id}`
Delete outfit and associated analyses.

**Authentication:** Required

**Response:** `204 No Content`

**Side Effects:**
- Cascades delete to all `outfit_analyses` records

#### Outfit Analysis (Premium Feature)

##### `POST /api/v1/outfits/custom/{outfit_id}/analyze`
Analyze outfit using Layer 4 LLM.

**Authentication:** Required + Premium

**Request Body:**
```json
{
  "user_season": "summer",
  "body_shape": "hourglass"
}
```

**Response:** `201 Created`
```json
{
  "id": "analysis-uuid",
  "outfit_id": "outfit-uuid",
  "user_id": "user-123",
  "analysis_type": "improvement",
  "improvement_explanation": {
    "additions": [...],
    "replacements": [...],
    "purchases": [...]
  },
  "styling_tips": {},
  "generated_at": "2024-01-15T...",
  "user_season": "summer",
  "body_shape": "hourglass"
}
```

**Errors:**
- `403 Forbidden`: User not premium
- `404 Not Found`: Outfit not found
- `400 Bad Request`: Outfit has no garments or analysis failed

**Side Effects:**
- Increments `user.analyses_performed` counter
- Updates `outfit.last_analyzed_at`
- Caches result in database

##### `GET /api/v1/outfits/custom/{outfit_id}/analyses`
Get all analyses for an outfit.

**Authentication:** Required

**Response:** `200 OK` with list of analyses

**Ordering:** Most recent first (by `generated_at`)

### 4. API Models (Pydantic Schemas)

**Location:** `src/api/models/outfits.py`

#### Request Models
- `CustomOutfitCreate`: For POST /outfits/custom
- `CustomOutfitUpdate`: For PUT /outfits/custom/{outfit_id}
- `OutfitAnalysisRequest`: For POST /outfits/custom/{outfit_id}/analyze
- `UserSubscriptionCreate`: For future subscription endpoints
- `UserSubscriptionUpdate`: For future subscription management

#### Response Models
- `CustomOutfitResponse`: Standard outfit data
- `CustomOutfitWithAnalysis`: Outfit + latest analysis
- `OutfitAnalysisResponse`: Analysis data
- `UserSubscriptionResponse`: Subscription data
- `ImprovementExplanation`: Structured LLM output

#### Error Models
- `ErrorResponse`: Base error
- `UnauthorizedResponse`: 401
- `ForbiddenResponse`: 403
- `NotFoundResponse`: 404
- `PremiumRequiredResponse`: 403 with upgrade URL

### 5. Database Setup

**Location:** `src/database/__init__.py`

#### Configuration
```python
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./outfit_builder.db")
```

Supports:
- SQLite (development)
- PostgreSQL (production)
- MySQL (production)

#### Session Management
```python
def get_db():
    """Dependency for injecting DB sessions into routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

#### Database Initialization
```python
def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
```

Call during application startup:
```python
@app.on_event("startup")
async def startup():
    init_db()
```

### 6. Integration with Main App

**Location:** `src/main.py` and `src/api/__init__.py`

The custom outfits router is included in the main API:
```python
router.include_router(custom_outfits.router, prefix="", tags=["Custom Outfits & Premium"])
```

All routes are under `/api/v1/` prefix.

### 7. Testing

**Location:** `tests/api/test_custom_outfits.py`

Comprehensive test suite with 29 tests covering:

#### Test Classes

1. **TestAuthentication** (5 tests)
   - Token creation
   - Missing/invalid/expired tokens
   - Auto-user creation

2. **TestSubscriptions** (3 tests)
   - Free user subscription
   - Premium user subscription
   - Expired subscriptions

3. **TestCustomOutfitCRUD** (10 tests)
   - Create, read, update, delete
   - Empty garments validation
   - Pagination
   - User data isolation

4. **TestOutfitAnalysis** (4 tests)
   - Premium requirement check
   - Successful analysis
   - Empty outfit validation
   - Analysis history

5. **TestDatabaseModels** (5 tests)
   - Model serialization
   - Relationships
   - Subscription active status

6. **TestIntegration** (2 tests)
   - Full workflow (signup → create → analyze → premium upgrade)
   - Multiple outfits per user

#### Running Tests
```bash
# All custom outfit tests
pytest tests/api/test_custom_outfits.py -v

# Specific test class
pytest tests/api/test_custom_outfits.py::TestCustomOutfitCRUD -v

# Single test
pytest tests/api/test_custom_outfits.py::TestCustomOutfitCRUD::test_create_custom_outfit_success -v

# With coverage
pytest tests/api/test_custom_outfits.py --cov=src.api.routes.custom_outfits --cov=src.database
```

#### Test Database
- Uses file-based SQLite for test isolation
- Each test cleans up its data
- Tables are created once per test session

### 8. Usage Examples

#### Creating an Outfit
```python
import requests

# 1. Get auth token
token_response = requests.post("http://localhost:8000/api/v1/auth/token?user_id=user-123")
token = token_response.json()["access_token"]

headers = {"Authorization": f"Bearer {token}"}

# 2. Create outfit
outfit_data = {
    "outfit_name": "Summer Vibes",
    "garment_ids": ["g1", "g2", "g3"],
    "user_season": "summer",
    "intended_occasion": "casual"
}
response = requests.post("http://localhost:8000/api/v1/outfits/custom", json=outfit_data, headers=headers)
outfit_id = response.json()["id"]

# 3. Analyze outfit (requires premium)
analysis_data = {"user_season": "summer", "body_shape": "hourglass"}
response = requests.post(
    f"http://localhost:8000/api/v1/outfits/custom/{outfit_id}/analyze",
    json=analysis_data,
    headers=headers
)
```

### 9. Environment Variables

For production deployment:
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/outfit_db

# JWT
SECRET_KEY=your-super-secret-key-change-this

# API
DEBUG=False
LOG_LEVEL=INFO
```

### 10. Future Enhancements

- [ ] Integrate with Layer 4 LLM for real outfit analysis
- [ ] Add outfit sharing/collaboration
- [ ] Implement wardrobe syncing
- [ ] Add seasonal rotation recommendations
- [ ] Build frontend UI for custom outfit builder
- [ ] Add photo upload for custom garments
- [ ] Implement style quiz for better recommendations
- [ ] Add social features (follow, like outfits)
- [ ] Create mobile app
- [ ] Add push notifications for analysis results

### 11. Error Handling

All errors follow a consistent format:
```json
{
  "detail": "Error message",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-15T10:30:00"
}
```

Common HTTP Status Codes:
- `200 OK`: Successful GET/PUT
- `201 Created`: Successful POST
- `204 No Content`: Successful DELETE
- `400 Bad Request`: Invalid input validation
- `401 Unauthorized`: Missing/invalid authentication
- `403 Forbidden`: Insufficient permissions or premium required
- `404 Not Found`: Resource doesn't exist
- `500 Internal Server Error`: Server error

### 12. Rate Limiting Considerations

For production, recommend adding:
- 100 outfit creations per user per day
- 10 analyses per premium user per day
- 100 analysis requests per user per month (free tier)
- 1000 analysis requests per user per month (premium tier)

