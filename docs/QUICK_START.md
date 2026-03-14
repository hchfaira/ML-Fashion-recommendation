## Quick Start: Custom Outfit Builder API

### Installation

All dependencies are already installed. Verify with:
```bash
cd /home/chfaira-hajar/work/repos/LLM_project
.venv/bin/python -c "import sqlalchemy; import pyjwt; import fastapi; print('✅ All dependencies installed')"
```

### Running Tests

```bash
# All custom outfit tests
.venv/bin/python -m pytest tests/api/test_custom_outfits.py -v

# Specific test category
.venv/bin/python -m pytest tests/api/test_custom_outfits.py::TestCustomOutfitCRUD -v
.venv/bin/python -m pytest tests/api/test_custom_outfits.py::TestOutfitAnalysis -v

# Full test suite (verify no regressions)
.venv/bin/python -m pytest -x
```

### Files Created/Modified

#### New Files
1. `src/database/__init__.py` - Database engine and session factory
2. `src/database/models.py` - ORM models (UserSubscription, CustomOutfit, OutfitAnalysis)
3. `src/api/models/outfits.py` - Pydantic schemas (request/response models)
4. `src/api/middleware/__init__.py` - Middleware package
5. `src/api/middleware/auth.py` - JWT authentication and premium checking
6. `src/api/routes/custom_outfits.py` - 14 API endpoints
7. `tests/api/test_custom_outfits.py` - 29 comprehensive tests
8. `docs/API_DATABASE_CUSTOM_OUTFITS.md` - Full API documentation
9. `docs/IMPLEMENTATION_SUMMARY.md` - This implementation summary

#### Modified Files
1. `src/api/__init__.py` - Added custom_outfits router

### Database Setup

#### For Development (SQLite)
Default configuration uses SQLite. Database file is created automatically:
```
outfit_builder.db
```

To reset the database:
```bash
rm outfit_builder.db
```

#### For Production (PostgreSQL)
Set environment variable:
```bash
export DATABASE_URL="postgresql://user:password@localhost/outfit_db"
```

Initialize schema:
```python
from src.database import init_db
init_db()
```

### Quick API Usage

#### 1. Create Authentication Token
```bash
curl -X POST "http://localhost:8000/api/v1/auth/token?user_id=user-123"
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 604800
}
```

Store the `access_token` for the next requests.

#### 2. Check Subscription Status
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/subscriptions/me
```

#### 3. Create a Custom Outfit
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."
curl -X POST http://localhost:8000/api/v1/outfits/custom \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "outfit_name": "Summer Casual",
    "garment_ids": ["g1", "g2", "g3"],
    "user_season": "summer",
    "intended_occasion": "casual",
    "notes": "Perfect for weekends"
  }'
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user-123",
  "outfit_name": "Summer Casual",
  "garment_ids": ["g1", "g2", "g3"],
  "overall_score": null,
  "score_grade": null,
  "user_season": "summer",
  "intended_occasion": "casual",
  "notes": "Perfect for weekends",
  "created_at": "2024-01-15T10:30:00",
  "saved_at": null,
  "last_analyzed_at": null
}
```

#### 4. List Your Outfits
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/outfits/custom
```

#### 5. Get Specific Outfit
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/outfits/custom/550e8400-e29b-41d4-a716-446655440000
```

#### 6. Update Outfit (Free Feature)
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."
curl -X PUT http://localhost:8000/api/v1/outfits/custom/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "outfit_name": "Updated Summer Casual",
    "notes": "Added a new note"
  }'
```

#### 7. Analyze Outfit (Premium Feature - Will Fail Without Premium)
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."
curl -X POST http://localhost:8000/api/v1/outfits/custom/550e8400-e29b-41d4-a716-446655440000/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_season": "summer",
    "body_shape": "hourglass"
  }'
```

Expected response (without premium):
```json
{
  "detail": "This feature requires a premium subscription. Please upgrade to continue.",
  "error_code": "FORBIDDEN"
}
```

#### 8. To Test Premium Analysis

First, make the user premium (development only):
```python
from src.database import SessionLocal
from src.database.models import UserSubscription
from datetime import datetime, timedelta

db = SessionLocal()
user = db.query(UserSubscription).filter_by(user_id="user-123").first()
user.is_premium = True
user.subscription_tier = "premium"
user.expires_at = datetime.utcnow() + timedelta(days=30)
db.commit()
```

Then retry the analysis:
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."
curl -X POST http://localhost:8000/api/v1/outfits/custom/550e8400-e29b-41d4-a716-446655440000/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_season": "summer",
    "body_shape": "hourglass"
  }'
```

Now returns:
```json
{
  "id": "660f8500-f40c-52e5-b827-557766551111",
  "outfit_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user-123",
  "analysis_type": "improvement",
  "improvement_explanation": {
    "additions": [],
    "replacements": [],
    "purchases": []
  },
  "styling_tips": null,
  "generated_at": "2024-01-15T10:31:00",
  "user_season": "summer",
  "body_shape": "hourglass"
}
```

#### 9. Get Analysis History
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/outfits/custom/550e8400-e29b-41d4-a716-446655440000/analyses
```

#### 10. Delete Outfit
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."
curl -X DELETE http://localhost:8000/api/v1/outfits/custom/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer $TOKEN"
```

Returns `204 No Content` on success.

### Python Usage Example

```python
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"
USER_ID = "user-456"

# 1. Create token
token_resp = requests.post(f"{BASE_URL}/auth/token?user_id={USER_ID}")
token = token_resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Check subscription
sub = requests.get(f"{BASE_URL}/subscriptions/me", headers=headers).json()
print(f"User tier: {sub['subscription_tier']}, Premium: {sub['is_premium']}")

# 3. Create outfit
outfit = requests.post(
    f"{BASE_URL}/outfits/custom",
    headers=headers,
    json={
        "outfit_name": "Business Casual",
        "garment_ids": ["b1", "b2", "b3"],
        "intended_occasion": "business"
    }
).json()
outfit_id = outfit["id"]
print(f"Created outfit: {outfit['outfit_name']} (ID: {outfit_id})")

# 4. List outfits
outfits = requests.get(f"{BASE_URL}/outfits/custom", headers=headers).json()
print(f"Total outfits: {len(outfits)}")

# 5. Try to analyze (will fail without premium)
analysis_resp = requests.post(
    f"{BASE_URL}/outfits/custom/{outfit_id}/analyze",
    headers=headers,
    json={"user_season": "winter"}
)
if analysis_resp.status_code == 403:
    print("❌ Premium required for analysis")
else:
    analysis = analysis_resp.json()
    print(f"✅ Analysis completed")
```

### Integration Points (For Next Phase)

#### 1. Connect Layer 4 LLM
File: `src/api/routes/custom_outfits.py`, line ~335
```python
# REPLACE THIS:
improvement_explanation = ImprovementExplanation(
    additions=[],
    replacements=[],
    purchases=[],
)

# WITH THIS:
from src.layer4_llm.outfit_improvement_explainer import OutfitImprovementExplainer
# Fetch garments from IDs, call LLM, get real improvement_explanation
```

#### 2. Validate Garment IDs
File: `src/api/routes/custom_outfits.py`, line ~80
```python
# ADD VALIDATION:
if not outfit_data.garment_ids:
    raise HTTPException(400, "At least one garment required")

# TODO: Query wardrobe/inventory to ensure garments exist
# TODO: Check user owns these garments
```

#### 3. Calculate Outfit Scores
File: `src/api/routes/custom_outfits.py`, line ~105
```python
# AFTER CREATING OUTFIT:
from src.layer2_style import TotalStyleScorer
scorer = TotalStyleScorer()
# TODO: Calculate outfit.overall_score and outfit.score_grade
```

#### 4. Add Payment Processing
File: `src/api/routes/custom_outfits.py`
```python
# FUTURE: Add Stripe webhook handling
# FUTURE: Update user.expires_at on successful payment
# FUTURE: Add subscription upgrade endpoint
```

### Test Coverage

Current: 29 tests covering
- ✅ Authentication (token creation, expiration, auto-user creation)
- ✅ Subscriptions (free, premium, expired)
- ✅ CRUD operations (create, read, update, delete)
- ✅ Pagination (skip, limit)
- ✅ User isolation (users can't see each other's outfits)
- ✅ Premium gating (403 without premium)
- ✅ Database models (serialization, relationships)
- ✅ End-to-end workflows

Run with:
```bash
.venv/bin/python -m pytest tests/api/test_custom_outfits.py -v
```

### Troubleshooting

#### "no such table: user_subscriptions"
This means the database hasn't been initialized. Run:
```python
from src.database import init_db
init_db()
```

Or delete the .db file and restart:
```bash
rm outfit_builder.db
```

#### "Invalid token" / "Token has expired"
Generate a new token:
```bash
curl -X POST "http://localhost:8000/api/v1/auth/token?user_id=YOUR_USER_ID"
```

#### "Premium required"
Your user isn't marked as premium. Either:
1. Wait for payment processing integration
2. Manually upgrade for testing:
```python
from src.database import SessionLocal
from src.database.models import UserSubscription
from datetime import datetime, timedelta

db = SessionLocal()
user = db.query(UserSubscription).filter_by(user_id="YOUR_USER_ID").first()
user.is_premium = True
user.subscription_tier = "premium"
user.expires_at = datetime.utcnow() + timedelta(days=30)
db.commit()
```

#### Database locked error
SQLite file lock issue. Delete and restart:
```bash
rm outfit_builder.db
```

### Next Steps

1. ✅ **Database & API implemented** ← YOU ARE HERE
2. 📝 Connect Layer 4 LLM to analyze outfits
3. 📝 Integrate with wardrobe/inventory system
4. 📝 Add Stripe for payment processing
5. 📝 Build frontend UI (web/mobile)
6. 📝 Deploy to production

### Support

Refer to full documentation:
- API Reference: `docs/API_DATABASE_CUSTOM_OUTFITS.md`
- Implementation Details: `docs/IMPLEMENTATION_SUMMARY.md`
- Code: Check docstrings in implementation files

