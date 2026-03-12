# Smart Removal Analyzer - Configuration System

## Overview

The `SmartRemovalAnalyzer` is now fully configurable via JSON. All subjective constants and thresholds are externalized into `config/data/smart_removal_config.json`, eliminating hardcoded values and enabling:

- **Multi-profile support**: 6 pre-built profiles + custom profiles
- **User personalization**: Different removal strategies without code changes
- **Easy A/B testing**: Compare removal verdicts across profiles
- **Cultural/regional adaptation**: Customize occasion importance, neutral colors, formality hierarchy

---

## Configuration File

**Location**: `config/data/smart_removal_config.json`

**Structure**:
```json
{
  "profiles": {
    "profile_name": {
      "name": "Human name",
      "description": "Description",
      "recency_half_life_days": 60,
      "formality_flexibility": 1,
      "replacement_coverage_threshold": 0.6,
      "removal_aggressiveness": {
        "donate_threshold": 0.15,
        "safe_to_remove_threshold": 0.30,
        "consider_threshold": 0.60
      },
      "confidence_strictness": {
        "penalty_per_data_gap": 0.15,
        "minimum_confidence_for_action": 0.4
      },
      "goal_weights": {
        "MAXIMIZE_OPTIONS": { ... },
        "MINIMALIST": { ... },
        "STYLE_UPGRADE": { ... }
      },
      "newness_guard": { ... },
      "frequency_soft_cap": { ... },
      "versatility": { ... },
      "replacement_quality_weights": { ... },
      "neutral_colors": [ ... ],
      "occasion_importance": { ... }
    }
  }
}
```

---

## Available Profiles

### 1. **`default`** (Balanced)
**Use case**: General users who value both outfit diversity and regular wears.

**Key settings**:
- Recency decay: 60 days
- Formality flexibility: ±1 level
- Goal weights balanced across all signals
- Occasion importance: Standard distribution (formal=1.0, beach=0.2)

**When to use**: Safe default for most users.

---

### 2. **`minimalist_eco`** (Minimalist, Eco-conscious)
**Use case**: Users who want a small, highly versatile wardrobe. Forget items faster.

**Key settings**:
- Recency decay: **45 days** (forgets sooner)
- Formality flexibility: ±2 levels (more lenient)
- Replacement threshold: **0.85** (very strict - needs perfect match)
- Versatility weight: **0.35** (highest priority)
- Neutral colors: Includes trendy colors (sage, mushroom, taupe)
- Occasion importance: Emphasizes casual/daily (1.0) over formal (0.3)

**When to use**: Minimalists, eco-conscious users, limited closet space.

---

### 3. **`fashion_addict`** (Trend-conscious)
**Use case**: Users who love fashion, trend-following, update wardrobe frequently.

**Key settings**:
- Recency decay: **21 days** (forgets very fast - trends change)
- Formality flexibility: ±3 levels (very lenient)
- Replacement threshold: **0.4** (lenient - "close enough" is OK)
- Newness grace period: **7 days** (quick turnaround testing)
- Rating weight: **0.40** in STYLE_UPGRADE goal (prioritizes love, not logic)
- Neutral colors: Only classic neutrals (black, white, grey)
- Occasion importance: Emphasizes event/cocktail/evening (1.0) over daily (0.3)

**When to use**: Fashion enthusiasts, influencers, trend-followers.

---

### 4. **`budget_conscious`** (Conservative, Budget-aware)
**Use case**: Users with limited budget. Very hesitant to remove items.

**Key settings**:
- Recency decay: **90 days** (keeps items longer)
- Replacement threshold: **0.95** (highest strictness - MUST find perfect replacement)
- Data gap penalty: **0.25** (distrusts incomplete data)
- Verdict thresholds: Very conservative (DONATE=0.05, SAFE=0.15, CONSIDER=0.50)
- Frequency soft cap: **15 max wears** (longer before hitting cap)
- Replacement weight: **0.50** in all goals (dominant)

**When to use**: Budget-conscious users, slow shoppers, maximalists.

---

### 5. **`work_professional`** (Corporate)
**Use case**: Corporate/professional environments. Prioritizes work occasions, strict formality.

**Key settings**:
- Formality flexibility: **1** (strict hierarchy)
- Formality compatibility weight: **0.40** (highest in replacement evaluation)
- Occasion importance: business=1.0, formal=0.9, work=0.95; casual=0.1, beach=0.0
- Outfit impact weight: **0.30** (second-highest)
- Neutral colors: Only professional neutrals (black, white, grey, navy, charcoal, ivory)

**When to use**: Corporate professionals, lawyers, executives.

---

### 6. **`data_driven`** (Analytical)
**Use case**: Users who trust data and want comprehensive wear history.

**Key settings**:
- Data gap penalty: **0.30** (highest - demands complete data)
- Minimum confidence for action: **0.70** (very strict)
- Verdict thresholds: Near-default (data-agnostic)
- All signal weights balanced

**When to use**: Analytical users, data enthusiasts, researchers.

---

## Configuration Parameters Explained

### A. **Recency Control**

**Parameter**: `recency_half_life_days` (default: 60)

**Effect**: Controls how quickly items lose "recency attachment"

**Formula**: `score = 0.5 ^ (days_since_worn / half_life)`

| Profile | Days | Meaning |
|---------|------|---------|
| default | 60 | Item worn 60 days ago = 50% score |
| minimalist_eco | 45 | Forgets faster → more willing to remove |
| fashion_addict | 21 | Trends change → very fast decay |
| budget_conscious | 90 | Keep items longer before decay |

---

### B. **Formality Hierarchy**

**Parameter**: `formality_flexibility` (1–3)

**Effect**: How many formality levels apart can items be considered "compatible"?

| Level | Compatibility | Use Case |
|-------|----------------|----------|
| 1 | CASUAL ↔ SMART_CASUAL only | Conservative (work_professional) |
| 2 | CASUAL ↔ BUSINESS | Balanced (default) |
| 3 | CASUAL ↔ FORMAL | Lenient (fashion_addict) |

---

### C. **Replacement Evaluation**

**Parameter**: `replacement_coverage_threshold` (0.4–0.95)

**Effect**: Minimum coverage match to consider a garment a "valid replacement"

| Threshold | Meaning | Use Case |
|-----------|---------|----------|
| 0.40 | Lenient - "close enough works" | fashion_addict |
| 0.60 | Balanced | default |
| 0.85–0.95 | Strict - must cover nearly all use cases | minimalist_eco, budget_conscious |

---

### D. **Removal Aggressiveness**

**Parameter**: `removal_aggressiveness` (3 thresholds)

**Effect**: Maps `regret_risk` [0–1] to verdict

```
regret_risk in [0, donate_threshold]           → DONATE
regret_risk in [donate, safe_remove_threshold] → SAFE_TO_REMOVE
regret_risk in [safe_remove, consider]         → CONSIDER
regret_risk >= consider_threshold              → KEEP
```

| Profile | DONATE | SAFE | CONSIDER | Philosophy |
|---------|--------|------|----------|------------|
| default | 0.15 | 0.30 | 0.60 | Balanced |
| minimalist_eco | 0.10 | 0.25 | 0.55 | Remove more |
| budget_conscious | 0.05 | 0.15 | 0.50 | Remove very conservatively |
| fashion_addict | 0.25 | 0.40 | 0.70 | Remove aggressively |

---

### E. **Confidence & Data Gaps**

**Parameter**: `confidence_strictness`

**Effect**: How harshly to penalize missing data

```python
confidence = 1.0 - (penalty_per_data_gap * num_gaps)
if confidence < minimum_confidence_for_action:
    verdict = "CONSIDER"  # Force safe choice
```

| Profile | Penalty/gap | Min confidence | Philosophy |
|---------|-------------|---|---|
| default | 0.15 | 0.40 | Trusts data reasonably |
| data_driven | 0.30 | 0.70 | Demands comprehensive data |
| fashion_addict | 0.10 | 0.20 | Trusts intuition over data |

---

### F. **Goal Weights**

**Parameters**: `goal_weights[MAXIMIZE_OPTIONS/MINIMALIST/STYLE_UPGRADE]`

**Effect**: How much each signal (recency, frequency, versatility, etc.) affects final `regret_risk`

**Signals**: `recency`, `frequency`, `versatility`, `outfit_impact`, `replacement`, `rating`, `newness`

**Constraints**: Must sum to ~1.0 (tolerance: ±0.05)

**Examples**:

```json
{
  "MAXIMALIST_OPTIONS": {
    "recency": 0.10,      // Not too important
    "frequency": 0.10,    // Not too important
    "versatility": 0.25,  // HIGH - preserve outfit diversity
    "outfit_impact": 0.25, // HIGH - minimize outfit loss
    "replacement": 0.15,
    "rating": 0.10,
    "newness": 0.05
  },
  
  "MINIMALIST": {
    "recency": 0.20,
    "frequency": 0.10,
    "versatility": 0.15,
    "outfit_impact": 0.15,
    "replacement": 0.25,  // HIGH - must find replacement
    "rating": 0.10,
    "newness": 0.05
  },
  
  "STYLE_UPGRADE": {
    "recency": 0.15,
    "frequency": 0.15,
    "versatility": 0.10,
    "outfit_impact": 0.10,
    "replacement": 0.15,
    "rating": 0.25,       // HIGH - prioritize quality/love
    "newness": 0.10
  }
}
```

---

### G. **Newness Guard**

**Parameters**: `newness_guard`

**Effect**: Protects recently acquired items from premature removal

```python
# 0-30 days → score ~1 (keep!), 60 days → ~0.5, 120 days → ~0.25
score = 0.5 ^ (days_owned / half_life)
```

| Profile | Grace (days) | Half-life | Philosophy |
|---------|--------------|-----------|-----------|
| default | 30 | 60 | Give items normal trial time |
| fashion_addict | 7 | 21 | Quick test (return-window friendly) |
| budget_conscious | 60 | 120 | Long trial time for precious items |

---

### H. **Frequency Soft-Cap**

**Parameters**: `frequency_soft_cap`

**Effect**: Normalizes wear count into [0, 1] score

```python
recent_equivalent_wears = sum(0.5 ^ (days_ago / half_life))
score = min(1.0, recent_equivalent_wears / max_wears)
```

| Profile | Max wears | Meaning |
|---------|-----------|---------|
| default | 10 | 10 recent-equivalent wears → 100% |
| minimalist_eco | 8 | Stricter - requires fewer wears for high score |
| budget_conscious | 15 | Lenient - accepts more wears before capping |

---

### I. **Versatility Weights**

**Parameters**: `versatility` sub-component weights

**Effect**: How versatility is computed from season/occasion/category breadth

```python
versatility = (
    season_weight * season_breadth +
    occasion_weight * occasion_breadth +
    category_weight * category_breadth
)
```

**Default profile**:
```json
{
  "season_weight": 0.40,    // 40% - seasons matter
  "occasion_weight": 0.35,  // 35% - occasions matter
  "category_weight": 0.25   // 25% - category pairings matter
}
```

**minimalist_eco**: Season up to 0.45, occasion down to 0.30 (prioritizes seasonality)

---

### J. **Replacement Quality Weights**

**Parameters**: `replacement_quality_weights`

**Effect**: How to score replacement candidates

```python
coverage = (
    season_coverage_weight * season_match +
    formality_compat_weight * formality_match +
    color_role_weight * color_match +
    occasion_coverage_weight * occasion_match
)
```

**Default profile**:
```json
{
  "season_coverage": 0.40,
  "formality_compatibility": 0.30,
  "color_role_match": 0.15,
  "occasion_coverage": 0.15
}
```

**work_professional**: Formality up to 0.40 (most important), occasion down to 0.10

---

### K. **Neutral Colors**

**Parameter**: `neutral_colors` (array of color names)

**Effect**: Determines color-role matching for replacements

**Default profile**:
```json
["black", "white", "grey", "gray", "navy", "beige",
 "cream", "khaki", "brown", "tan", "charcoal", "ivory"]
```

**minimalist_eco** (adds trendy 2024-2026 neutrals):
```json
["black", "white", "grey", "gray", "navy", "beige",
 "cream", "khaki", "brown", "tan", "charcoal", "ivory",
 "sage", "mushroom", "taupe"]
```

**work_professional** (only professional neutrals):
```json
["black", "white", "grey", "gray", "navy",
 "charcoal", "ivory", "cream"]
```

---

### L. **Occasion Importance**

**Parameter**: `occasion_importance` (dict of occasion → 0–1 score)

**Effect**: Weights occasions when computing outfit impact and versatility

**Default profile**:
```json
{
  "formal": 1.0,
  "wedding": 1.0,
  "interview": 0.95,
  "business": 0.85,
  "work": 0.8,
  ...
  "beach": 0.2,
  "gym": 0.2,
  "sport": 0.2
}
```

**minimalist_eco** (flip for daily focus):
```json
{
  "daily_wear": 1.0,
  "casual": 1.0,
  "weekend": 0.8,
  ...
  "formal": 0.3,
  "wedding": 0.5
}
```

**work_professional** (corporate focus):
```json
{
  "interview": 1.0,
  "business": 1.0,
  "work": 0.95,
  "formal": 0.9,
  ...
  "date": 0.2,
  "beach": 0.0,
  "gym": 0.1
}
```

---

## Usage

### Python API

```python
from src.layer2_style.smart_removal_analyzer import SmartRemovalAnalyzer
from src.layer2_style.smart_removal_config import list_smart_removal_profiles

# List available profiles
profiles = list_smart_removal_profiles()
# Output: ['budget_conscious', 'data_driven', 'default', 'fashion_addict', 'minimalist_eco', 'work_professional']

# Create analyzer with specific profile
analyzer = SmartRemovalAnalyzer(profile_name="minimalist_eco")

# Analyze a garment
verdict = analyzer.analyze(
    garment_id="top1",
    wardrobe=my_wardrobe,
    user_goal=UserRemovalGoal.MINIMALIST
)

# Analyze entire wardrobe
all_verdicts = analyzer.analyze_wardrobe(wardrobe=my_wardrobe)
```

### REST API

```bash
# Single garment analysis
curl -X POST http://localhost:8000/api/v1/wardrobe-analysis/smart-removal \
  -H "Content-Type: application/json" \
  -d '{
    "garment_id": "top1",
    "wardrobe": [...],
    "user_goal": "minimalist"
  }'

# Full wardrobe analysis
curl -X POST http://localhost:8000/api/v1/wardrobe-analysis/smart-removal/wardrobe \
  -H "Content-Type: application/json" \
  -d '{
    "wardrobe": [...],
    "user_goal": "minimalist"
  }'
```

---

## Configuration Validation

### Validate all profiles

```python
from src.layer2_style.smart_removal_config import SmartRemovalConfigLoader

errors = SmartRemovalConfigLoader.validate_all_profiles()
for profile_name, error_list in errors.items():
    print(f"{profile_name}: {error_list}")
```

### Validation checks

- Goal weights sum to ~1.0 (tolerance: ±0.05)
- Verdict thresholds in ascending order: DONATE < SAFE < CONSIDER
- Replacement quality weights sum to ~1.0
- Versatility weights sum to ~1.0
- Confidence strictness parameters in valid ranges

---

## Creating Custom Profiles

### Method 1: Direct JSON edit

Edit `config/data/smart_removal_config.json` and add your profile:

```json
"my_startup_profile": {
  "name": "Startup Culture",
  "description": "For tech startups: casual dominant, fast moving.",
  "recency_half_life_days": 30,
  "formality_flexibility": 3,
  "replacement_coverage_threshold": 0.5,
  ...
}
```

Then load:
```python
analyzer = SmartRemovalAnalyzer(profile_name="my_startup_profile")
```

### Method 2: Programmatically create and save

```python
from src.layer2_style.smart_removal_config import (
    SmartRemovalProfile, SmartRemovalConfigLoader, GoalWeights,
    RemovalAggressiveness, ConfidenceStrictness, NewnessGuard,
    FrequencySoftCap, VersatilityWeights, ReplacementQualityWeights
)

profile = SmartRemovalProfile(
    name="My Custom Profile",
    description="Tailored for my style",
    recency_half_life_days=50,
    formality_flexibility=2,
    replacement_coverage_threshold=0.7,
    removal_aggressiveness=RemovalAggressiveness(
        donate_threshold=0.18,
        safe_to_remove_threshold=0.32,
        consider_threshold=0.62
    ),
    # ... other fields
    goal_weights={
        "MAXIMIZE_OPTIONS": GoalWeights(recency=0.12, frequency=0.12, ...),
        # ...
    },
    # ... remaining fields
)

SmartRemovalConfigLoader.save_profile("my_custom_profile", profile)
```

---

## Regional/Cultural Customization

### Example: Middle Eastern fashion focus

```json
{
  "name": "Middle Eastern",
  "neutral_colors": ["black", "navy", "beige", "cream", "gold"],
  "occasion_importance": {
    "formal": 1.0,
    "wedding": 1.0,
    "event": 0.95,
    "business": 0.85,
    "daily_wear": 0.4
  },
  "formality_flexibility": 1
}
```

### Example: California tech culture

```json
{
  "name": "California Tech",
  "neutral_colors": ["white", "grey", "black", "navy", "khaki", "earth"],
  "occasion_importance": {
    "casual": 1.0,
    "daily_wear": 1.0,
    "weekend": 0.9,
    "outdoor": 0.8,
    "business": 0.3,
    "formal": 0.1
  },
  "formality_flexibility": 3
}
```

---

## Summary Table: All Configurable Parameters

| Parameter | Type | Range | Impact |
|-----------|------|-------|--------|
| `recency_half_life_days` | float | 14–180 | How fast items lose recency attachment |
| `formality_flexibility` | int | 1–3 | How lenient formality matching is |
| `replacement_coverage_threshold` | float | 0.3–0.95 | Strictness of replacement evaluation |
| `donate_threshold` | float | 0–0.5 | When to suggest DONATE |
| `safe_to_remove_threshold` | float | 0–0.5 | When to suggest SAFE_TO_REMOVE |
| `consider_threshold` | float | 0.5–1.0 | When to suggest CONSIDER vs KEEP |
| `penalty_per_data_gap` | float | 0.05–0.30 | How much missing data hurts confidence |
| `minimum_confidence_for_action` | float | 0.2–0.9 | Confidence floor before forcing CONSIDER |
| `goal_weights` | dict | {0–1} | Signal weights per user goal |
| `grace_period_days` | int | 0–90 | Days before newness decay starts |
| `newness_half_life_days` | float | 30–180 | Newness decay rate |
| `max_recent_equivalent_wears` | int | 5–20 | Soft cap for frequency score |
| `season_weight` | float | 0–1 | Versatility: season importance |
| `occasion_weight` | float | 0–1 | Versatility: occasion importance |
| `category_weight` | float | 0–1 | Versatility: category importance |
| `neutral_colors` | array | strings | Color list for role matching |
| `occasion_importance` | dict | 0–1 | Occasion weights for impact |

---

## Next Steps

1. **Load a profile**: `analyzer = SmartRemovalAnalyzer(profile_name="minimalist_eco")`
2. **Analyze a garment**: `verdict = analyzer.analyze(garment_id, wardrobe)`
3. **Iterate**: Tweak profile parameters in JSON, reload, compare verdicts
4. **Share**: Export profiles as JSON for team/community use

---

**Version**: 1.0  
**Last updated**: 2026-03-12
