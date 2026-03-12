# Smart Removal Config - Quick Reference

## Files Location

| File | Purpose |
|------|---------|
| `config/data/smart_removal_config.json` | All profiles & parameters |
| `src/layer2_style/smart_removal_config.py` | Config loader & validators |
| `src/layer2_style/smart_removal_analyzer.py` | Analyzer using profiles |
| `docs/SMART_REMOVAL_CONFIG.md` | Full documentation |

---

## Quick Start

```python
# Load profile
analyzer = SmartRemovalAnalyzer(profile_name="minimalist_eco")

# Analyze one garment
verdict = analyzer.analyze("top1", wardrobe)
print(f"Verdict: {verdict.verdict} (risk: {verdict.regret_risk:.1%})")

# Analyze entire wardrobe
all_verdicts = analyzer.analyze_wardrobe(wardrobe)
for v in all_verdicts:
    print(f"{v.garment_id}: {v.verdict} ({v.regret_risk:.1%})")
```

---

## Profile Comparison Chart

| Profile | Philosophy | Best For |
|---------|------------|----------|
| **default** | Balanced | Most users |
| **minimalist_eco** | Small, versatile wardrobe | Minimalists, eco-conscious |
| **fashion_addict** | Trend-focused, fast turnover | Fashion enthusiasts, influencers |
| **budget_conscious** | Conservative, keep items long | Budget-aware, slow shoppers |
| **work_professional** | Corporate environment | Professionals, executives |
| **data_driven** | Analytics-focused, data demands | Researchers, data enthusiasts |

---

## Key Parameters by Use Case

### I want to remove more aggressively
```json
"removal_aggressiveness": {
  "donate_threshold": 0.25,
  "safe_to_remove_threshold": 0.45,
  "consider_threshold": 0.75
}
```

### I want to keep items longer
```json
"recency_half_life_days": 120,
"removal_aggressiveness": {
  "donate_threshold": 0.05,
  "safe_to_remove_threshold": 0.15,
  "consider_threshold": 0.50
}
```

### I'm very fashion-conscious (trends matter)
```json
"recency_half_life_days": 14,
"formality_flexibility": 3,
"goal_weights": {
  "STYLE_UPGRADE": {
    "rating": 0.40,
    "recency": 0.25
  }
}
```

### I need perfect replacements
```json
"replacement_coverage_threshold": 0.90,
"replacement_quality_weights": {
  "season_coverage": 0.50,
  "formality_compatibility": 0.35,
  "color_role_match": 0.10,
  "occasion_coverage": 0.05
}
```

### I'm minimalist
```json
"replacement_coverage_threshold": 0.85,
"goal_weights": {
  "MINIMALIST": {
    "replacement": 0.35,
    "versatility": 0.25
  }
}
```

---

## Configuration API

```python
from src.layer2_style.smart_removal_config import SmartRemovalConfigLoader

# List profiles
profiles = SmartRemovalConfigLoader.list_profiles()
# → ['budget_conscious', 'data_driven', 'default', ...]

# Get profile descriptions
descriptions = SmartRemovalConfigLoader.get_profile_descriptions()
# → {'default': 'Balanced...', 'minimalist_eco': 'For eco...', ...}

# Load a profile
profile = SmartRemovalConfigLoader.get_profile("minimalist_eco")

# Validate all profiles
errors = SmartRemovalConfigLoader.validate_all_profiles()
# → {} if valid, or {profile_name: [error_list]}

# Validate single profile
errors = SmartRemovalConfigLoader.validate_profile(profile)

# Save custom profile
SmartRemovalConfigLoader.save_profile("my_profile", custom_profile)

# Force reload from disk
profiles = SmartRemovalConfigLoader.load_profiles(force_reload=True)
```

---

## Configuration Validation Rules

✅ Goal weights must sum to ~1.0 (±0.05 tolerance)  
✅ Verdict thresholds: DONATE < SAFE < CONSIDER  
✅ Replacement quality weights must sum to ~1.0  
✅ Versatility weights must sum to ~1.0  
✅ Confidence penalties: 0–0.5 range  
✅ Recency half-life: 14–180 days recommended  
✅ Formality flexibility: 1–3 levels  
✅ Coverage threshold: 0.3–0.95 range  

---

## Signal Weights: What They Mean

| Signal | Purpose | High Weight = | Low Weight = |
|--------|---------|---------------|--------------|
| **recency** | Recent wears matter | Keep recently-worn items | Don't care about recency |
| **frequency** | Wear count matters | Keep frequently-worn items | Don't care about frequency |
| **versatility** | Context diversity matters | Keep versatile pieces | Accept specialty items |
| **outfit_impact** | Outfit loss matters | Minimize outfit disruption | OK with outfit loss |
| **replacement** | Replacement matters | Require good replacements | OK without replacement |
| **rating** | User love matters | Keep well-liked items | Ignore ratings |
| **newness** | Recent acquisitions matter | Protect new items | No grace period |

---

## Common Tweaks

### Profile is too aggressive (removing too much)
```python
# Increase thresholds
config.removal_aggressiveness.donate_threshold = 0.05
config.removal_aggressiveness.safe_to_remove_threshold = 0.15
config.removal_aggressiveness.consider_threshold = 0.50
```

### Profile is too conservative (keeping too much)
```python
# Decrease thresholds
config.removal_aggressiveness.donate_threshold = 0.30
config.removal_aggressiveness.safe_to_remove_threshold = 0.50
config.removal_aggressiveness.consider_threshold = 0.75
```

### Versatility not valued enough
```python
# Increase versatility weight in all goals
config.goal_weights["MAXIMIZE_OPTIONS"].versatility = 0.35
config.goal_weights["MINIMALIST"].versatility = 0.25
config.goal_weights["STYLE_UPGRADE"].versatility = 0.15
```

### Need very strict replacements
```python
# Increase threshold and formality weight
config.replacement_coverage_threshold = 0.90
config.replacement_quality_weights.formality_compatibility = 0.45
```

---

## Testing Across Profiles

```python
from src.core.models import UserRemovalGoal
from src.layer2_style.smart_removal_analyzer import SmartRemovalAnalyzer

garment_id = "top1"
wardrobe = [...]
goals = [
    UserRemovalGoal.MAXIMIZE_OPTIONS,
    UserRemovalGoal.MINIMALIST,
    UserRemovalGoal.STYLE_UPGRADE
]

for profile_name in ["default", "minimalist_eco", "budget_conscious"]:
    print(f"\n=== Profile: {profile_name} ===")
    analyzer = SmartRemovalAnalyzer(profile_name=profile_name)
    
    for goal in goals:
        verdict = analyzer.analyze(garment_id, wardrobe, user_goal=goal)
        print(f"{goal.value}: {verdict.verdict} (risk: {verdict.regret_risk:.1%})")
```

---

## REST API with Profiles

### Implicit: uses "default" profile
```bash
curl -X POST http://localhost:8000/api/v1/wardrobe-analysis/smart-removal \
  -H "Content-Type: application/json" \
  -d '{"garment_id": "top1", "wardrobe": [...]}'
```

### With profile parameter (when implemented)
```bash
curl -X POST http://localhost:8000/api/v1/wardrobe-analysis/smart-removal \
  -H "Content-Type: application/json" \
  -d '{
    "garment_id": "top1",
    "wardrobe": [...],
    "profile": "minimalist_eco",
    "user_goal": "minimalist"
  }'
```

---

## Customize for Region/Culture

### Make neutral colors culturally appropriate
Edit `config/data/smart_removal_config.json`:
```json
"neutral_colors": [
  "your", "regional", "neutral", "colors"
]
```

### Adjust occasion importance
```json
"occasion_importance": {
  "your_important_occasion": 1.0,
  "less_important": 0.2
}
```

### Adjust formality hierarchy
```json
"formality_flexibility": 2  // More lenient
```

---

## Summary

**3 files control smart removal**:
1. **JSON config** = all parameters (easy to edit)
2. **Config loader** = loads & validates profiles (Python API)
3. **Analyzer** = uses profiles for verdicts (no hardcoded values)

**Key benefit**: Change wardrobe removal strategy by swapping JSON profiles — no code changes needed.

---

**For full docs**: See `docs/SMART_REMOVAL_CONFIG.md`
