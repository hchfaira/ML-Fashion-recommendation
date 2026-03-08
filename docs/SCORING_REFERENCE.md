# Style Scoring System - Complete Reference

## Overview

This document describes all scoring functions in the Layer 2 Style Intelligence system. Each scorer evaluates a specific aspect of outfit styling and returns a normalized score (0-1) that can be aggregated into a final style score.

---

## Table of Contents

1. [Core Scorers](#1-core-scorers)
2. [Color Rules](#2-color-rules)
3. [Proportion & Volume Rules](#3-proportion--volume-rules)
4. [Pattern Rules](#4-pattern-rules)
5. [Context Rules](#5-context-rules)
6. [Design Principles](#6-design-principles)
7. [Aggregation Strategies](#7-aggregation-strategies)

---

## 1. Core Scorers

### 1.1 Seven-Point Rule
**File:** `seven_point_rule.py`

| Metric | Description |
|--------|-------------|
| **Concept** | Total "visual weight" of outfit should be 7-10 points |
| **Output** | `score: float (0-1)`, `total_points: int`, `is_harmonious: bool` |

**Calculation:**
```
Basic items (solid, neutral) = 1 point each
Statement items (pattern, texture, bright color) = 2 points each

total_points = sum(item_points for all items)

if 7 <= total_points <= 10:
    score = 1.0  # Harmonious
elif total_points < 7:
    score = total_points / 7  # Too minimal
else:
    score = max(0, 1 - (total_points - 10) * 0.1)  # Too busy
```

---

### 1.2 Compatibility Scorer
**File:** `compatibility_scorer.py`

| Metric | Description |
|--------|-------------|
| **Concept** | How well garments work together based on learned patterns |
| **Output** | `score: float (0-1)` |

**Calculation:**
```
For each pair of items:
    pair_score = embedding_similarity(item1, item2)
    
score = average(all_pair_scores)
```

---

## 2. Color Rules

### 2.1 Season Color Harmony
**File:** `season_color_harmony.py`

| Metric | Description |
|--------|-------------|
| **Concept** | Colors should match user's skin tone color season |
| **Input** | `user_season: ColorSeason (SPRING/SUMMER/AUTUMN/WINTER)` |
| **Output** | `score: float (0-1)`, `matches: list`, `clashes: list` |

**Calculation:**
```
Season Palettes:
- SPRING: warm, bright (coral, peach, warm green)
- SUMMER: cool, muted (lavender, soft blue, rose)
- AUTUMN: warm, muted (rust, olive, burgundy)
- WINTER: cool, bright (black, white, royal blue, red)

For each garment color:
    if color in user_season_palette: points += 2
    elif color is neutral: points += 0
    else: points -= 2

score = normalize(points, min=-2*n, max=2*n)
```

---

### 2.2 Skin Contrast Rule
**File:** `skin_contrast_scorer.py`

| Metric | Description |
|--------|-------------|
| **Concept** | Outfit contrast should match natural skin/hair contrast |
| **Input** | `user_contrast: ContrastType (HIGH/MEDIUM/LOW)` |
| **Output** | `score: float (0-1)`, `is_flattering: bool` |

**Calculation:**
```
Analyze outfit:
- is_monochrome = all colors same family or neutrals
- has_complementary = contains complementary color pair
- has_high_saturation = contains vibrant colors

Scoring:
if user_contrast == HIGH:
    if has_complementary: +3 pts
    if has_high_saturation: +2 pts
    
if user_contrast == LOW:
    if is_monochrome: +3 pts
    if is_tonal: +2 pts
    
if user_contrast == MEDIUM:
    +2 pts for either approach

score = points / 5
```

---

### 2.3 Three Color Rule
**File:** `three_color_scorer.py`

| Metric | Description |
|--------|-------------|
| **Concept** | Outfit should not exceed 3 distinct colors (excluding neutrals) |
| **Input** | `user_aesthetic: StyleAesthetic (optional)` |
| **Output** | `score: float (0-1)`, `unique_colors: int`, `within_limit: bool` |

**Calculation:**
```
Neutrals (don't count): black, white, gray, beige, cream, nude

unique_colors = count(non_neutral_color_families)

Standard scoring:
| Colors | Points |
|--------|--------|
| 1-2    | 5/5    |
| 3      | 5/5    |
| 4      | 3/5    |
| 5+     | 1/5    |

Maximalist exception:
| Colors | Points |
|--------|--------|
| 4      | 4/5    |
| 5      | 3/5    |
| 6+     | 2/5    |

score = points / 5
```

---

### 2.4 Sandwich Rule (Color)
**File:** `sandwich_rule_scorer.py`

| Metric | Description |
|--------|-------------|
| **Concept** | Top and shoes should match, bottom is different |
| **Output** | `score: float (0-1)`, `has_sandwich: bool` |

**Calculation:**
```
top_color = top.primary_color
bottom_color = bottom.primary_color
shoes_color = shoes.primary_color

points = 0
if colors_match(top_color, shoes_color):
    points += 2
    
if not colors_match(bottom_color, top_color) and 
   not colors_match(bottom_color, shoes_color):
    points += 1

score = points / 3
has_sandwich = points >= 2
```

---

## 3. Proportion & Volume Rules

### 3.1 Proportion Scorer (Rule of Thirds)
**File:** `proportion_scorer.py`

| Metric | Description |
|--------|-------------|
| **Concept** | Visual split should follow 1/3-2/3 ratio |
| **Output** | `score: float (0-1)`, `ratio: ProportionRatio` |

**Calculation:**
```
Ratio Analysis:
| Top Length  | Bottom Rise | Ratio      | Score |
|-------------|-------------|------------|-------|
| CROP        | HIGH_RISE   | 1:2        | 1.0   |
| REGULAR     | HIGH_RISE   | 1:1.5      | 0.85  |
| REGULAR     | MID_RISE    | 1:1 (50/50)| 0.5   |
| LONGLINE    | LOW_RISE    | inverted   | 0.3   |

Bonus: if top is tucked: +0.15
```

---

### 3.2 Volume Balance Scorer
**File:** `volume_balance_scorer.py`

| Metric | Description |
|--------|-------------|
| **Concept** | Fitted + Relaxed creates balance |
| **Input** | `body_shape: BodyShape (optional)` |
| **Output** | `score: float (0-1)`, `balance_type: str` |

**Calculation:**
```
Volume Classification:
- SLIM: fitted, slim, skinny, tailored
- MEDIUM: regular, straight
- WIDE: oversized, relaxed, wide-leg, voluminous

Scoring:
| Top Volume | Bottom Volume | Points | Balance Type    |
|------------|---------------|--------|-----------------|
| WIDE       | SLIM          | +2     | balanced        |
| SLIM       | WIDE          | +2     | balanced        |
| SLIM       | SLIM          | 0      | both_fitted     |
| WIDE       | WIDE          | -1     | both_voluminous |
| MEDIUM     | any           | +1     | moderate        |

Body Shape Adjustments:
- PEAR + wider bottom: bonus +0.5
- INVERTED_TRIANGLE + wider top: bonus +0.5

score = normalize(points, -1, 3)
```

---

## 4. Pattern Rules

### 4.1 Pattern Mixing Scorer
**File:** `pattern_mixing_scorer.py`

| Metric | Description |
|--------|-------------|
| **Concept** | 5 sub-rules for mixing patterns correctly |
| **Output** | `overall_score: float (0-1)`, individual rule scores |

**Sub-rules and Weights:**

| Rule | Weight | Max Points | Description |
|------|--------|------------|-------------|
| 60/40 Balance | 25% | 5 | Ratio of patterns to solids |
| Scale Contrast | 25% | 5 | Different pattern sizes |
| Pattern Family | 20% | 5 | Compatible pattern types |
| Pattern Sandwich | 15% | 5 | Solids separate patterns |
| Texture Harmony | 15% | 5 | Avoid clashing textures |

**Calculations:**

#### 4.1.1 Balance Rule (60/40)
```
pattern_ratio = patterned_items / total_items

| Condition | Points |
|-----------|--------|
| 1 pattern + solids + color callback | 5 |
| 1 pattern + solids | 4 |
| 2 patterns + solid separator | 3-4 |
| All solids | 3 |
| >70% patterns | 1 |

Color Callback Bonus:
if any(solid.color in pattern.colors): +1
```

#### 4.1.2 Scale Contrast Rule
```
Scale Values: MICRO=1, SMALL=2, MEDIUM=3, LARGE=4, OVERSIZED=5

scale_difference = max(scales) - min(scales)

| Difference | Points |
|------------|--------|
| >= 3       | 5      |
| 2          | 4      |
| 1          | 2      |
| 0 (same)   | 1      |
```

#### 4.1.3 Pattern Family Compatibility
```
Compatible Pairs:
- STRIPES + {DOTS, FLORAL, ANIMAL, CHECKS}
- CHECKS + {STRIPES, DOTS, FLORAL}
- DOTS + {STRIPES, FLORAL, CHECKS}

Neutral Patterns (go with everything):
- Breton stripes, gingham, small checks, pinstripes

score = compatible_pairs / total_pairs
```

#### 4.1.4 Pattern Sandwich Rule
```
If patterns are separated by solid items: 5 pts
If 2 adjacent patterns: 3 pts
If 3+ adjacent patterns: 1 pt
```

#### 4.1.5 Texture Harmony
```
Good combinations: SMOOTH + TEXTURED (0.9-1.0)
Bad combinations: TEXTURED + TEXTURED (0.3-0.4)

score = average(texture_compatibility for all pattern pairs)
```

**Final Pattern Score:**
```
overall = (balance * 0.25) + (scale * 0.25) + (family * 0.20) + 
          (sandwich * 0.15) + (texture * 0.15)
score = overall / 5
```

---

## 5. Context Rules

### 5.1 Occasion Scorer
**File:** `occasion_scorer.py`

| Metric | Description |
|--------|-------------|
| **Concept** | Outfit formality must match event type |
| **Input** | `occasion: Occasion`, `context: UserContext` |
| **Output** | `score: float (0-1)`, `is_appropriate: bool`, `mismatch_level: int` |

**Formality Order:**
```
VERY_CASUAL < CASUAL < SMART_CASUAL < BUSINESS_CASUAL < BUSINESS < FORMAL < BLACK_TIE
     0          1          2              3               4          5        6
```

**Occasion to Expected Formality:**
```
| Occasion  | Primary     | Acceptable                      |
|-----------|-------------|--------------------------------|
| CASUAL    | CASUAL      | VERY_CASUAL, SMART_CASUAL      |
| BUSINESS  | BUSINESS    | BUSINESS_CASUAL, SMART_CASUAL  |
| FORMAL    | FORMAL      | BUSINESS, BLACK_TIE            |
| SPORT     | VERY_CASUAL | CASUAL                         |
| EVENING   | FORMAL      | SMART_CASUAL, BLACK_TIE        |
| BEACH     | VERY_CASUAL | CASUAL                         |
| DATE      | SMART_CASUAL| CASUAL, BUSINESS_CASUAL        |
```

**Calculation:**
```
outfit_formality = weighted_average(item_formality_levels)
mismatch_level = abs(outfit_index - expected_index)

| Mismatch | Points | Score |
|----------|--------|-------|
| 0        | 5      | 1.0   |
| 1        | 3      | 0.6   |
| 2        | 1      | 0.2   |
| 3+       | 0      | 0.0   |
```

---

## 6. Design Principles

### 6.1 Design Principles Scorer
**File:** `design_principles_scorer.py`

| Metric | Description |
|--------|-------------|
| **Concept** | 11 fundamental fashion design principles |
| **Output** | `overall_score: float (0-1)`, individual analyses |

**Principles and Weights:**

| Principle | Weight | Description |
|-----------|--------|-------------|
| Proportion | 12% | Golden ratio (1:1.618) |
| Balance | 10% | Symmetric vs Asymmetric |
| Rhythm | 8% | Visual movement |
| Emphasis | 12% | Single focal point |
| Contrast | 10% | Strong differences |
| Harmony | 15% | Elements work together |
| Unity | 10% | Cohesive whole |
| Scale | 8% | Size relationships |
| Repetition | 5% | Repeated elements |
| Gradation | 5% | Progressive variation |
| Line Direction | 5% | Vertical/Horizontal/Diagonal |

**Key Calculations:**

#### 6.1.1 Proportion
```
| Top Length | Bottom Rise | Follows Golden Ratio | Score |
|------------|-------------|----------------------|-------|
| CROP       | HIGH_RISE   | Yes (1:2)            | 0.95  |
| REGULAR    | HIGH_RISE   | Yes (1:1.5)          | 0.85  |
| REGULAR    | MID_RISE    | No (50:50)           | 0.70  |
| LONGLINE   | LOW_RISE    | No (inverted)        | 0.40  |

Tucked-in bonus: +0.10
```

#### 6.1.2 Emphasis (Focal Point)
```
Focal Point Types:
- COLOR: bright/contrasting color
- TEXTURE: velvet, sequins, metallic
- DETAIL: embellishments, statement buttons
- NECKLINE: halter, off-shoulder, cowl
- SILHOUETTE: dramatic, sculptural

| # Focal Points | Score |
|----------------|-------|
| 0              | 0.5   |
| 1              | 1.0   |
| 2              | 0.7   |
| 3+             | 0.4   |
```

#### 6.1.3 Line Direction
```
Line Types:
- VERTICAL: v_neck, long cardigan, maxi, vertical stripes → elongating
- HORIZONTAL: boat_neck, crop top, horizontal stripes → widening
- DIAGONAL: wrap dress, asymmetric hem → dynamic

| Dominant Line | Base Score | Effect |
|---------------|------------|--------|
| VERTICAL      | 0.90       | Slimming |
| DIAGONAL      | 0.85       | Modern |
| HORIZONTAL    | 0.60       | Widening |
| MIXED         | 0.70       | Neutral |

Body Type Adjustments:
- PEAR + horizontal: +0.2 (balances proportions)
```

#### 6.1.4 Harmony
```
Check:
1. Color harmony (complementary, analogous, monochrome)
2. Formality consistency (max 2 levels apart)
3. Style consistency (no mixing casual + formal tags)

harmony_score = harmonious_elements / (harmonious + conflicting)
```

**Final Design Principles Score:**
```
overall = (proportion * 0.12) + (balance * 0.10) + (rhythm * 0.08) + 
          (emphasis * 0.12) + (contrast * 0.10) + (harmony * 0.15) + 
          (unity * 0.10) + (scale * 0.08) + (repetition * 0.05) + 
          (gradation * 0.05) + (line_direction * 0.05)
```

---

## 7. Aggregation Strategies

### 7.1 Current Default Weights

```python
STYLE_WEIGHTS = {
    # Core rules
    "seven_point_rule": 0.15,
    "compatibility": 0.10,
    
    # Color rules
    "color_harmony": 0.12,
    "three_color_rule": 0.08,
    "sandwich_rule": 0.05,
    
    # Proportion rules
    "proportion": 0.10,
    "volume_balance": 0.08,
    
    # Pattern rules
    "pattern_mixing": 0.10,
    
    # Context rules
    "occasion": 0.12,
    
    # Design principles
    "design_principles": 0.10,
}
```

### 7.2 Alternative Aggregation Approaches

#### Approach A: Weighted Average (Current)
```python
final_score = sum(score * weight for score, weight in scores.items())
```
**Pros:** Simple, predictable
**Cons:** A low score in one area can be masked by high scores elsewhere

#### Approach B: Minimum Gate
```python
# Must pass minimum threshold on critical rules
critical_rules = ["occasion", "harmony"]
if any(scores[rule] < 0.4 for rule in critical_rules):
    final_score = max_possible_score * 0.5  # Capped

final_score = weighted_average(scores)
```
**Pros:** Prevents style disasters (wrong occasion = fail)
**Cons:** Harsh for beginners

#### Approach C: Tiered Scoring
```python
# Tier 1: Must-pass (weight = gate)
tier1 = ["occasion", "formality_match"]  # If fail, max score = C

# Tier 2: Core aesthetics (60% of remaining)
tier2 = ["seven_point", "proportion", "color_harmony", "emphasis"]

# Tier 3: Refinement (40% of remaining)
tier3 = ["pattern_mixing", "sandwich_rule", "line_direction"]

if any(scores[r] < 0.3 for r in tier1):
    final_score = 0.5 * average(tier2 + tier3)
else:
    final_score = 0.4 * average(tier1) + 0.35 * average(tier2) + 0.25 * average(tier3)
```

#### Approach D: Geometric Mean
```python
# Geometric mean penalizes low scores more than arithmetic mean
import math
final_score = math.exp(sum(math.log(s) * w for s, w in weighted_scores) / sum(weights))
```
**Pros:** Low scores hurt more (encourages balance)
**Cons:** Zero scores break the calculation

#### Approach E: Category-Based
```python
categories = {
    "color": ["color_harmony", "three_color", "sandwich", "skin_contrast"],
    "silhouette": ["proportion", "volume_balance", "line_direction"],
    "pattern": ["pattern_mixing", "seven_point"],
    "context": ["occasion", "formality"],
    "design": ["emphasis", "harmony", "unity", "contrast"]
}

category_scores = {cat: average(scores[r] for r in rules) for cat, rules in categories.items()}

final_score = weighted_average(category_scores, weights=[0.25, 0.25, 0.15, 0.15, 0.20])
```

### 7.3 Personalization Weights

User preferences can adjust weights:

```python
# Style-based adjustments
if user.style_aesthetic == "MINIMALIST":
    weights["three_color_rule"] *= 1.5
    weights["pattern_mixing"] *= 0.5
    
elif user.style_aesthetic == "MAXIMALIST":
    weights["three_color_rule"] *= 0.5
    weights["pattern_mixing"] *= 1.5
    weights["seven_point_rule"] *= 0.7  # Allow more points

# Context-based adjustments
if user.occasion == "BUSINESS":
    weights["occasion"] *= 2.0
    weights["formality"] *= 2.0
    
elif user.occasion == "CASUAL":
    weights["occasion"] *= 0.5
```

### 7.4 Scoring Summary Table

| Scorer | Range | Critical? | Personalized? |
|--------|-------|-----------|---------------|
| Seven-Point Rule | 0-1 | No | No |
| Season Color Harmony | 0-1 | No | Yes (color_season) |
| Skin Contrast | 0-1 | No | Yes (contrast_type) |
| Three Color Rule | 0-1 | No | Yes (aesthetic) |
| Sandwich Rule | 0-1 | No | No |
| Proportion | 0-1 | No | No |
| Volume Balance | 0-1 | No | Yes (body_shape) |
| Pattern Mixing | 0-1 | No | No |
| Occasion | 0-1 | **Yes** | Yes (occasion) |
| Design Principles | 0-1 | No | Yes (body_type) |

---

## 8. Quick Reference: All Scores

```python
from src.layer2_style import (
    # Quick functions
    calculate_seven_point_score,
    score_skin_contrast,
    get_sandwich_score,
    get_three_color_score,
    get_occasion_score,
    get_pattern_mixing_score,
    get_design_principles_score,
    
    # Full scorers
    SevenPointRuleScorer,
    SeasonColorHarmonyScorer,
    SkinContrastScorer,
    ThreeColorScorer,
    SandwichRuleScorer,
    ProportionScorer,
    VolumeBalanceScorer,
    PatternMixingScorer,
    OccasionScorer,
    DesignPrinciplesScorer,
    TotalStyleScorer,
)

# Example: Get all scores for aggregation
def get_all_scores(items, user_context):
    return {
        "seven_point": calculate_seven_point_score(items),
        "skin_contrast": score_skin_contrast(items, user_context.contrast_type),
        "sandwich": get_sandwich_score(items),
        "three_color": get_three_color_score(items),
        "occasion": get_occasion_score(items, user_context.occasion),
        "pattern": get_pattern_mixing_score(items),
        "design": get_design_principles_score(items),
        # ... add others
    }
```
