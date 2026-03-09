# Fashion Recommendation System

An AI-powered fashion recommendation engine built on a 7-layer intelligent architecture, combining computer vision, style intelligence, contextual awareness, and personalized user profiling.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 6: Virtual Try-On                      │
│           (CatVTON, Replicate AI clothing overlay)              │
├─────────────────────────────────────────────────────────────────┤
│                   Layer 5: Visualization                        │
│              (Outfit rendering & composition)                   │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 4: LLM Interface                       │
│       (Explanations, Conversations, Styling Advice)             │
├─────────────────────────────────────────────────────────────────┤
│                  Layer 3: Context Engine                        │
│   (Weather, Occasion, User Profile, Body Measurements)          │
├─────────────────────────────────────────────────────────────────┤
│              Layer 2: Style Intelligence Model                  │
│      (11+ Scoring Criteria - Compatibility Analysis)            │
├─────────────────────────────────────────────────────────────────┤
│              Layer 1: Vision & Attribute Extraction             │
│        (Gemini Vision AI, Color/Pattern Analysis)               │
├─────────────────────────────────────────────────────────────────┤
│              Layer 0: Garment Segmentation                      │
│         (SAM, GroundingDINO, SCHP Human Parsing)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
fashion_recommender/
├── src/
│   ├── layer0_segmentation/    # Garment extraction from images
│   ├── layer1_vision/          # Vision AI & Attribute extraction
│   ├── layer2_style/           # Style Intelligence & Scoring
│   ├── layer3_context/         # Context Engine & User Profiling
│   ├── layer4_llm/             # LLM Explanations & Conversations
│   ├── layer5_visualization/   # Outfit image generation
│   ├── layer6_tryon/           # Virtual try-on
│   ├── api/                    # REST API endpoints
│   ├── core/                   # Shared utilities & models
│   └── main.py
├── config/
│   ├── scoring_config.json     # Scoring profiles & weights
│   ├── data/                   # JSON rules (colors, weather, etc.)
│   ├── parameters/             # Model & conversation settings
│   └── prompts/                # LLM prompt templates
├── data/
│   ├── raw/                    # Raw fashion images
│   ├── processed/              # Processed embeddings
│   ├── sample_wardrobe/        # Sample wardrobe for testing
│   └── runway/                 # Runway looks for training
├── models/
│   ├── compatibility/          # Trained compatibility models
│   └── embeddings/             # Cached embeddings
├── scripts/
│   ├── test_full_pipeline.py   # End-to-end pipeline testing
│   └── pipeline/               # Pipeline modules
├── tests/
│   ├── unit/                   # Unit tests (670+ tests)
│   ├── integration/            # Integration tests
│   └── e2e/                    # End-to-end tests
└── notebooks/                  # Experimentation
```

---

## 🎯 Layer Details

### Layer 0: Garment Segmentation

Extracts individual garments from outfit images using state-of-the-art segmentation models.

| Module | Description |
|--------|-------------|
| `garment_extractor.py` | Main extraction pipeline |
| `groundingdino_detector.py` | GroundingDINO for garment detection |
| `sam_refiner.py` | SAM (Segment Anything Model) for precise masks |
| `schp_parser.py` | SCHP human parsing for body segmentation |
| `mask_fusion.py` | Combines multiple segmentation masks |
| `mask_postprocess.py` | Mask cleanup and refinement |
| `taxonomy.py` | Garment category taxonomy |

**Features:**
- Multi-model ensemble (GroundingDINO + SAM + SCHP)
- Precise garment boundary detection
- Category-aware segmentation
- Mask refinement and post-processing

---

### Layer 1: Vision & Attribute Extraction

Analyzes garment images to extract detailed attributes using Google Gemini Vision AI.

| Module | Description |
|--------|-------------|
| `attribute_extractor.py` | Main attribute extraction pipeline |
| `vision_service.py` | Gemini Vision API integration |
| `color_utils.py` | Color analysis & fashion color mapping |
| `embedding_generator.py` | CLIP/ResNet embeddings for similarity |
| `segmentation.py` | Integration with Layer 0 |

**Extracted Attributes:**
```python
GarmentAttributes:
  ├── category          # top, bottom, shoes, outerwear, accessories
  ├── subcategory       # t-shirt, jeans, sneakers, etc.
  ├── color             # ColorProfile (primary, secondary, hex codes)
  ├── pattern           # PatternInfo (type, scale, density)
  ├── material          # fabric type
  ├── fit               # slim, regular, loose, oversized
  ├── formality_level   # casual, smart_casual, formal
  ├── season            # spring, summer, fall, winter
  └── style_tags        # minimalist, bohemian, streetwear, etc.
```

---

### Layer 2: Style Intelligence Model ⭐ (Core Scoring Engine)

The heart of the system - 11+ scoring criteria that evaluate outfit compatibility using fashion design principles.

#### 📊 Scoring Criteria

| Criterion | Weight* | Description |
|-----------|---------|-------------|
| **7-Point Rule** | 15% | Balance between basics (1-2 pts) and statement pieces (3+ pts) |
| **Color Harmony** | 15% | Color wheel relationships (complementary, analogous, triadic) |
| **3-Color Rule** | 10% | Outfit should use ≤3 main colors + neutrals |
| **Proportions** | 15% | Rule of thirds, golden ratio in outfit composition |
| **Volume Balance** | 10% | Top/bottom volume relationship (fitted-loose, loose-fitted) |
| **Pattern Mixing** | 10% | Pattern scale variation, density balance |
| **Design Principles** | 15% | Balance, harmony, rhythm, emphasis, contrast |
| **Creativity** | 10% | Intentional rule-breaking, fashion-forward elements |

*Weights shown for "default" profile - varies by scoring profile

#### 🎨 Scoring Profiles

| Profile | Focus | Use Case |
|---------|-------|----------|
| `default` | Balanced scoring | Everyday outfits |
| `minimalist` | Color harmony, 3-color rule | Clean, simple looks |
| `creative` | Creativity, pattern mixing | Fashion-forward styles |
| `business` | Formality, proportions | Professional settings |
| `casual` | Comfort, creativity | Weekend casual |

#### Key Modules

| Module | Description |
|--------|-------------|
| `outfit_scorecard.py` | Main scoring orchestrator |
| `outfit_builder.py` | Outfit candidate generation |
| `outfit_search.py` | Optimized outfit search algorithms |
| `seven_point_rule.py` | Point value calculation (1-7 scale) |
| `color_harmony.py` | Color wheel analysis |
| `season_color_harmony.py` | Seasonal color analysis (Spring/Summer/Autumn/Winter) |
| `three_color_scorer.py` | Color count analysis |
| `proportion_scorer.py` | Rule of thirds, golden ratio |
| `volume_balance_scorer.py` | Top/bottom volume matching |
| `pattern_mixing_scorer.py` | Pattern compatibility analysis |
| `design_principles_scorer.py` | Fashion design fundamentals |
| `creativity_scorer.py` | Rule-breaking intelligence |
| `silhouette_analyzer.py` | Overall outfit silhouette |
| `formality_matcher.py` | Formality level consistency |
| `skin_contrast_scorer.py` | Skin tone contrast matching |
| `sandwich_rule_scorer.py` | Color sandwiching technique |
| `total_style_scorer.py` | Aggregate style scoring |

#### Scoring Output Example

```
🎯 Overall Score: 78.5%
📝 Grade: B+

📊 Criteria evaluated (9):
   7-Point Rule              [████████░░] 80.0%
   Color Harmony             [█████████░] 90.0%
   3-Color Rule              [██████████] 100.0%
   Proportions               [███████░░░] 70.0%
   Volume Balance            [████████░░] 83.3%
   Pattern Mixing            [█████████░] 90.0%
   Design Principles         [███████░░░] 73.0%
   Total Style               [███████░░░] 65.0%
   Creativity                [████░░░░░░] 45.0%
```

---

### Layer 3: Context Engine

Adapts outfit recommendations based on user context, environment, and personal characteristics.

#### 🌍 Context Criteria

| Criterion | Description |
|-----------|-------------|
| **WEATHER** | Temperature, precipitation, humidity-based filtering |
| **OCCASION** | Event type (work, date, casual, formal) |
| **ACTIVITY** | Physical activity level requirements |
| **SCHEDULE** | Multi-event day outfit transitions |
| **ROTATION** | Wardrobe freshness, avoid recent repeats |
| **MORPHOLOGY** | Body type recommendations |
| **FIT** | Size compatibility scoring |
| **PROPORTION** | Body proportion harmony |
| **COLOR_HARMONY** | Personal coloring (skin/hair/contrast) |

#### 👤 User Profile Pipeline

Extracts comprehensive user style profile from a photo:

```python
StyleProfile:
  ├── body_metrics           # BMI, frame size, proportions
  │   ├── height_cm
  │   ├── weight_kg
  │   ├── bmi / bmi_category
  │   ├── frame_size         # small, medium, large
  │   ├── torso_leg_ratio
  │   ├── shoulder_hip_ratio
  │   ├── estimated_top_size
  │   └── estimated_bottom_size
  │
  ├── skin_analysis
  │   ├── skin_tone          # fair, light, medium, tan, deep
  │   └── undertone          # warm, cool, neutral
  │
  ├── hair_analysis
  │   ├── hair_color         # blonde, brown, black, red, gray
  │   └── hair_tone          # warm, cool, neutral
  │
  ├── contrast_level         # low, medium, high, very_high
  └── color_season           # SPRING, SUMMER, AUTUMN, WINTER
```

#### Key Modules

| Module | Description |
|--------|-------------|
| `context_engine.py` | Main context orchestrator |
| `weather_service.py` | Weather API integration |
| `occasion_analyzer.py` | Event type detection |
| `activity_analyzer.py` | Activity-based comfort scoring |
| `schedule_analyzer.py` | Multi-event day planning |
| `wardrobe_rotation.py` | Outfit freshness tracking |
| `morphology_advisor.py` | Body type recommendations |
| `fit_predictor.py` | Size compatibility scoring |
| `proportion_harmonizer.py` | Body proportion balancing |
| `color_harmony_advisor.py` | Personal color analysis |
| `user_profile/` | User profile extraction pipeline |

#### Context-Aware Scoring Output

```
🎯 Applying Context-Aware Scoring
============================================================
   👕 Fit Score: 0.85 (good)
   📐 Proportion Score: 0.78 (good)
   🎨 Color Harmony Score: 0.92 (excellent)

   ⭐ Overall Context Score: 0.85
   🔄 Blended Score (60% style + 40% context): 0.81
```

---

### Layer 4: LLM Interface

Natural language explanations and conversational styling advice using LLM models.

| Module | Description |
|--------|-------------|
| `llm_service.py` | LLM API integration (OpenAI/Gemini) |
| `outfit_explainer.py` | Outfit recommendation explanations |
| `conversation_handler.py` | Multi-turn styling conversations |

**Features:**
- Natural language outfit explanations
- Personalized styling tips
- Conversational Q&A about fashion
- Tone adaptation based on user preferences

**Prompt Templates:**
- `llm_explain_outfit_system.md` - System prompt for explanations
- `llm_outfit_comparison.md` - Compare two outfits
- `llm_styling_tip.md` - Generate styling tips
- `llm_personalize_message.md` - Personalize recommendations
- `conversation_system.md` - Conversational system prompt

---

### Layer 5: Visualization

Generates visual representations of outfit recommendations.

| Module | Description |
|--------|-------------|
| `OutfitVisualizer` | Main visualization class |
| `create_outfit_image()` | Compose outfit collage |

**Features:**
- Outfit collage generation
- Score overlay visualization
- Before/after comparisons

---

### Layer 6: Virtual Try-On

AI-powered virtual try-on using state-of-the-art models.

| Module | Description |
|--------|-------------|
| `tryon_service.py` | Main try-on orchestrator |
| `catvton_backend.py` | CatVTON model integration |
| `replicate_backend.py` | Replicate API integration |
| `mask_generator.py` | Body region masking |

**Features:**
- Realistic clothing overlay on user photos
- Multiple backend support (CatVTON, Replicate)
- Body-aware garment placement

---

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your API keys (GOOGLE_API_KEY, etc.)

# 4. Run the full pipeline test
python scripts/test_full_pipeline.py --demo

# 5. Test with a wardrobe folder
python scripts/test_full_pipeline.py --wardrobe ./data/sample_wardrobe/

# 6. Test with user profile (context-aware scoring)
python scripts/test_full_pipeline.py \
  --wardrobe ./data/sample_wardrobe/ \
  --user-photo ./my_photo.jpg \
  --height 175 \
  --weight 70

# 7. Run the API
uvicorn src.main:app --reload
```

---

## 🧪 Testing

```bash
# Run all unit tests (670+ tests)
pytest tests/unit/ -v

# Run specific layer tests
pytest tests/unit/layer2/ -v  # Style scoring tests
pytest tests/unit/layer3/ -v  # Context engine tests

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

---

## ⚙️ Configuration

### Scoring Configuration (`config/scoring_config.json`)

```json
{
  "scoring_profiles": {
    "default": {
      "criteria": {
        "seven_point": { "enabled": true, "weight": 0.15 },
        "color_harmony": { "enabled": true, "weight": 0.15 },
        "three_color": { "enabled": true, "weight": 0.10 },
        "proportion": { "enabled": true, "weight": 0.15 },
        "volume_balance": { "enabled": true, "weight": 0.10 },
        "pattern_mixing": { "enabled": true, "weight": 0.10 },
        "design_principles": { "enabled": true, "weight": 0.15 },
        "creativity": { "enabled": true, "weight": 0.10 }
      }
    }
  }
}
```

### Data Configuration (`config/data/`)

| File | Description |
|------|-------------|
| `color_data.json` | Color harmony rules, color wheel relationships |
| `weather_data.json` | Weather-to-clothing mappings |
| `occasion_data.json` | Occasion formality requirements |
| `morphology_data.json` | Body type recommendations |
| `activity_data.json` | Activity-based requirements |
| `compatibility_data.json` | Garment compatibility rules |
| `body_profile_config.json` | User profile scoring weights |

---

## 🔑 Environment Variables

```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key

# Optional
OPENAI_API_KEY=your_openai_key
WEATHER_API_KEY=your_weather_api_key
REPLICATE_API_TOKEN=your_replicate_token
DATABASE_URL=your_database_url
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analysis/garment` | POST | Analyze single garment |
| `/api/analysis/outfit` | POST | Analyze complete outfit |
| `/api/recommendation/suggest` | POST | Get outfit suggestions |
| `/api/wardrobe/upload` | POST | Upload wardrobe items |
| `/api/context/update` | POST | Update user context |
| `/api/scoring/evaluate` | POST | Score an outfit |
| `/api/chat/message` | POST | Conversational styling |

---

## 📈 Scoring Algorithm Deep Dive

### 7-Point Rule

Based on the fashion principle that a well-balanced outfit has a point value between 4-7:

| Item Type | Points |
|-----------|--------|
| Basic solid | 1 |
| Neutral with texture | 1.5 |
| Colored basic | 2 |
| Statement piece | 3 |
| Bold pattern/accessory | 4 |

**Scoring:**
- 4-7 points → High score (optimal range)
- <4 points → Too basic, needs visual interest
- >7 points → Too busy, competing elements

### Color Harmony

Analyzes outfit colors using color wheel relationships:

| Relationship | Description | Score Bonus |
|--------------|-------------|-------------|
| Complementary | Opposite colors (high contrast) | +20% |
| Analogous | Adjacent colors (harmonious) | +15% |
| Triadic | Three equidistant colors | +18% |
| Monochromatic | Same color, different shades | +10% |
| Neutral + Accent | Neutrals with one pop color | +12% |

### Design Principles

Based on fashion school fundamentals:

1. **Balance** - Visual weight distribution (symmetric/asymmetric)
2. **Proportion** - Size relationships (golden ratio, rule of thirds)
3. **Rhythm** - Repetition and flow of design elements
4. **Emphasis** - Focal point creation
5. **Harmony** - Unity of all elements
6. **Contrast** - Intentional differences for interest

---

## 📄 License

MIT License

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/unit/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request
