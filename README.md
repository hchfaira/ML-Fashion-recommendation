# Fashion Recommendation System

An AI-powered fashion recommendation engine built on 4 intelligent layers.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 4: LLM Interface                       │
│              (Explanation & User Experience)                    │
├─────────────────────────────────────────────────────────────────┤
│                  Layer 3: Context Engine                        │
│        (Weather, Geography, Occasion, Morphology)               │
├─────────────────────────────────────────────────────────────────┤
│              Layer 2: Style Intelligence Model                  │
│           (Compatibility Scoring - Core Moat)                   │
├─────────────────────────────────────────────────────────────────┤
│              Layer 1: Vision & Clothing Parsing                 │
│        (Segmentation, Attributes, Embeddings)                   │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
fashion_recommender/
├── src/
│   ├── layer1_vision/          # Vision & Clothing Parsing
│   ├── layer2_style/           # Style Intelligence Model
│   ├── layer3_context/         # Context Engine
│   ├── layer4_llm/             # LLM Interface
│   ├── api/                    # REST API
│   ├── core/                   # Shared utilities
│   └── main.py
├── data/
│   ├── raw/                    # Raw fashion images
│   ├── processed/              # Processed embeddings
│   └── runway/                 # Runway looks for training
├── models/
│   ├── compatibility/          # Trained compatibility models
│   └── embeddings/             # Cached embeddings
├── tests/
├── notebooks/                  # Experimentation
├── config/
└── scripts/
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Run the API
uvicorn src.main:app --reload
```

## Layer Details

### Layer 1: Vision & Clothing Parsing
- OpenAI Vision API for image analysis
- Clothing segmentation
- Attribute extraction (color, pattern, material, style)
- Embedding generation for similarity matching

### Layer 2: Style Intelligence Model (Core Moat)
- Compatibility scoring between garments
- Color harmony analysis
- Silhouette coherence
- Proportion ratios
- Formality level matching

### Layer 3: Context Engine
- Weather integration
- Geographic style preferences
- Occasion-based filtering
- Body morphology adaptation
- User history learning

### Layer 4: LLM Interface
- Outfit explanations
- Personalized recommendations
- Conversational styling advice
- Tone adaptation

## Environment Variables

```
OPENAI_API_KEY=your_openai_key
WEATHER_API_KEY=your_weather_api_key
DATABASE_URL=your_database_url
```

## License

MIT License
