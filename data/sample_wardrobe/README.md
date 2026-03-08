# Sample Wardrobe for Testing

This folder is for testing the outfit recommendation pipeline with real images.

## How to Use

### Option 1: Test with Individual Garment Images

Place one image per garment file:
```
sample_wardrobe/
├── top_white_tshirt.jpg
├── top_blue_shirt.jpg
├── pants_jeans.jpg
├── pants_chinos.jpg
├── shoes_sneakers.jpg
├── jacket_blazer.jpg
└── ...
```

Then run:
```bash
python scripts/test_full_pipeline.py --wardrobe data/sample_wardrobe/ --profile default
```

### Option 2: Test with a Single Outfit Image

If you have a photo of a complete outfit (e.g., someone wearing top + pants + shoes):
```bash
python scripts/test_full_pipeline.py --outfit-image path/to/outfit.jpg
```

### Option 3: Test Specific Images

```bash
python scripts/test_full_pipeline.py --images top.jpg pants.jpg shoes.jpg --profile business
```

## Image Requirements

- Formats: `.jpg`, `.jpeg`, `.png`, `.webp`
- Recommended: Good lighting, clear view of the garment
- For individual items: Single garment per image, neutral background preferred

## Scoring Profiles

| Profile | Description | Focus |
|---------|-------------|-------|
| `default` | Balanced scoring | All criteria equally weighted |
| `minimalist` | Clean, simple | Color harmony, 7-point rule |
| `creative` | Fashion-forward | Creativity, pattern mixing |
| `business` | Professional | Formality, design principles |
| `casual` | Everyday wear | Comfort, relaxed rules |

## Output

The pipeline will:
1. Extract garment attributes using Vision AI
2. Generate outfit combinations
3. Score each combination
4. Show the BEST outfit with detailed breakdown
