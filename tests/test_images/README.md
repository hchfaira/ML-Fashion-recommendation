# Test Images Directory

This folder contains test images organized by test file.

## 📁 Directory Structure

```
tests/test_images/
├── README.md
├── segmentation/           # For test_segmentation.py
│   └── *.jpg, *.png       # Full outfit images to segment
│
├── outfit_builder/         # For test_outfit_builder.py
│   ├── outfit_1/          # One complete outfit (for single scoring)
│   │   ├── top.jpg
│   │   ├── bottom.jpg
│   │   └── shoes.jpg
│   └── wardrobe/          # Multiple items to combine (for best selection)
│       ├── tops/
│       │   ├── white_tshirt.jpg
│       │   └── blue_shirt.jpg
│       ├── bottoms/
│       │   ├── jeans.jpg
│       │   └── chinos.jpg
│       └── shoes/
│           ├── sneakers.jpg
│           └── loafers.jpg
│
└── outfit_from_images/     # For test_outfit_from_images.py
    └── *.jpg, *.png        # Individual garment images
```

## 📂 Folder Details

### `segmentation/` → `test_segmentation.py`
Images of **full outfits** (people wearing clothes) for the segmentation tests:
- `ClothingSegmenter` - Identifies individual clothing items
- `OutfitDecomposer` - Decomposes outfits into components

**What to add:**
- Full-body outfit photos
- Images with multiple visible garments

### `outfit_builder/` → `test_outfit_builder.py`
Images organized for outfit building tests:

**`outfit_builder/outfit_1/`** - Single outfit scoring
- One image per garment category (top, bottom, shoes)
- Tests scoring one complete outfit

**`outfit_builder/wardrobe/`** - Best outfit selection
- Subfolders by category: `tops/`, `bottoms/`, `shoes/`, `outerwear/`, `accessories/`
- Multiple options per category
- Tests generating combinations and selecting the best

### `outfit_from_images/` → `test_outfit_from_images.py`
Individual garment images for attribute extraction tests:
- Single garment per image (isolated)
- Good for testing the Vision API extraction

**Naming convention:**
- `top_*.jpg` - Tops (t-shirts, shirts, blouses)
- `bottom_*.jpg` - Bottoms (jeans, pants, skirts)
- `shoes_*.jpg` - Footwear
- `outerwear_*.jpg` - Jackets, coats
- `dress_*.jpg` - Dresses
- `accessory_*.jpg` - Accessories

## 🖼️ Supported Formats
- `.jpg` / `.jpeg`
- `.png`
- `.webp`

## 🧪 Running Tests

```bash
# Run segmentation tests
pytest tests/test_segmentation.py -v -s

# Run outfit builder tests (mocked + real images)
pytest tests/test_outfit_builder.py -v -s

# Run outfit from images tests
pytest tests/test_outfit_from_images.py -v -s

# Run all tests with integration marker
pytest tests/ -v -m integration
```

## ⚙️ Environment Setup

Make sure you have a `.env` file with your API key:

```bash
GOOGLE_API_KEY=your-gemini-api-key-here
```

## 📥 Adding Test Images

1. **Segmentation**: Add full outfit photos to `segmentation/`
2. **Outfit Builder**: 
   - Add a complete outfit to `outfit_builder/outfit_1/`
   - Add multiple items to `outfit_builder/wardrobe/{category}/`
3. **Individual Garments**: Add isolated garment images to `outfit_from_images/`

## 🔄 Migration from Previous Structure

If you had images directly in `test_images/`, move them to the appropriate subfolder:
- Full outfit photos → `segmentation/`
- Individual garments → `outfit_from_images/`
