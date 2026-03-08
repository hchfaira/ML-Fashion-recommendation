# Simple Vision Analysis Prompt

Analyze this clothing item and extract the following attributes in JSON format:

```json
{
    "category": "top|bottom|dress|outerwear|shoes|accessory|bag",
    "subcategory": "specific type (e.g., 't-shirt', 'blazer', 'jeans')",
    "color": {
        "primary": "main color name",
        "secondary": "secondary color if any",
        "accent": "accent color if any",
        "hex_codes": ["#XXXXXX"]
    },
    "material": {
        "primary": "main material (cotton, silk, leather, etc.)",
        "secondary": "secondary material if any",
        "texture": "texture description"
    },
    "pattern": {
        "type": "solid|stripes|plaid|floral|geometric|animal|abstract|other",
        "scale": "small|medium|large",
        "description": "brief pattern description"
    },
    "style_tags": ["list", "of", "style", "descriptors"],
    "formality_level": "very_casual|casual|smart_casual|business_casual|business|formal|black_tie",
    "season_suitable": ["spring", "summer", "fall", "winter"],
    "silhouette": "fitted|relaxed|oversized|structured|flowing",
    "fit": "slim|regular|relaxed|oversized",
    "description": "Brief natural language description of the item"
}
```

Be precise and accurate in your analysis.
