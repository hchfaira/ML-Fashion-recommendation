# Attribute Extraction Prompt

Step-by-step reasoning:

1. Identify garment category and outfit role.
2. Identify colors using STANDARD CSS/HTML color names (see list below).
3. Identify silhouette and volume.
4. Identify material and seasonal weight.
5. Identify style aesthetics.
6. Extract styling compatibility attributes.

## IMPORTANT: Color Naming Convention

For `primary_color` and `secondary_color`, use ONLY these standard color names:

**Neutrals:** white, ivory, beige, cream, tan, khaki, brown, chocolate, black, gray, silver, charcoal
**Reds:** red, crimson, maroon, burgundy, coral, salmon, tomato, firebrick
**Pinks:** pink, hotpink, deeppink, lightpink, rose, blush, fuchsia, magenta
**Oranges:** orange, darkorange, peach, apricot, rust, terracotta, copper
**Yellows:** yellow, gold, goldenrod, mustard, lemon, amber, honey
**Greens:** green, olive, sage, mint, emerald, forestgreen, lime, teal, seafoam
**Blues:** blue, navy, royalblue, skyblue, lightblue, turquoise, aqua, cobalt, denim, indigo, steelblue
**Purples:** purple, violet, lavender, plum, orchid, lilac, mauve, grape, eggplant
**Browns:** brown, tan, camel, cognac, chocolate, coffee, mocha, sienna, taupe

Always include hex code in `hex_codes` as backup (e.g., "#4A90D9").

Return the final JSON with this schema:

```json
{
  "taxonomy": {
    "category": "top|bottom|dress|outerwear|shoes|bag|accessory",
    "subcategory": "specific item type (e.g., t-shirt, blazer, jeans)",
    "product_type": "detailed product name",
    "outfit_role": "base_layer|main_piece|layering_piece|statement_piece|supporting_piece|footwear|accessory"
  },

  "color_profile": {
    "primary_color": "REQUIRED: use standard color name from list above (e.g., navy, lavender, beige)",
    "secondary_color": "standard color name or null",
    "color_palette": ["all colors using standard names"],
    "hex_codes": ["#XXXXXX - REQUIRED: always include at least one hex code"],
    "color_temperature": "warm|cool|neutral",
    "color_depth": "light|medium|dark",
    "contrast_level": "low|medium|high",
    "pattern": "solid|stripe|plaid|check|floral|graphic|animal|abstract"
  },

  "material_profile": {
    "material": "cotton|denim|linen|wool|cashmere|polyester|leather|suede|knit|jersey|satin|chiffon|silk|velvet|tweed|nylon|fleece",
    "secondary_material": "secondary material or null",
    "fabric_weight": "light|medium|heavy",
    "texture": "smooth|ribbed|fuzzy|structured|matte|glossy|textured|woven",
    "stretch": "none|low|medium|high"
  },

  "silhouette_profile": {
    "fit": "skinny|slim|regular|relaxed|oversized",
    "structure": "structured|semi_structured|soft|flowing",
    "length": "cropped|regular|long|mini|midi|maxi",
    "volume": "low|medium|high"
  },

  "style_identity": {
    "aesthetic_styles": ["minimalist", "classic", "streetwear", "athleisure", "bohemian", "preppy", "elegant", "sporty", "vintage", "modern", "edgy", "romantic"],
    "trend_alignment": "timeless|trend_forward|seasonal",
    "statement_level": "basic|moderate|statement"
  },

  "styling_compatibility": {
    "layering_compatibility": ["works_under_jackets", "works_over_shirts", "works_with_knits", "works_with_blazers", "standalone_piece"],
    "matching_bottoms": ["jeans", "tailored_trousers", "skirts", "shorts", "leggings", "chinos", "joggers"],
    "matching_tops": ["tshirt", "shirt", "sweater", "blouse", "tank_top", "polo", "crop_top"],
    "matching_outerwear": ["blazer", "trench", "coat", "denim_jacket", "leather_jacket", "cardigan", "bomber", "parka"],
    "matching_shoes": ["sneakers", "loafers", "boots", "heels", "sandals", "oxford", "mules", "flats"]
  },

  "occasion_profile": {
    "formality_level": "very_casual|casual|smart_casual|business_casual|business|formal",
    "occasions": ["daily_wear", "work", "weekend", "evening", "travel", "event", "date", "sport"]
  },

  "seasonality": {
    "seasons": ["spring", "summer", "fall", "winter"],
    "temperature_suitability": "hot|warm|mild|cold"
  },

  "garment_details": {
    "neckline": "v_neck|crewneck|turtleneck|boat_neck|collared|scoop|square|halter|off_shoulder|cowl|null",
    "sleeves": {
      "length": "short|long|3/4|sleeveless|null",
      "style": "raglan|puff|bell|bishop|cap|dolman|null"
    },
    "length_type": "crop|regular|longline|midi|maxi|mini|knee_length|ankle_length|null",
    "waist_rise": "low_rise|mid_rise|high_rise|null",
    "closure": "zipper|buttons|snap|wrap|pull_on|hook_and_eye|drawstring|velcro|null",
    "has_pockets": true|false,
    "distressed": true|false,
    "embellishments": [],
    "transparency": "opaque|semi_sheer|sheer"
  },

  "versatility": {
    "versatility_score": 0.0 to 1.0,
    "capsule_wardrobe_friendly": true|false
  },

  "confidence_score": 0.0 to 1.0
}
```
