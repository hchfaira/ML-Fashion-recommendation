```markdown
# Simple Attribute Extraction Prompt

Analyze this clothing item and extract key attributes.

IMPORTANT: Always identify and return the PRIMARY COLOR - this is the most important attribute.

Return ONLY this JSON structure (no markdown, no extra text):

{
  "category": "top|bottom|dress|outerwear|shoes|accessory",
  "subcategory": "specific type (e.g., t-shirt, jeans, blazer, sneakers)",
  "primary_color": "THE MAIN COLOR NAME (e.g., black, white, navy, lavender, beige)",
  "secondary_color": "second color if any, or null",
  "hex_code": "#XXXXXX for the primary color",
  "pattern": "solid|stripe|plaid|floral|graphic|abstract|other",
  "material": "cotton|denim|wool|silk|leather|synthetic|knit|other",
  "formality": "casual|smart_casual|business|formal",
  "fit": "slim|regular|relaxed|oversized"
}

COLOR IDENTIFICATION RULES:
- Look at the dominant color covering most of the garment
- Use common color names: black, white, gray, navy, blue, red, pink, green, beige, brown, cream, lavender, burgundy, olive, coral, mustard, etc.
- If multicolored, list the primary (dominant) color first
- Always provide a hex code approximation

RESPOND WITH JSON ONLY. NO MARKDOWN FORMATTING.
```
