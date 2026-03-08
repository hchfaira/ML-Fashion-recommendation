# Color Extraction Prompt

Analyze the colors in this clothing item and provide:

1. Primary color name
2. Secondary color (if any)
3. Hex codes for main colors
4. Color temperature (warm/cool/neutral)
5. Color depth (light/medium/dark)

Return as JSON:
```json
{
    "primary": "color name",
    "secondary": "color name or null",
    "hex_codes": ["#XXXXXX"],
    "temperature": "warm|cool|neutral",
    "depth": "light|medium|dark"
}
```
