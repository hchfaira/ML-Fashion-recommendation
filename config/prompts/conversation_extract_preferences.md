# Preference Extraction Prompt

Analyze this conversation and extract any style preferences mentioned:

{history}

Return a JSON object with:
```json
{
    "liked_colors": [],
    "disliked_colors": [],
    "preferred_styles": [],
    "avoided_styles": [],
    "occasions_mentioned": [],
    "body_type_mentioned": null,
    "any_other_preferences": []
}
```

Only include what was explicitly mentioned.
