# Configuration System Documentation

## Overview

The Fashion Recommendation System uses a centralized configuration system that externalizes:
- **Prompts**: LLM and Vision model prompts stored as Markdown files
- **Parameters**: Model parameters, thresholds, and scoring weights as JSON
- **Data**: Static data like color mappings, style patterns, occasion rules as JSON

This separation allows modifying system behavior without changing source code.

## Directory Structure

```
config/
├── __init__.py              # Exports get_config, ConfigLoader
├── loader.py                # Central configuration loader
├── settings.py              # Environment-based settings (API keys, etc.)
├── prompts/                 # LLM and Vision prompts
│   ├── vision_system.md
│   ├── vision_analysis.md
│   ├── vision_simple_analysis.md
│   ├── attribute_extraction.md
│   ├── targeted_extraction.md
│   ├── color_extraction.md
│   ├── llm_explain_outfit_system.md
│   ├── llm_explain_outfit_user.md
│   ├── llm_styling_tip.md
│   ├── llm_personalize_message.md
│   ├── llm_outfit_comparison.md
│   ├── llm_outfit_rejection.md
│   ├── conversation_system.md
│   ├── conversation_user.md
│   ├── conversation_clarifying.md
│   └── conversation_extract_preferences.md
├── parameters/              # Model parameters and thresholds
│   ├── model_parameters.json
│   └── conversation_settings.json
└── data/                    # Static data mappings
    ├── color_data.json
    ├── occasion_data.json
    ├── morphology_data.json
    ├── silhouette_data.json
    ├── compatibility_data.json
    ├── activity_data.json
    ├── weather_data.json
    └── schedule_data.json
```

## Usage

### Basic Usage

```python
from config import get_config

config = get_config()

# Load a prompt with variable substitution
prompt = config.get_prompt("llm_explain_outfit_system", tone_description="friendly")

# Load parameters
max_tokens = config.get_parameters("model_parameters", "llm.default_max_tokens")

# Load static data
neutrals = config.get_data("color_data", "neutrals")
```

### Loading Prompts

Prompts are stored as Markdown files in `config/prompts/`. The first line (header) is stripped.

```python
# Load full prompt
system_prompt = config.get_prompt("vision_system")

# Load with variable substitution
prompt = config.get_prompt(
    "llm_styling_tip",
    tone_description="professional",
    improvement_area="color harmony",
    outfit_items="- Blue blazer\n- White shirt"
)
```

**Variable syntax in prompt files**: Use `{variable_name}` placeholders.

### Loading Parameters

Parameters are JSON files in `config/parameters/`.

```python
# Load all parameters from a file
all_llm_params = config.get_parameters("model_parameters", "llm")

# Load a specific nested value
temperature = config.get_parameters(
    "model_parameters", 
    "llm.default_temperature",
    default=0.7
)

# Dot notation for nested access
category_weight = config.get_parameters(
    "model_parameters",
    "scoring.compatibility.category_weight"
)
```

### Loading Data

Static data files are in `config/data/`.

```python
# Load full data file
color_data = config.get_data("color_data")

# Load specific key
neutrals = config.get_data("color_data", "neutrals")

# With default fallback
harmony_rules = config.get_data(
    "color_data", 
    "harmony_rules",
    default={}
)
```

## Adding New Configuration

### Adding a New Prompt

1. Create a new file in `config/prompts/` with `.md` extension
2. First line should be a Markdown header (will be stripped)
3. Use `{variable}` syntax for template variables

Example `config/prompts/my_new_prompt.md`:
```markdown
# My New Prompt

You are a {role} assistant.

Please analyze the following: {content}

Respond in JSON format.
```

Usage:
```python
prompt = config.get_prompt(
    "my_new_prompt",
    role="fashion",
    content="outfit description"
)
```

### Adding New Parameters

1. Add to existing JSON file or create new file in `config/parameters/`
2. Use nested structure for organization

Example structure:
```json
{
    "feature_name": {
        "threshold": 0.5,
        "enabled": true,
        "weights": {
            "factor_a": 0.3,
            "factor_b": 0.7
        }
    }
}
```

### Adding New Data

1. Create or edit JSON file in `config/data/`
2. Document the structure in comments or README

## Configuration in Tests

### Mocking Configuration

```python
from unittest.mock import patch, MagicMock
from config import reset_config

def test_with_mock_config():
    reset_config()  # Clear singleton
    
    mock_config = MagicMock()
    mock_config.get_prompt.return_value = "Test prompt"
    mock_config.get_parameters.return_value = {"key": "value"}
    
    with patch("config.get_config", return_value=mock_config):
        # Your test code
        pass
```

### Using Test Fixtures

```python
import pytest
from pathlib import Path
from config.loader import ConfigLoader

@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration with custom values."""
    # Create test prompts
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "test_prompt.md").write_text("# Test\nTest content")
    
    # Create test parameters
    params_dir = tmp_path / "parameters"
    params_dir.mkdir()
    (params_dir / "test_params.json").write_text('{"key": "value"}')
    
    # Create test data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test_data.json").write_text('{"items": [1, 2, 3]}')
    
    return ConfigLoader(base_path=tmp_path)
```

## Best Practices

1. **Never hardcode values** that might need to change - use configuration
2. **Use meaningful defaults** when loading configuration
3. **Document all configuration** files with comments or README
4. **Version control** all configuration files
5. **Use environment variables** for secrets (via `config/settings.py`)
6. **Test with both** real config and mocked config
7. **Reload configuration** when needed with `config.reload_all()`

## File Reference

### model_parameters.json

Contains all model-related parameters:
- `llm`: LLM generation parameters (temperatures, max_tokens)
- `vision`: Vision model parameters
- `embedding`: Embedding generation parameters
- `scoring`: Scoring weights for compatibility, formality, color harmony
- `thresholds`: System thresholds

### conversation_settings.json

Contains conversation-related settings:
- `tone_presets`: Tone descriptions for different styles
- `intent_patterns`: Keywords for intent detection
- `intent_system_additions`: System prompt additions per intent

### color_data.json

Contains color-related data:
- `neutrals`: List of neutral colors
- `color_hsl_map`: Color name to HSL mapping
- `harmony_rules`: Color harmony thresholds

### occasion_data.json

Contains occasion-related mappings:
- `occasion_formality`: Occasion to formality level mapping
- `occasion_styles`: Preferred styles per occasion
- `occasion_avoid`: Styles to avoid per occasion
- `formality_ranges`: Formality ranges per occasion

### morphology_data.json

Contains body type recommendations:
- `body_type_recommendations`: Per body type style advice
- `general_recommendations`: Default recommendations
- `priority_areas`: Focus areas per body type

## Migration Notes

When migrating existing code to use centralized configuration:

1. Identify hardcoded values in the module
2. Determine if it's a prompt, parameter, or data
3. Create/update the appropriate configuration file
4. Update the module to load from config
5. Update tests to work with both real and mocked config
