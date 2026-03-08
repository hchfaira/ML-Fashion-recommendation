"""
Configuration Loader Module

Centralized configuration loading system for the Fashion Recommendation System.
Loads prompts, parameters, and data from external files.

This module provides:
- ConfigLoader: Main configuration loader class
- get_config(): Cached singleton access to configuration
- Prompt loading from markdown files
- Parameter loading from JSON files
- Data loading from JSON files

Usage:
    from config.loader import get_config

    config = get_config()
    prompt = config.get_prompt("vision_system")
    params = config.get_parameters("llm")
    data = config.get_data("color_data")
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union
from functools import lru_cache
import json
import logging

# Use standard logging to avoid circular import with src.core.logger
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration loading fails."""
    pass


class ConfigLoader:
    """
    Centralized configuration loader.
    
    Loads and caches:
    - Prompts from markdown files (config/prompts/)
    - Parameters from JSON files (config/parameters/)
    - Data from JSON files (config/data/)
    
    Attributes:
        base_path: Base path to configuration directory
        prompts_path: Path to prompts directory
        parameters_path: Path to parameters directory
        data_path: Path to data directory
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the configuration loader.
        
        Args:
            base_path: Base path to config directory. Defaults to config/ in project root.
        """
        if base_path is None:
            # Default to config/ directory relative to this file
            base_path = Path(__file__).parent
        
        self.base_path = Path(base_path)
        self.prompts_path = self.base_path / "prompts"
        self.parameters_path = self.base_path / "parameters"
        self.data_path = self.base_path / "data"
        
        # Caches
        self._prompts_cache: Dict[str, str] = {}
        self._parameters_cache: Dict[str, Dict] = {}
        self._data_cache: Dict[str, Dict] = {}
        
        # Validate paths exist
        self._validate_paths()
        
        logger.info(f"ConfigLoader initialized with base path: {self.base_path}")
    
    def _validate_paths(self) -> None:
        """Validate that configuration directories exist."""
        required_paths = [
            self.prompts_path,
            self.parameters_path,
            self.data_path
        ]
        
        for path in required_paths:
            if not path.exists():
                logger.warning(f"Configuration directory not found: {path}")
                # Create directory if it doesn't exist
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created configuration directory: {path}")
    
    # ==================== Prompt Loading ====================
    
    def get_prompt(
        self,
        prompt_name: str,
        **kwargs: Any
    ) -> str:
        """
        Load a prompt from a markdown file.
        
        Args:
            prompt_name: Name of the prompt file (without .md extension)
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Prompt content as string, with variables substituted
            
        Raises:
            ConfigurationError: If prompt file not found
        """
        # Check cache first
        if prompt_name not in self._prompts_cache:
            self._load_prompt(prompt_name)
        
        prompt = self._prompts_cache[prompt_name]
        
        # Substitute variables if provided
        if kwargs:
            try:
                prompt = prompt.format(**kwargs)
            except KeyError as e:
                logger.warning(f"Missing variable in prompt {prompt_name}: {e}")
                # Do partial substitution for missing keys
                for key, value in kwargs.items():
                    prompt = prompt.replace(f"{{{key}}}", str(value))
        
        return prompt
    
    def _load_prompt(self, prompt_name: str) -> None:
        """Load a prompt file into cache."""
        file_path = self.prompts_path / f"{prompt_name}.md"
        
        if not file_path.exists():
            raise ConfigurationError(
                f"Prompt file not found: {file_path}. "
                f"Available prompts: {self.list_prompts()}"
            )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove markdown header if present (first line starting with #)
        lines = content.split('\n')
        if lines and lines[0].startswith('#'):
            content = '\n'.join(lines[1:]).strip()
        
        self._prompts_cache[prompt_name] = content
        logger.debug(f"Loaded prompt: {prompt_name}")
    
    def list_prompts(self) -> list:
        """List available prompt names."""
        if not self.prompts_path.exists():
            return []
        return [f.stem for f in self.prompts_path.glob("*.md")]
    
    def reload_prompt(self, prompt_name: str) -> str:
        """Force reload a prompt from disk."""
        if prompt_name in self._prompts_cache:
            del self._prompts_cache[prompt_name]
        return self.get_prompt(prompt_name)
    
    # ==================== Parameters Loading ====================
    
    def get_parameters(
        self,
        params_name: str,
        key: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """
        Load parameters from a JSON file.
        
        Args:
            params_name: Name of the parameters file (without .json extension)
            key: Optional nested key to retrieve (dot notation: "llm.default_temperature")
            default: Default value if key not found
            
        Returns:
            Parameters dict or specific value if key provided
            
        Raises:
            ConfigurationError: If parameters file not found
        """
        # Check cache first
        if params_name not in self._parameters_cache:
            self._load_parameters(params_name)
        
        params = self._parameters_cache[params_name]
        
        # Return full dict if no key specified
        if key is None:
            return params
        
        # Navigate nested keys
        return self._get_nested_value(params, key, default)
    
    def _load_parameters(self, params_name: str) -> None:
        """Load a parameters file into cache."""
        file_path = self.parameters_path / f"{params_name}.json"
        
        if not file_path.exists():
            raise ConfigurationError(
                f"Parameters file not found: {file_path}. "
                f"Available parameters: {self.list_parameters()}"
            )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self._parameters_cache[params_name] = json.load(f)
        
        logger.debug(f"Loaded parameters: {params_name}")
    
    def list_parameters(self) -> list:
        """List available parameter file names."""
        if not self.parameters_path.exists():
            return []
        return [f.stem for f in self.parameters_path.glob("*.json")]
    
    def reload_parameters(self, params_name: str) -> Dict:
        """Force reload parameters from disk."""
        if params_name in self._parameters_cache:
            del self._parameters_cache[params_name]
        return self.get_parameters(params_name)
    
    # ==================== Data Loading ====================
    
    def get_data(
        self,
        data_name: str,
        key: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """
        Load data from a JSON file.
        
        Args:
            data_name: Name of the data file (without .json extension)
            key: Optional nested key to retrieve (dot notation)
            default: Default value if key not found
            
        Returns:
            Data dict or specific value if key provided
            
        Raises:
            ConfigurationError: If data file not found
        """
        # Check cache first
        if data_name not in self._data_cache:
            self._load_data(data_name)
        
        data = self._data_cache[data_name]
        
        # Return full dict if no key specified
        if key is None:
            return data
        
        # Navigate nested keys
        return self._get_nested_value(data, key, default)
    
    def _load_data(self, data_name: str) -> None:
        """Load a data file into cache."""
        file_path = self.data_path / f"{data_name}.json"
        
        if not file_path.exists():
            raise ConfigurationError(
                f"Data file not found: {file_path}. "
                f"Available data files: {self.list_data()}"
            )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self._data_cache[data_name] = json.load(f)
        
        logger.debug(f"Loaded data: {data_name}")
    
    def list_data(self) -> list:
        """List available data file names."""
        if not self.data_path.exists():
            return []
        return [f.stem for f in self.data_path.glob("*.json")]
    
    def reload_data(self, data_name: str) -> Dict:
        """Force reload data from disk."""
        if data_name in self._data_cache:
            del self._data_cache[data_name]
        return self.get_data(data_name)
    
    # ==================== Utility Methods ====================
    
    def _get_nested_value(
        self,
        data: Dict,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get a nested value using dot notation.
        
        Args:
            data: Dictionary to search
            key: Dot-separated key path (e.g., "llm.default_temperature")
            default: Default value if not found
            
        Returns:
            Value at key path or default
        """
        keys = key.split('.')
        value = data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def reload_all(self) -> None:
        """Clear all caches and force reload on next access."""
        self._prompts_cache.clear()
        self._parameters_cache.clear()
        self._data_cache.clear()
        logger.info("All configuration caches cleared")
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Get a summary of all loaded configuration.
        
        Returns:
            Dict with prompts, parameters, and data keys
        """
        return {
            "prompts": self.list_prompts(),
            "parameters": self.list_parameters(),
            "data": self.list_data(),
            "cached": {
                "prompts": list(self._prompts_cache.keys()),
                "parameters": list(self._parameters_cache.keys()),
                "data": list(self._data_cache.keys())
            }
        }


# Singleton instance
_config_loader: Optional[ConfigLoader] = None


@lru_cache()
def get_config() -> ConfigLoader:
    """
    Get the cached configuration loader singleton.
    
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def reset_config() -> None:
    """Reset the configuration loader singleton (useful for testing)."""
    global _config_loader
    _config_loader = None
    get_config.cache_clear()
