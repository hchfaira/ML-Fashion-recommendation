from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # Application
    app_name: str = "Fashion Recommendation System"
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # Google Gemini API (primary)
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    
    # OpenAI API (optional, for embeddings)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    
    # Model Settings
    # Embedding models: 
    #   - Gemini: "gemini-embedding-001" (default, uses GOOGLE_API_KEY)
    #   - OpenAI: "text-embedding-3-small", "text-embedding-3-large" (requires OPENAI_API_KEY)
    embedding_model: str = Field(default="gemini-embedding-001", alias="EMBEDDING_MODEL")
    vision_model: str = Field(default="gemini-1.5-flash", alias="VISION_MODEL")
    llm_model: str = Field(default="gemini-1.5-flash", alias="LLM_MODEL")
    
    # Weather API
    weather_api_key: Optional[str] = Field(default=None, alias="WEATHER_API_KEY")
    
    # Database
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")
    
    # Neo4j Graph Database
    neo4j_uri: Optional[str] = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: Optional[str] = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: Optional[str] = Field(default=None, alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    
    # Outfit Storage (json, neo4j, auto)
    outfit_storage_type: str = Field(default="auto", alias="OUTFIT_STORAGE_TYPE")
    
    # Cache
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")
    embedding_cache_size: int = Field(default=10000, alias="EMBEDDING_CACHE_SIZE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # silently ignore unknown env vars (e.g. NEO4J_USER)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
