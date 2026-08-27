from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "agentic-true-vectorless-rag"
    app_env: str = "development"
    log_level: str = "INFO"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    index_dir: str = "./data/indexes"
    metadata_dir: str = "./data/metadata"

    retrieval_top_k: int = Field(default=10, ge=1, le=100)
    retrieval_max_pages: int = Field(default=20, ge=1, le=500)

    ocr_enabled: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()