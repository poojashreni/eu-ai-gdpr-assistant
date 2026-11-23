from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables and an optional `.env` file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    llm_provider: Literal["gemini", "openai"] = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4o-mini"
    embed_model: str = "intfloat/multilingual-e5-small"
    chroma_dir: Path = Path("data/chroma")
    chunks_path: Path = Path("data/chunks.jsonl")
    sources_path: Path = Path("data/raw/sources.json")
    top_k: int = 6
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    rate_limit_per_hour: int = 10
    daily_cap: int = 200
    max_question_chars: int = 1000
    api_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
