"""Application configuration via environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Amref Help Desk RAG", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")

    secret_key: str = Field(default="change-me", alias="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=1440, alias="JWT_EXPIRE_MINUTES")
    rate_limit: str = Field(default="30/minute", alias="RATE_LIMIT")

    # PostgreSQL (Docker)
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="amref", alias="POSTGRES_USER")
    postgres_password: str = Field(default="amref_secret", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="amref_helpdesk", alias="POSTGRES_DB")

    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8001, alias="CHROMA_PORT")
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")

    # LLM Provider: "anthropic" | "openai" | "ollama"
    llm_provider: Literal["anthropic", "openai", "ollama"] = Field(
        default="anthropic", alias="LLM_PROVIDER"
    )

    # Anthropic (primary LLM)
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    # claude-3-5-haiku-20241022  → fastest + cheapest
    # claude-3-5-sonnet-20241022 → best quality
    anthropic_model: str = Field(default="claude-3-5-haiku-20241022", alias="ANTHROPIC_MODEL")

    # OpenAI (kept as optional fallback)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")

    # Ollama (local fallback)
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")

    # Embedding provider: "ollama" | "sentence-transformers"
    embedding_provider: Literal["ollama", "sentence-transformers"] = Field(
        default="ollama", alias="EMBEDDING_PROVIDER"
    )
    # Used when embedding_provider = "sentence-transformers"
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    # Used when embedding_provider = "ollama"
    ollama_embedding_model: str = Field(
        default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL"
    )

    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(default=5, alias="TOP_K_RETRIEVAL")
    top_k_images: int = Field(default=3, alias="TOP_K_IMAGES")
    mmr_diversity: float = Field(default=0.3, alias="MMR_DIVERSITY")
    rerank_top_n: int = Field(default=5, alias="RERANK_TOP_N")

    kb_base_url: str = Field(default="https://helpdesk.amref.ac.ke", alias="KB_BASE_URL")
    kb_index_url: str = Field(
        default="https://helpdesk.amref.ac.ke/knowledgebase.php", alias="KB_INDEX_URL"
    )

    data_dir: str = Field(default="./data", alias="DATA_DIR")
    raw_data_dir: str = Field(default="./data/raw", alias="RAW_DATA_DIR")
    processed_data_dir: str = Field(default="./data/processed", alias="PROCESSED_DATA_DIR")
    images_dir: str = Field(default="./data/images", alias="IMAGES_DIR")
    static_images_dir: str = Field(
        default="./backend/app/static/images", alias="STATIC_IMAGES_DIR"
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", alias="LOG_FILE")

    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field
    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def ensure_dirs(self) -> None:
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.images_dir,
            self.static_images_dir,
            self.chroma_persist_dir,
            str(Path(self.log_file).parent),
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
