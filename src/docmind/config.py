from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DOCMIND_")

    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 400
    chunk_overlap: int = 80
    max_upload_mb: int = 10


settings = Settings()