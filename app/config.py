from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    weaviate_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    data_dir: str = "data/papers"
    chunk_size: int = 900
    chunk_overlap: int = 120
    retrieval_top_k: int = 6
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
