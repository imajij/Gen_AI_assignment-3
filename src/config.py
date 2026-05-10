"""Configuration utilities for environment-driven settings."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    base_url: str
    api_key: str
    model: str
    embedding_model: str
    vector_store_path: str


def load_config() -> AppConfig:
    """Load configuration from environment variables with defaults."""
    base_url = os.getenv("BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    api_key = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))
    model = os.getenv("MODEL", "gemma-4-26b-a4b-it")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")

    return AppConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        embedding_model=embedding_model,
        vector_store_path=vector_store_path,
    )
