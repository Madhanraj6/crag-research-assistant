"""
Configuration module — loads settings from .env file with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_secret(key_name: str, default: str = "") -> str:
    """Helper to get secret from Streamlit secrets first, then os.getenv."""
    try:
        import streamlit as st
        # Streamlit >= 1.28 raises FileNotFoundError/KeyError if no secrets.toml or missing key
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name, default)

class Settings:
    # --- OpenRouter ---
    OPENROUTER_API_KEY: str = get_secret("OPENROUTER_API_KEY", "")

    # --- Embeddings ---
    EMBEDDING_MODEL: str = get_secret("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # --- Vector Store (FAISS) ---
    FAISS_INDEX_PATH: str = get_secret("FAISS_INDEX_PATH", "data/faiss_index")

    # --- Retrieval ---
    TOP_K: int = int(os.getenv("TOP_K", "10"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.40"))
    MAX_CORRECTION_ATTEMPTS: int = int(os.getenv("MAX_CORRECTION_ATTEMPTS", "2"))

    # --- Server ---
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
