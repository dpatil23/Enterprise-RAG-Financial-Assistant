from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration class.
    Pydantic automatically reads values from the .env file.
    Supports Groq, Puter.js, OpenAI, and Ollama.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Unified LLM Provider Settings (Puter / Groq / OpenAI / Ollama)
    LLM_PROVIDER: str = "puter"
    LLM_BASE_URL: str = "https://api.puter.com/puterai/openai/v1/"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "claude-3-5-sonnet"

    # Legacy / Alternative Groq settings
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Embeddings & Vector DB
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 4

    # Neo4j Graph Database Settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "ragproject123"


# Singleton instance
settings = Settings()
