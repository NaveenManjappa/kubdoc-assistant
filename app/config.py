import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_URL")
    QDRANT_COLLECTION = "enterprise_rag"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_GUARD_MODEL = os.getenv("GROQ_GUARD_MODEL", "openai/gpt-oss-20b")
    PORTKEY_API_KEY=os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG="rag1"
    GROQ_SLUG2="rag2"
    PORTKEY_CONFIG=os.getenv("PORTKEY_CONFIG")
    
settings = Settings()
