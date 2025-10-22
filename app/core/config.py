"""
Configuration settings for O!Store Agent
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import os

os.environ["HTTP_PROXY"] = "http://172.27.129.0:3128"
os.environ["HTTPS_PROXY"] = "http://172.27.129.0:3128"

class Settings:
    """Application settings"""

    def __init__(self):
        # Load environment variables
        env_path = Path(__file__).parent.parent.parent / ".env"
        load_dotenv(env_path)

        # OpenAI settings
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        # Qdrant settings
        self.qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))

        # Collections
        self.specs_collection: str = os.getenv("SPECS_COLLECTION", "mobiles_specs")
        self.docs_collection: str = os.getenv("DOCS_COLLECTION", "mobile_docs")

        # Paths
        self.data_dir: Path = Path(__file__).parent.parent.parent / "data"
        self.logs_dir: Path = Path(__file__).parent.parent.parent / "logs"

        # Ensure logs directory exists
        self.logs_dir.mkdir(exist_ok=True)

        # Logging
        self.analytics_log: Path = self.logs_dir / "analytics.json"
        self.query_log: Path = self.logs_dir / "queries.json"

        # Retriever settings
        self.retriever_k: int = int(os.getenv("RETRIEVER_K", "3"))

    def validate(self) -> bool:
        """Validate required settings"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        return True


# Global settings instance
settings = Settings()