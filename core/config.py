import os
import json
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class Settings(BaseSettings):
    # Embedding Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
    HF_TOKEN: str = os.getenv("HF_TOKEN")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP"))

    # LLM Settings
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST")

    # Retrieval Settings
    TOP_K: int = int(os.getenv("TOP_K", 5))

    # Prompt Template
    SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT")

    # Pricing Data
    _PRICING: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_pricing()

    def _load_pricing(self):
        pricing_path = os.path.join(os.path.dirname(__file__), "pricing.json")
        try:
            with open(pricing_path, 'r') as f:
                self._PRICING = json.load(f)
        except Exception:
            self._PRICING = {}

    def get_pricing(self, model: str) -> Dict[str, float]:
        # Return model specific pricing or default to 0
        if model in self._PRICING:
            return self._PRICING[model]
        if "ollama" in model.lower():
            return self._PRICING.get("ollama_default", {"input_price_per_1m": 0.0, "output_price_per_1m": 0.0})
        # Default fallback
        return {"input_price_per_1m": 0.0, "output_price_per_1m": 0.0}

    class Config:
        env_file = ".env"

settings = Settings()
