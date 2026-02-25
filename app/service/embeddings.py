from sentence_transformers import SentenceTransformer
from core.config import settings
from core.logging import logger
from typing import List
import numpy as np
import logging
import warnings

# Suppress transformers and sentence-transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Specifically suppress the "sharded" weights warning from transformers
warnings.filterwarnings("ignore", message=".*The following layers were not sharded.*")
# Suppress the Streamlit ScriptRunContext warning
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

import os

class EmbeddingService:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        
        # Define a local path for the model
        model_cache_dir = "model"
        safe_model_name = self.model_name.replace("/", "_")
        self.local_path = os.path.join(model_cache_dir, safe_model_name)
        
        if os.path.exists(self.local_path):
            logger.success(f"Loading embedding model from local storage: {self.local_path}")
            self.model = SentenceTransformer(self.local_path)
        else:
            logger.info(f"Downloading and saving model to local storage: {self.local_path}")
            self.model = SentenceTransformer(self.model_name, use_auth_token=settings.HF_TOKEN)
            os.makedirs(model_cache_dir, exist_ok=True)
            self.model.save(self.local_path)
            logger.success(f"Model saved locally at {self.local_path}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of strings without showing progress bars.
        """
        try:
            # Silence batches progress bar
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise e

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generates embedding for a single query string.
        """
        return self.generate_embeddings([query])[0]
