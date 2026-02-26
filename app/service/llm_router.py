import ollama
import litellm
import time
from typing import List, Dict, Any, Optional
from core.config import settings
from core.logging import logger
from app.schema.schemas import DocumentChunk

class LLMRouter:
    def __init__(self):
        self.system_prompt = settings.SYSTEM_PROMPT
        self._ensure_ollama_model()

    def _ensure_ollama_model(self):
        """
        Checks if the configured Ollama model exists locally, pulls it if not.
        """
        model_full_name = settings.OLLAMA_MODEL
        try:
            client = ollama.Client(host=settings.OLLAMA_HOST)
            
            # List local models
            logger.info(f"Checking Ollama models at {settings.OLLAMA_HOST}...")
            resp = client.list()
            models = resp.models if hasattr(resp, 'models') else (resp if isinstance(resp, list) else [])
            
            installed_names = []
            for m in models:
                name = getattr(m, 'model', getattr(m, 'name', None))
                if name:
                    installed_names.append(name)
            
            # Precise check
            exists = any(name == model_full_name for name in installed_names)
            
            if exists:
                logger.success(f"Ollama model '{model_full_name}' is ready.")
            else:
                logger.warning(f"Ollama model '{model_full_name}' NOT found. Starting pull...")
                
                # Pull with streaming progress
                current_status = ""
                for progress in client.pull(model_full_name, stream=True):
                    status = progress.get('status')
                    if status and status != current_status:
                        logger.info(f"Pulling {model_full_name}: {status}")
                        current_status = status
                        
                logger.success(f"Successfully pulled '{model_full_name}'")
        except Exception as e:
            logger.error(f"Ollama setup failed: {e}")

    def _prepare_prompt(self, query: str, context: List[DocumentChunk]) -> str:
        context_text = "\n\n".join([f"Source: {c.source}\nContent: {c.text}" for c in context])
        return f"Context:\n{context_text}\n\nQuestion: {query}"

    def _get_litellm_response(self, model: str, query: str, context: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Unified method to get response and calculate metrics using LiteLLM.
        """
        start_time = time.time()
        prompt = self._prepare_prompt(query, context)
        
        # Get pricing for the specific model name (e.g., gemini-2.5-flash-lite)
        clean_model_name = model.split("/")[-1]
        pricing = settings.get_pricing(clean_model_name)
        price_input = pricing["input_price_per_1m"]
        price_output = pricing["output_price_per_1m"]
        
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                api_key=settings.GEMINI_API_KEY if "gemini" in model else None,
                api_base=settings.OLLAMA_HOST if "ollama" in model else None
            )
            
            generation_time = time.time() - start_time
            usage = response.usage
            
            # Pricing from settings (per 1M tokens)
            cost = (usage.prompt_tokens * price_input / 1000000) + (usage.completion_tokens * price_output / 1000000)
            
            return {
                "answer": response.choices[0].message.content,
                "generation_time": generation_time,
                "token_usage": usage.total_tokens,
                "cost": cost
            }
        except Exception as e:
            logger.error(f"LiteLLM error for model {model}: {str(e)}")
            raise e

    def get_gemini_response(self, query: str, context: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Inference using Gemini API via LiteLLM.
        """
        return self._get_litellm_response(
            model=f"gemini/{settings.GEMINI_MODEL}",
            query=query,
            context=context
        )

    def get_ollama_response(self, query: str, context: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Inference using local Ollama via LiteLLM.
        """
        return self._get_litellm_response(
            model=f"ollama/{settings.OLLAMA_MODEL}",
            query=query,
            context=context
        )

