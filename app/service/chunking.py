import re
from typing import List
from transformers import AutoTokenizer
from core.config import settings
from core.logging import logger

class ChunkingService:
    def __init__(self):
        logger.info(f"Loading tokenizer for model: {settings.EMBEDDING_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.EMBEDDING_MODEL, 
            token=settings.HF_TOKEN
        )

    def _validate_params(self, chunk_size: int, overlap: int):
        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    def sliding_window_chunking(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Implements high-performance sliding-window chunking strictly based on tokens.
        """
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP
        self._validate_params(chunk_size, overlap)
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return []

        chunks = []
        step = chunk_size - overlap
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            if i + chunk_size >= len(tokens):
                break
                
        logger.info(f"Generated {len(chunks)} sliding-window chunks")
        return chunks

    def semantic_chunking(self, text: str) -> List[str]:
        """
        Optimized semantic chunking using incremental token counting to avoid O(n^2).
        """
        # Split by paragraphs while preserving them
        paragraphs = re.split(r'(\n\s*\n)', text)
        semantic_chunks = []
        
        current_chunk_paragraphs = []
        current_chunk_token_count = 0
        
        for i in range(0, len(paragraphs), 2):
            p = paragraphs[i]
            # Paragraph separator if it exists
            sep = paragraphs[i+1] if i+1 < len(paragraphs) else ""
            
            p_full = p + sep
            p_tokens = self.tokenizer.encode(p_full, add_special_tokens=False)
            p_token_count = len(p_tokens)
            
            if current_chunk_token_count + p_token_count <= settings.CHUNK_SIZE:
                current_chunk_paragraphs.append(p_full)
                current_chunk_token_count += p_token_count
            else:
                if current_chunk_paragraphs:
                    semantic_chunks.append("".join(current_chunk_paragraphs).strip())
                
                # If a single paragraph is larger than chunk_size, it will be handled by hybrid logic
                current_chunk_paragraphs = [p_full]
                current_chunk_token_count = p_token_count
        
        if current_chunk_paragraphs:
            semantic_chunks.append("".join(current_chunk_paragraphs).strip())
            
        logger.info(f"Generated {len(semantic_chunks)} semantic segments")
        return semantic_chunks

    def hybrid_chunking(self, text: str) -> List[str]:
        """
        Production-ready hybrid chunking: semantic grouping with sliding-window fallback 
        for large blocks, maintaining consistent overlap.
        """
        semantic_blocks = self.semantic_chunking(text)
        final_chunks = []
        
        for block in semantic_blocks:
            block_tokens = self.tokenizer.encode(block, add_special_tokens=False)
            if len(block_tokens) > settings.CHUNK_SIZE:
                # Optimized: encode once, decode slices
                step = settings.CHUNK_SIZE - settings.CHUNK_OVERLAP
                for i in range(0, len(block_tokens), step):
                    chunk_tokens = block_tokens[i : i + settings.CHUNK_SIZE]
                    final_chunks.append(self.tokenizer.decode(chunk_tokens))
                    if i + settings.CHUNK_SIZE >= len(block_tokens):
                        break
            else:
                final_chunks.append(block)
        
        logger.info(f"Total hybrid chunks produced: {len(final_chunks)}")
        return final_chunks
