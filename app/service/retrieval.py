import faiss
import numpy as np
import time
import pickle
import os
from typing import List, Tuple, Dict, Optional
from core.logging import logger
from app.schema.schemas import DocumentChunk

class RetrievalService:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: Dict[int, DocumentChunk] = {}

    def add_documents(self, chunks: List[str], embeddings: np.ndarray, source: str):
        """
        Adds chunks and their embeddings to the FAISS index with metadata.
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        start_idx = self.index.ntotal
        self.index.add(embeddings.astype('float32'))
        
        for i, chunk_text in enumerate(chunks):
            idx = start_idx + i
            self.metadata[idx] = DocumentChunk(
                chunk_id=str(idx),
                source=source,
                text=chunk_text
            )
        
        logger.info(f"Added {len(chunks)} documents to FAISS index. Total: {self.index.ntotal}")

    def save_index(self, folder_path: str):
        """
        Saves the FAISS index and metadata to the specified folder.
        """
        os.makedirs(folder_path, exist_ok=True)
        index_path = os.path.join(folder_path, "index.faiss")
        metadata_path = os.path.join(folder_path, "metadata.pkl")
        
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump((self.dimension, self.metadata), f)
        
        logger.success(f"Index and metadata saved to {folder_path}")

    @classmethod
    def load_index(cls, folder_path: str) -> Optional['RetrievalService']:
        """
        Loads a RetrievalService instance from the specified folder.
        """
        index_path = os.path.join(folder_path, "index.faiss")
        metadata_path = os.path.join(folder_path, "metadata.pkl")
        
        if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
            return None
            
        try:
            index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                dimension, metadata = pickle.load(f)
            
            instance = cls(dimension)
            instance.index = index
            instance.metadata = metadata
            logger.success(f"Index loaded from {folder_path} with {index.ntotal} vectors")
            return instance
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return None

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves top-K similar documents from FAISS.
        """
        start_time = time.time()
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.metadata:
                results.append((self.metadata[idx], float(dist)))
        
        retrieval_time = time.time() - start_time
        logger.info(f"Retrieval took {retrieval_time:.4f}s for k={k}")
        
        return results, retrieval_time
