from pydantic import BaseModel
from typing import List, Optional

class DocumentChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    metadata: Optional[dict] = {}

class RAGResponse(BaseModel):
    answer: str
    context: List[DocumentChunk]
    retrieval_time: float
    generation_time: float
    total_latency: float
    token_usage: int
    cost: float
