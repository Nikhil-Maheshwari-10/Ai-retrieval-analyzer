import streamlit as st
import pandas as pd
import time
import os
from tempfile import NamedTemporaryFile
from core.config import settings
from core.logging import metrics_logger, logger
from app.service.ingestion import IngestionService
from app.service.chunking import ChunkingService
from app.service.embeddings import EmbeddingService
from app.service.retrieval import RetrievalService
from app.service.llm_router import LLMRouter

st.set_page_config(page_title="AI Retrieval Analyzer", layout="wide")

st.title("🚀 AI Retrieval Analyzer")
st.markdown("---")

# Constants for persistence
DATA_DIR = "data"
DOC_DIR = os.path.join(DATA_DIR, "doc")
EMBED_DIR = os.path.join(DATA_DIR, "embeddings")
os.makedirs(DOC_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

# Session State Initialization
if "retriever" not in st.session_state:
    # Try to load existing index
    st.session_state.retriever = RetrievalService.load_index(EMBED_DIR)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "metrics" not in st.session_state:
    st.session_state.metrics = []

if "clear_msg" in st.session_state:
    st.success("✅ Processed documents cleared!")
    del st.session_state.clear_msg

# Document Ingestion Block
st.subheader("📁 Document Ingestion")
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

col_proc, col_cls, col_spacer = st.columns([1.2, 1.2, 7.6], gap="small")

with col_proc:
    if uploaded_files:
        if st.button("🚀 Process Documents"):
            with st.spinner(f"Processing {len(uploaded_files)} PDF(s)..."):
                # Initialize or get retriever
                if st.session_state.retriever is None:
                    temp_embedder = EmbeddingService()
                    dimension = temp_embedder.model.get_sentence_embedding_dimension()
                    retriever = RetrievalService(dimension)
                else:
                    retriever = st.session_state.retriever

                for uploaded_file in uploaded_files:
                    # Save file to doc directory
                    file_path = os.path.join(DOC_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    # 1. Ingestion
                    text = IngestionService.extract_text_from_bytes(uploaded_file.read())
                    
                    # 2. Chunking
                    chunker = ChunkingService()
                    chunks = chunker.hybrid_chunking(text)
                    
                    # 3. Embeddings
                    embedder = EmbeddingService()
                    embeddings = embedder.generate_embeddings(chunks)
                    
                    # 4. Retrieval (Update FAISS)
                    retriever.add_documents(chunks, embeddings, uploaded_file.name)
                
                # Save index to disk after all files processed
                retriever.save_index(EMBED_DIR)
                st.session_state.retriever = retriever
                st.success(f"Successfully processed {len(uploaded_files)} document(s).")
                st.rerun()

with col_cls:
    if st.session_state.retriever:
        if st.button("🗑️ Clear Storage", help="Delete all uploaded documents and indices"):
            import shutil
            if os.path.exists(DATA_DIR):
                shutil.rmtree(DATA_DIR)
            os.makedirs(DOC_DIR, exist_ok=True)
            os.makedirs(EMBED_DIR, exist_ok=True)
            st.session_state.retriever = None
            st.session_state.metrics = []
            metrics_logger.clear()
            if "query_input" in st.session_state:
                st.session_state.query_input = ""
            st.session_state.clear_msg = True
            st.rerun()

st.markdown("---")

# Query Interface Section
st.subheader("💬 Query Interface")
query = st.text_input("Ask a question about the document", key="query_input")

if query and st.session_state.retriever:
    router = LLMRouter()
    embedder = EmbeddingService()
    
    # Retrieval
    query_embedding = embedder.generate_query_embedding(query)
    context, retrieval_time = st.session_state.retriever.retrieve(query_embedding, k=settings.TOP_K)
    
    st.markdown("### Results")
    
    # Gemini Inference
    with st.status(f"Running {settings.GEMINI_MODEL} Inference..."):
        gemini_result = router.get_gemini_response(query, [c for c, s in context])
        metrics_logger.log_performance(f"Gemini ({settings.GEMINI_MODEL})", {
            "retrieval_time": retrieval_time,
            "generation_time": gemini_result["generation_time"],
            "total_latency": retrieval_time + gemini_result["generation_time"],
            "token_usage": gemini_result["token_usage"],
            "cost": gemini_result["cost"]
        })
        st.write(f"**Gemini ({settings.GEMINI_MODEL}) Answer:**")
        st.write(gemini_result["answer"])
    
    # Ollama Inference
    with st.status(f"Running {settings.OLLAMA_MODEL} Inference..."):
        try:
            ollama_result = router.get_ollama_response(query, [c for c, s in context])
            metrics_logger.log_performance(f"Ollama ({settings.OLLAMA_MODEL})", {
                "retrieval_time": retrieval_time,
                "generation_time": ollama_result["generation_time"],
                "total_latency": retrieval_time + ollama_result["generation_time"],
                "token_usage": ollama_result["token_usage"],
                "cost": ollama_result["cost"]
            })
            st.write(f"**Ollama ({settings.OLLAMA_MODEL}) Answer:**")
            st.write(ollama_result["answer"])
        except Exception as e:
            st.error(f"Ollama Error: {e}")

# Performance Comparison Table
st.markdown("---")
st.subheader("📊 Performance Comparison")
comparison_data = metrics_logger.get_comparison_table()
if comparison_data:
    df = pd.DataFrame(comparison_data)
    cols = ["model", "retrieval_time", "generation_time", "total_latency", "token_usage", "cost"]
    df = df[cols]
    df.index = df.index + 1
    
    # Apply styling
    st.table(df.style.format({
        "retrieval_time": "{:.4f}s",
        "generation_time": "{:.4f}s",
        "total_latency": "{:.4f}s",
        "cost": "${:.6f}"
    }))
else:
    st.info("No queries performed yet.")
