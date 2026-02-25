# 🚀 AI Retrieval Analyzer

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-FF4B4B.svg)](https://streamlit.io/)
[![LiteLLM](https://img.shields.io/badge/Powered%20by-LiteLLM-green.svg)](https://github.com/BerriAI/litellm)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-blue.svg)](https://github.com/facebookresearch/faiss)

A high-performance, modular **Retrieval-Augmented Generation (RAG)** pipeline designed for document intelligence. This project features dual-inference capabilities, allowing side-by-side comparison between cloud-based (Google Gemini) and local (Ollama) LLMs.

---

## 🏗️ Architecture: RAG Pipeline

```mermaid
graph TD
    %% Ingestion Pipeline
    subgraph Ingestion [Ingestion Pipeline]
        style Ingestion fill:#f5faff,stroke:#0055b3,stroke-width:2px
        Upload[User Uploads PDFs] --> Extractor[<b>Ingestion Service</b><br/>Extract text via pypdf]
        Extractor --> Splitter[<b>Chunking Service</b><br/>Hybrid: Semantic + Sliding Window<br/><i>Size: 500, Overlap: 75</i>]
        Splitter --> EmbedText[<b>Embedding Service</b><br/>Generate Embeddings<br/><i>all-MiniLM-L6-v2</i>]
        EmbedText --> StoreFAISS[<b>Vector Storage</b><br/>FAISS Index<br/><i>Local Persistence</i>]
    end

    %% Retrieval Pipeline
    subgraph Retrieval [Retrieval Pipeline]
        style Retrieval fill:#fff9f5,stroke:#b35900,stroke-width:2px
        Query[User Enters Query] --> EmbedQuery[<b>Embedding Service</b><br/>Generate Query Embedding]
        EmbedQuery --> VectorSearch[<b>Vector Search</b><br/>Search FAISS Index]
        VectorSearch --> Context[<b>Context Prep</b><br/>Retrieve Top K Chunks]
        Context --> Router[<b>LLM Router</b><br/>Route to Dual Providers]
        Router --> Gemini[<b>Gemini 2.5 Flash</b><br/>Cloud Inference]
        Router --> Ollama[<b>Ollama (Gemma 3)</b><br/>Local Inference]
        Gemini --> Results[<b>Metrics Dashboard</b><br/>Display Answers & Latency]
        Ollama --> Results
    end

    %% Connection
    StoreFAISS -.->|Persistent Index| VectorSearch

    %% Styling
    classDef service fill:#fff,stroke:#333,stroke-width:1px;
    class Extractor,Splitter,EmbedText,StoreFAISS,EmbedQuery,VectorSearch,Router,Gemini,Ollama service;
```

---

## ✨ Key Features

- **📥 Intelligent Ingestion**: Structured PDF text extraction using `pypdf`.
- **🧩 Hybrid Chunking**: Combines Semantic Splitting with Sliding-Window logic for optimal context retention.
- **⚡ Vector Search**: Real-time retrieval using **FAISS** with local persistence.
- **🤖 Dual-Inference Router**: 
  - **Cloud**: Google Gemini Model (via LiteLLM).
  - **Local**: Ollama Model for privacy and zero-cost inference.
- **📈 Performance Analytics**: Detailed dashboard tracking latency, token usage, and estimated costs per query.
- **🎨 Modern UI**: Clean, responsive Streamlit interface with session state management.

---

## 🏗️ Project Architecture

```bash
├── app/
│   ├── schema/         # Pydantic schemas for data validation
│   └── service/        # Core RAG logic
│       ├── ingestion.py   # PDF text extraction
│       ├── chunking.py    # Text splitting strategies
│       ├── embeddings.py  # Vector generation (Sentence-Transformers)
│       ├── retrieval.py   # FAISS management and search
│       └── llm_router.py  # LiteLLM routing logic
├── core/               # App configuration and logging
├── data/               # Local storage for PDFs and FAISS indices (Gitignored)
├── model/              # Local storage for embedding models (Gitignored)
├── ui/                 # Streamlit frontend implementation
└── main.py             # Entry point script
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) (installed and running)
- Google Gemini API Key

### 🛠️ Installation & Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/Nikhil-Maheshwari-10/Ai-retrieval-analyzer.git
cd Ai-retrieval-analyzer
```

#### 2. Install Dependencies
Choose **one** of the following methods to set up your environment:

**Option A: Using Poetry (Recommended)**
```bash
# Install dependencies
poetry install
```

**Option B: Using Pip & Virtual Environment (venv)**
```bash
# Create a virtual environment
python -m venv venv

# Activate the environment

# On Mac/Linux:
source venv/bin/activate

# On Windows:
# .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Environment Setup**:
   Create a `.env` file in the root directory:
   ```env
   # Embedding Configuration
   EMBEDDING_MODEL=your_embedding_model_name
   CHUNK_SIZE=your_chunk_size
   CHUNK_OVERLAP=your_chunk_overlap

   # Huggingface Configuration
   HF_TOKEN=your_huggingface_token

   # LLM Configuration
   GEMINI_MODEL=your_gemini_model_name
   GEMINI_API_KEY=your_gemini_api_key

   # Ollama Configuration
   OLLAMA_MODEL=your_ollama_model_name
   OLLAMA_HOST=your_ollama_host_url

   # Retrieval Configuration
   TOP_K=your_top_k_value

   # Prompt 
   SYSTEM_PROMPT="your_system_prompt_here"
   ```

### Running the Application

Start the system via the entry point:

**If using Poetry:**
```bash
poetry run python main.py
```

**If using venv:**
```bash
python main.py
```

---

## 🛠️ Usage Guide

1. **Upload Documents**: Drag and drop PDF files into the ingestion block. The system will extract, chunk, and index them automatically.
2. **Process**: Click "Process Documents" to update the vector database.
3. **Query**: Enter your question in the search bar.
4. **Compare**: View results from both Gemini and Ollama side-by-side.
5. **Analyze**: Check the "Performance Comparison" table at the bottom to compare latency and cost efficiency.

---

## 📊 Comparison Strategy

| Feature | Local (Ollama) | Cloud (Gemini) |
| :--- | :--- | :--- |
| **Latency** | Hardware-dependent | Low / Medium |
| **Privacy** | High (Internal only) | Public Cloud |
| **Cost** | Free ($0) | Pay-per-token |
| **Reliability** | Works offline | Requires Internet |

---

## 📝 Author
Developed with ❤️ by **Nikhil Maheshwari**
