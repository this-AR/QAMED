# Medical RAG System 

*An advanced Retrieval-Augmented Generation system specifically designed for medical question answering using vector databases and medical-specialized language models.*

## Overview

This project implements a comprehensive Medical RAG (Retrieval-Augmented Generation) system that processes medical textbooks and provides accurate, contextual answers to complex medical questions. The system leverages Qdrant vector database for efficient document storage and retrieval, combined with medical-specific embeddings and language models to ensure high-quality medical information retrieval. Generation is handled via Groq LLMs with token streaming support.

The system addresses the critical need for accurate medical information access by breaking down complex medical queries into focused sub-questions, retrieving relevant textbook content, and generating well-sourced answers. This approach ensures that medical professionals, students, and researchers can quickly access precise information from authoritative medical texts.

Key innovations include intelligent query decomposition, medical-specific content labeling, chapter-aware document organization, and a sophisticated reranking pipeline that maximizes retrieval accuracy for medical domain knowledge.

## Architecture Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Medical PDF    │    │   Query Input    │    │  Final Answer   │
│   Textbook      │    │ (Complex Medical │    │   Synthesis     │
│                 │    │   Question)      │    │                 │
└────────┬────────┘    └─────────┬────────┘    └─────────▲───────┘
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────┐    ┌──────────────────┐              │
│ Document        │    │ Query            │              │
│ Processing      │    │ Decomposition    │              │
│ • PDF Loading   │    │ • FLAN-T5 Model  │              │
│ • Chunking      │    │ • Sub-questions  │              │
│ • Metadata      │    │ • Deduplication  │              │
└────────┬────────┘    └─────────┬────────┘              │
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────┴───────┐
│ Vector Storage  │◄───┤ Retrieval &      │    │ Answer          │
│ • Qdrant DB     │    │ Reranking        │    │ Generation      │
│ • S-PubMedBert  │    │ • Semantic Search│    │ • Groq LLMs     │
│ • Medical Emb.  │    │ • BGE Reranker   │    │ • Medical Focus │
│ • Rich Metadata │    │ • Top-K Results  │    │ • Source Grounding│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Features

- **🏥 Medical-Specialized Components**
  - S-PubMedBert embeddings for medical text understanding
  - Chapter-aware document organization with medical textbook TOC mapping
  - Automatic content labeling (@definition, @symptoms, @diagnosis, @treatment, etc.)

- **🧠 Intelligent Query Processing**
  - Complex query decomposition using FLAN-T5 model
  - Semantic deduplication to avoid redundant sub-questions
  - Pronoun resolution and medical context preservation

- **🗄️ Advanced Vector Database**
  - Qdrant vector database with COSINE similarity
  - Rich metadata filtering and search capabilities
  - Scalable performance for large medical document collections

- **🎯 Sophisticated Retrieval Pipeline**
  - Two-stage retrieval with semantic search and reranking
  - BGE reranker for improved document relevance scoring
  - Sub-question specific document retrieval

- **🤖 High-Quality Answer Generation**
  - Groq LLMs with token streaming for fast responses
  - Medical-specific prompting for accuracy
  - Source-grounded responses with textbook attribution

- **📊 Comprehensive Metadata System**
  - Chapter mapping for better organization
  - Content type classification
  - Page-level source tracking

## Prerequisites & Installation

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Minimum 16GB RAM (32GB recommended for large textbooks)

### Step 1: Environment Setup

Create and activate a Python virtual environment:

```bash
# Using conda (recommended)
conda create -n medical-rag python=3.9
conda activate medical-rag

# Or using venv
python -m venv medical-rag
source medical-rag/bin/activate  # On Windows: medical-rag\Scripts\activate
```

### Step 2: Install Core Dependencies

```bash
# Install main packages
pip install langchain qdrant-client sentence-transformers transformers pymupdf groq python-dotenv

# Install LangChain community package
pip install -U langchain-community

# Install additional tokenization support
pip install sacremoses

# Install PyTorch with CUDA support (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Verify Installation

```python
# Quick verification script
import torch
from transformers import pipeline
from qdrant_client import QdrantClient

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("Installation successful!")
```

## Usage / Quick-Start

### Basic Setup and Demo

1. **Clone/Download the notebook** and place your medical PDF in an accessible location.

2. **Update the PDF path** in the notebook:
```python
# Update this path to your medical textbook
pdf_path = "/path/to/your/medical_textbook.pdf"
```

**CLI alternative (no notebook):**
```bash
python ingest.py --pdf "/path/to/your/medical_textbook.pdf"
```

3. **Run the core setup cells** to initialize the system:

```python
# Initialize embeddings
embedding_model = SentenceTransformerEmbeddings(
    model_name="pritamdeka/S-PubMedBert-MS-MARCO"
)

# Setup Qdrant vector database
qdrant_client = QdrantClient(":memory:")  # For testing
collection_name = "medical_documents"
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
```

4. **Process your medical textbook**:

```python
# Load and process the PDF
chunks = load_and_chunk_pdf(pdf_path)
print(f"Processed {len(chunks)} document chunks")

# Create vector store and add documents
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embedding_model
)
vectorstore.add_documents(chunks)
```

5. **Ask medical questions**:

```python
# Example medical queries
queries = [
    "What is the inguinal canal, and what are its contents?",
    "Describe the anatomy and function of the liver",
    "What are the surfaces and borders of the heart?"
]

# Process a query
query = "What is the anatomy of the stomach?"
answers = answer_medical_query(query)

# View results
for subquestion, answer in answers.items():
    print(f"Q: {subquestion}")
    print(f"A: {answer}\n")
```

### Advanced Usage

**Custom Chapter Mapping**: Modify the `CHAPTER_MAP` dictionary to match your specific textbook's table of contents.

**Persistent Storage (Qdrant Cloud)**: Use your cloud URL + API key:
```python
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
```

**Batch Processing**: Process multiple medical textbooks by running the document loading pipeline for each PDF.

### Expected Output

The system will output structured answers like:
```
🧩 Subquestion 1/2: What is the anatomy of the stomach?
📘 Answer: The stomach is a J-shaped dilated portion of the alimentary canal...

🧩 Subquestion 2/2: What are the anatomical relations of the stomach?
📘 Answer: The stomach has anterior and posterior surfaces with specific anatomical relations...
```

### Performance Tips

- Use GPU acceleration for faster model inference
- Adjust `chunk_size` and `chunk_overlap` based on your textbook structure
- Fine-tune retrieval parameters (`k` value) based on document collection size
- Consider using persistent Qdrant storage for large document collections

## 🔐 Environment Variables

Set the following variables (local dev or Hugging Face Spaces secrets):

- `QDRANT_URL` - Qdrant Cloud URL
- `QDRANT_API_KEY` - Qdrant API key
- `GROQ_API_KEY` - Groq API key
- `GROQ_MODEL` - Optional, e.g., `llama-3.1-70b-versatile`

## 🖥️ Streamlit App

Run locally:

```bash
streamlit run app.py
```

The app provides:
- Text input for questions
- Streaming answer display
- Source citations from Qdrant
- 1-10 rating feedback per question

## 🚀 Deploy on Hugging Face Spaces

1. Create a new **Streamlit** Space.
2. Add `app.py` and `requirements.txt` to the Space repo.
3. Configure **Secrets** in the Space settings:
  - `QDRANT_URL`, `QDRANT_API_KEY`, `GROQ_API_KEY`, optionally `GROQ_MODEL`
4. Make sure your Qdrant Cloud collection is already populated (run the notebook to index data).


## 🔮 Future Improvements

The current pipeline provides a strong baseline for medical RAG systems, but several enhancements are under consideration to improve **accuracy**, **scalability**, and **evaluation robustness**:

### 📈 Evaluation and Faithfulness

- **Answer Faithfulness Checks**  
  Integrate hallucination detection (e.g., Retrieval-Augmented Verification or confidence thresholds) to validate factual grounding.

- **Metric-Based Evaluation Pipeline**  
  Incorporate standard QA metrics such as:
  - **BLEU**, **ROUGE** for answer overlap with ground truth
  - **BERTScore** for semantic similarity
  - **F1/EM** scores for extractive subcomponents
  - **LangChain Evaluation** or **RAGAS** for end-to-end QA fidelity

- **Manual Test Set for Medical QA**  
  Build a curated benchmark of diverse medical queries with verified reference answers for better tuning.

---

### 🧠 Model Enhancements

- **Optimized Query Decomposition**  
  Replace FLAN-T5 with more robust instruction-tuned models like:
  - `LLaMA-3-Instruct` (4-bit quantized)
  - `OpenBioLLM-8B-Instruct` or `BioMistral` for medical-specific comprehension

- **Improved Generation Models**  
  Experiment with:
  - `OpenBioLLM-8B (Q4_K_M)` for fast, accurate generation
  - `MedLM` or `ClinicalCamel` for highly contextual responses
  - Evaluate trade-offs between Groq-hosted models and larger domain-specific models

- **Reranking Improvements**  
  Replace `BAAI/bge-reranker` with multilingual or domain-specialized rerankers (e.g., `bce-reranker` or BioELECTRA-based models)

---

### 🛠 Retrieval & Infrastructure

- **Persistent Qdrant Setup**  
  Use Docker or managed Qdrant cloud with disk-backed storage for multi-book deployment.

- **Chunking Parameter Autotuning**  
  Implement automatic chunk-size optimization based on TOC heuristics or passage entropy.

- **Optional Web Search Fallback**  
  Integrate web-based fallback (e.g., SerpAPI or Perplexity API) if internal document recall fails.

- **UI/API Layer**  
  Add minimal REST API or Gradio UI for easier access by non-technical users.

---

> 💡 *All improvements will be tracked and versioned to ensure reproducibility and performance benchmarking.*

For more detailed examples and advanced configurations, explore the complete notebook implementation.

## 📝 Third‑Party Licences
| Component | Author | Licence |
|-----------|--------|---------|
| FLAN‑T5 (google/flan‑t5‑base) | Google | Apache‑2.0 |
| Groq LLMs (API) | Groq | See provider terms |
| S‑PubMedBert‑MS‑MARCO | Pritam Deka | MIT |

The above models are downloaded at runtime via Hugging Face Hub; their original licences apply.

