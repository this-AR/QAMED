---
title: QAMED CHATBOT
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# Medical RAG System (QAMed)

*An advanced Retrieval-Augmented Generation system specifically designed for medical question answering using hybrid vector databases, cross-encoder reranking, and medical-specialized language models.*

## Overview

QAMed is a comprehensive Medical RAG system that processes dense medical textbooks and provides accurate, contextual answers to complex medical questions. It uses a decoupled **Parent-Child Retrieval Strategy**, ensuring that while the search happens on precise, semantic micro-chunks, the language model is fed the full, cohesive parent paragraphs (up to 2048 tokens) to generate its answers.

By leveraging **Docling** for deep PDF structural parsing, **Qdrant** + **BM25** for hybrid retrieval, and **Groq** for lightning-fast token streaming, QAMed ensures that medical professionals and students can quickly access precise information from authoritative texts without hallucinations.

## Architecture

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
│ • Docling Parse │    │ • Groq LLM       │              │
│ • 200-t Chunks  │    │ • Sub-questions  │              │
│ • LLM Micro-Tags│    │ • Caching (Redis)│              │
└────────┬────────┘    └─────────┬────────┘              │
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────┴───────┐
│ Storage         │◄───┤ Hybrid Retrieval │    │ Answer          │
│ • Qdrant (Dense)│    │ & Reranking      │    │ Generation      │
│ • SQLite (Text) │    │ • Dense + BM25   │    │ • Groq LLMs     │
│ • S-PubMedBert  │    │ • RRF Fusion     │    │ • v3 Prompt     │
│ • BM25 Index    │    │ • BGE Reranker   │    │ • Streaming     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Features

- **🏥 Advanced Ingestion Pipeline**
  - **Docling PDF Parsing:** Natively parses complex medical tables, clinical diagrams, and two-column layouts.
  - **Contextual Micro-Headers:** Every chunk is passed through an LLM during ingestion to generate hierarchical breadcrumbs and anatomical summaries, ensuring disambiguation.
  - **Parent-Child Strategy:** Chunks are small (200 tokens) for precise vector matching, but map to larger parent sections stored in a local SQLite database for context-rich generation.

- **🧠 Intelligent Query Processing**
  - **LLM-Based Decomposition:** Breaks down multi-part complex medical questions into independent sub-queries.
  - **Query Caching:** Skips expensive retrieval for exact-match or semantically similar repeated questions.
  - **SIMPLE vs. COMPLEX Routing:** Bypasses LLM generation for simple definitional queries, extracting raw textbook sentences directly.

- **🎯 Sophisticated Hybrid Retrieval**
  - **Dense Search (Qdrant):** Powered by `S-PubMedBert` embeddings for deep semantic understanding.
  - **Sparse Search (BM25):** Built dynamically in-memory at startup to catch exact drug names and anatomical terminology.
  - **Reciprocal Rank Fusion (RRF):** Fuses Dense and Sparse results (Top K=75).
  - **Cross-Encoder Reranking:** Filters the noisy Top 75 pool down to the absolute best 6 chunks using `BAAI/bge-reranker-base`.

- **🤖 High-Quality Answer Generation**
  - **Groq LLMs:** Llama 3.1 70B powers the synthesis with ultra-fast token streaming.
  - **Medical-Specific Prompting (v3):** Explicitly instructed to synthesize cohesive, textbook-quality flowing responses and filter out MCQs or non-textbook artifacts.

## Prerequisites & Installation

### System Requirements
- Python 3.9 or higher
- API Keys for Qdrant Cloud and Groq

### Step 1: Environment Setup

```bash
python -m venv qamed-env
source qamed-env/bin/activate  # On Windows: qamed-env\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Environment Variables
Create a `.env` file in the root directory:
```env
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-70b-versatile
PROMPT_VERSION=v3
```

## Usage

### 1. Ingest a Medical Textbook

Use the CLI to process a PDF textbook. The `--force` flag ensures any old chunks for that specific book are cleanly deleted from Qdrant and SQLite before inserting the new ones.

```bash
python data/ingest.py --pdf "/path/to/your/medical_textbook.pdf" --force
```
*Note: Docling layout parsing is CPU-intensive. A large medical textbook may take 30-60 minutes to extract before uploading begins.*

### 2. Run the Streamlit UI

Start the application interface:

```bash
streamlit run app.py
```
Upon startup, the app will connect to Qdrant, download your chunks, and dynamically build the BM25 sparse index in memory. 

## 🚀 Deploy on Hugging Face Spaces

1. Create a new **Streamlit** Space.
2. Push this repository to your Space.
3. Configure your **Secrets** in the Space settings (`QDRANT_URL`, `QDRANT_API_KEY`, `GROQ_API_KEY`).
4. Ensure your local `data/doc_store.db` is committed to Git (do not add it to `.gitignore`!) so the HF Space can access the parent sections.

## 📝 Third-Party Licenses
| Component | Author | License |
|-----------|--------|---------|
| Llama 3.1 (Groq API) | Meta | Llama 3.1 License |
| S-PubMedBert-MS-MARCO | Pritam Deka | MIT |
| Docling | IBM | MIT |
| BGE Reranker | BAAI | MIT |

*The above models are downloaded at runtime via Hugging Face Hub or accessed via API.*
