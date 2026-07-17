# Architecture Notes & Tradeoffs

## Phase 1: Ingestion Pipeline Upgrades

### Moving from PyMuPDF to Docling
- **Decision:** Replaced `PyMuPDFLoader` with `DoclingLoader` (`ExportType.DOC_CHUNKS`).
- **Reasoning:** PyMuPDF only extracts raw text, which completely scrambles tables, equations, and diagrams. Docling natively parses document structures and converts tables into markdown format, which is essential for medical RAG (e.g., drug dosage tables, muscle origins/insertions).
- **Tradeoff:** Docling is significantly heavier. It downloads and runs several HuggingFace models (layout analysis, OCR, sentence transformers) locally, which increases ingestion latency and memory overhead compared to PyMuPDF.

### Fixing Docling Memory Crashes on Large PDFs
- **Decision 1:** Forced single-threaded processing (`pipeline_options.accelerator_options.num_threads = 1`).
  - **Reasoning:** Large medical textbooks (>500 pages) caused memory spikes during multiprocessing, leading the OS to garbage-collect page handles and crash the pipeline (`AssertionError: Page backend was unloaded`).
  - **Tradeoff:** Ingestion is slower because it can no longer process multiple pages in parallel, but it is vastly more stable.

- **Decision 2:** Switched to `PyPdfiumDocumentBackend`.
  - **Reasoning:** The default `DoclingParseDocumentBackend` was still hitting memory limits and unloading the backend even on a single thread due to its heavy parsing logic.
  - **Tradeoff:** `PyPdfiumDocumentBackend` is highly memory-efficient but relies slightly more on visual layout heuristics rather than native PDF text streams. For highly visual medical PDFs with complex layouts, this is an acceptable (and sometimes preferable) tradeoff for stability.

### Handling OCR Warnings
- **Decision:** Retained default OCR behavior despite console warnings (`RapidOCR returned empty result!`).
  - **Reasoning:** Medical textbooks contain numerous images. When Docling's OCR engine scans an image with no text (e.g., a pure anatomy plate without labels), it logs a warning. This is expected and harmless. We chose *not* to disable OCR, because we still want it to extract text from diagrams that *do* have labels.
  - **Tradeoff:** The terminal output is noisy during ingestion, but data extraction quality is maximized.

## Phase 2: Hierarchical Chunking & Retrieval

### Single-Level Retrieval vs. Three-Level Indexing
- **Decision:** Implemented a single retrieval granularity (paragraph-level, ~300 tokens) with parent context expansion, rather than indexing 3 levels (sentence, paragraph, section) separately.
- **Reasoning:**
  1. Short sentence embeddings (80 tokens) are often too noisy for dense retrieval (models expect 100-500 tokens).
  2. A cross-encoder reranker can be biased by chunk length when comparing mixed granularities.
  3. Embedding the same text 3 times at different windows bloats the index for redundant content.
  4. Parent-child retrieval is achieved by finding the medium chunk and fetching the large chunk separately.
- **Tradeoff:** Queries requiring extremely localized fact retrieval might miss on strict sentence matching, but paragraph retrieval generally yields higher quality semantic matches. We will defer multi-level indexing until an eval dataset justifies it.

### SQLite for Parent Document Storage
- **Decision:** Used a local SQLite database (`data/doc_store.db`) to store H2-level parent sections instead of Upstash Redis.
- **Reasoning:** Redis is better for fast cache lookups, but SQLite is zero-cost, local, and better suited for storing large text blobs (2000+ tokens per section) persistently. We use fetch-by-key, so SQLite's read performance is more than adequate.
- **Tradeoff:** Introduces an extra local DB file to manage alongside Qdrant, but avoids polluting the Redis cache with persistent document data.

### Parent Truncation via `leaf_offset`
- **Decision:** We track the exact character offset (`leaf_offset`) of each chunk within its parent during chunking, and use this to center a truncation window (snapped to sentences) when a parent section exceeds the LLM context budget (e.g., 3000 tokens).
- **Reasoning:** Deriving the offset at query time using string `.find()` is fragile due to overlapping chunks sharing prefixes and potential whitespace normalization drift. Storing `leaf_offset` as metadata guarantees correctness at zero runtime cost.

### BM25 / RRF Fusion
- **Decision:** Dropped from scope for now.
- **Reasoning:** Although mentioned in early designs, the current production codebase only uses Qdrant dense retrieval. Introducing hybrid sparse-dense search is a separate architectural addition that needs its own testing and maintenance (e.g., idempotency on re-ingestion).

### Idempotent Re-ingestion
- **Decision:** Changed `--force` re-ingestion from an upsert-by-ID pattern to a delete-by-book-then-insert pattern.
- **Reasoning:** With upsert-by-ID, if Docling's OCR or layout parsing changes slightly between runs, the text hashes change, leading to new chunk IDs. The old chunks would be orphaned and never deleted. Deleting all points for a `book_name` before insertion guarantees a clean state.

## Phase 3: LLM & UI Refinements

### Query Decomposition Reliability
- **Decision:** Added few-shot examples directly into the query decomposer's system prompt (e.g. splitting "what is X and what are its contents").
- **Reasoning:** Zero-shot decomposition frequently failed to split multi-part questions if they were grammatically a single sentence. Hardcoded examples guarantee the LLM identifies and splits multiple conceptual entities, ensuring comprehensive vector search coverage.

### SIMPLE Query Handling
- **Decision:** Maintained the "SIMPLE" query bypass (which extracts raw text directly from the DB instead of generating a new LLM response) but increased the extracted sentence count from 2 to 5 and switched it to extract from the intact parent section rather than the raw chunk.
- **Reasoning:**
  1. The raw 300-token chunks were occasionally split midway through a sentence by the text splitter. Pulling from the SQLite parent section via `leaf_offset` guarantees clean, full sentences.
  2. 2 sentences were often too short for medical definitions. 5 sentences provide a comprehensive paragraph.
- **Tradeoff:** SIMPLE queries still do not get a synthesized LLM answer, but the raw text extraction is now high-quality and readable.

### Rating Slider Visibility
- **Decision:** Moved the Langfuse rating slider UI rendering outside of the COMPLEX RAG pipeline loop.
- **Reasoning:** Cached answers and SIMPLE answers stop the script early (`st.stop()`) to save time. By moving the rating UI up, we ensure users can log feedback metrics for *all* answer types, not just live-generated ones.


## Phase 4: Contextual Enrichment & RAG Tuning

### Contextual Micro-Headers & Breadcrumbs
- **Decision:** Implemented a pre-ingestion LLM pass to generate hierarchical breadcrumbs and micro-summaries for every 200-token chunk. This metadata is prepended to the raw chunk text.
- **Reasoning:** Medical text is extremely dense. By embedding both the raw text (for exact anatomical details) and the summary (for high-level contextual disambiguation), the vector database can better differentiate between similar structures (e.g., surgical neck of humerus vs. surgical neck of other bones).
- **Tradeoff:** Increases ingestion cost and time due to the extra LLM calls per chunk, but drastically improves retrieval precision.

### Expanding the Retrieval Net & Shrinking the Context Window
- **Decision:** Increased Top-K for both Dense and Sparse retrieval from 15 to 75. Increased Cross-Encoder Reranker Top-N from 4 to 6. Shrunk the LLM's maximum parent context window from 3000 tokens to 2048 tokens.
- **Reasoning:** Retrieving only 15 chunks starved the cross-encoder of candidates. By widening the initial net (Top-75) and letting the cross-encoder sift through the noise to pick the Top-6, we ensure high recall. Shrinking the context window to 2048 tokens prevents the LLM from getting distracted by bloated parent documents or copying irrelevant MCQ artifacts into the final answer.
- **Tradeoff:** Marginally higher latency for Qdrant/BM25 search and reranking, but significantly better grounded and accurate LLM synthesis.

### Prompt Refinement (v3)
- **Decision:** Developed the `v3` ("Expert Medical Synthesis") prompt template, which explicitly instructs the LLM to synthesize a cohesive, flowing response and to ignore non-textbook artifacts (like MCQs).
- **Reasoning:** Previous prompts (v1 and v2) resulted in robotic, rigidly structured bullet points, or occasionally leaked multiple-choice options into the answer. `v3` produces state-of-the-art textbook-quality responses.


## Pending Tasks Before Final Deploy (V2.0/V3.0 Remaining)

While the core pipeline (Docling, Hybrid Search, Qdrant Cloud, Groq streaming, Caching, Tracing, and Ragas Eval) is fully operational and deployed, the following architectural upgrades remain to reach the true Version 2.0 / 3.0 "Research Grade" standard:

### 1. Embeddings & Chunking
- **MedCPT Embeddings Migration:** We are currently using \`S-PubMedBert-MS-MARCO\`. We need to migrate to the asymmetric \`MedCPT\` encoders (separate query and article encoders) for state-of-the-art medical matching.
- **Multi-Level Chunking:** Transition from our current single-granularity 200-token chunks to a simultaneous 3-level index (Sentence 80t + Paragraph 250t + Section 1000t).
- **Automatic TOC Chapter Mapping:** Replace the current basic book tracking with automated extraction of Chapter and Section headers from Docling to enable metadata pre-filtering by chapter.

### 2. Advanced Retrieval Strategies
- **HyDE (Hypothetical Document Embeddings):** Implement selective HyDE for complex, multi-concept queries to bridge the vocabulary gap before hitting Qdrant.
- **LLMLingua Context Compression:** Trim the retrieved chunks down to just their most relevant sentences to save LLM context window space and reduce distraction.
- **Multi-Hop LangGraph Agent:** Build an agentic loop for cross-chapter retrieval (e.g., if a question requires combining anatomy from chapter 2 with pathology from chapter 5).

### 3. Evaluation & Feedback
- **Feedback Loop:** Build the UI thumbs-up/down buttons in Streamlit to collect user ratings and feed them back into the evaluation dataset.
- **Golden Dataset:** Run an automated script to generate 500+ QA pairs directly from our chunks to establish a baseline RAGAS score matrix.
- **Metrics Dashboard:** Create a separate Streamlit page or Grafana dashboard to visualize Langfuse latency, costs, and RAGAS scores over time.
