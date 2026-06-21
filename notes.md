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
