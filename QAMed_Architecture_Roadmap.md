# QAMed — Architecture, Gaps & Roadmap

---

## Current Pipeline (Notebook v5) — What Exists

```
PDF (PyMuPDF)
  → sentence-aware chunking (250 tokens, 50 overlap, NLTK punkt)
  → manual keyword label detection (@definition, @treatment etc)
  → hardcoded CHAPTER_MAP (manual per book)
  → S-PubMedBert embeddings
  → Qdrant (in-memory) + FAISS index
  → BM25 sparse retrieval
  → RRF fusion (alpha 0.8 dense, 0.2 sparse)
  → BGE reranker → top 2 docs per subquery
  → FLAN-T5 query decomposition
    → Groq API generation (streamed)
    → raw text answer (citations in UI, streaming enabled, no eval)
```

---

## Gaps in Current Pipeline

### Critical Gaps (Break functionality)

1. **PDF parsing loses all table and diagram content** — PyMuPDF extracts plain text only. Anatomy tables (nerve supplies, muscle origins), drug dosage tables, clinical diagrams are all invisible to the system.
2. **Qdrant is in-memory** — resets on every restart. All embeddings lost. Not usable for real users.
3. **No streaming** — users see blank screen for 20-30 seconds then wall of text. Unusable UX.
4. **Local inference is too slow** — T4 GPU gives 20-30 tokens/second. Groq gives 800+ tokens/second.
5. **No citation in answers** — system gives no source attribution. Users cannot verify.
6. **FLAN-T5 decomposition is weak** — fails on complex multi-concept medical queries.

### Functional Gaps (Reduce quality)

7. **Keyword label detection is brittle** — misses paraphrased content, no context awareness.
8. **Hardcoded CHAPTER_MAP** — breaks for every new book. Not scalable to 50 books.
9. **Fixed top-2 reranking** — may be too few for complex multi-hop questions. No dynamic N.
10. **No caching at any level** — every query hits full pipeline even for repeated questions.
11. **No conversation memory** — system forgets everything between queries in same session.
12. **No query routing** — simple definitional questions go through full expensive pipeline.
13. **Single granularity chunking** — 250 token chunks only. No multi-level indexing.
14. **NLTK punkt splits on medical abbreviations** — "Dr.", "M. tuberculosis", "Fig." cause wrong sentence boundaries.
15. **No HyDE** — complex medical queries retrieve poorly without hypothetical answer expansion.

### Production Gaps (Missing entirely)

16. **No observability** — no tracing, no latency tracking, no cost monitoring.
17. **No evaluation pipeline** — no RAGAS metrics, no faithfulness scoring, no quality tracking.
18. **No feedback loop** — user thumbs up/down not collected, no eval dataset being built.
19. **No output guardrails** — hallucinations not detected before reaching user.
20. **No multi-hop retrieval** — cannot connect information across chapters or books.
21. **No persistent document store** — raw chunks not stored separately from embeddings.
22. **No prompt versioning** — cannot track which prompt template produces better answers.

---

## Target Pipeline — Full Production Architecture

```
═══════════════════════════════════════════════════
LAYER 0 — INGESTION (offline, runs once per book)
═══════════════════════════════════════════════════

PDF
  → Docling (primary parser)
      handles: tables → markdown, diagrams → caption+context,
               two-column layouts, equations, figure captions
  → PyMuPDF (image extraction fallback)
  → Multi-level chunking
      Level 1: sentence chunks     ~80 tokens  (precise fact retrieval)
      Level 2: paragraph chunks   ~250 tokens  (topic matching)    ← current
      Level 3: section chunks    ~1000 tokens  (context-heavy questions)
  → PySBD sentence splitter (replaces NLTK punkt)
      medical abbreviation aware: Dr., Fig., M. tuberculosis, i.e., e.g.
  → Two-pass metadata tagging
      Pass 1 — structural (zero LLM cost, automated):
        book_id, book_title, chapter_number, chapter_title,
        page_number, section_heading, content_type
        (text / table / figure_caption / equation),
        chunk_level (sentence / paragraph / section),
        chunk_position_in_chapter
      Pass 2 — semantic (LLM via Groq, runs once at ingestion):
        primary_topic, body_system,
        content_categories: [@definition @anatomy @symptoms
          @diagnosis @treatment @investigation @pharmacology
          @physiology @pathology],
        clinical_relevance: high/medium/low,
        requires_prior_knowledge: [list of prerequisite topics]
  → MedCPT Article Encoder (replaces S-PubMedBert)
      asymmetric encoding: separate query encoder + article encoder
  → Qdrant Cloud (persistent, per-book collections)
  → BM25 index stored alongside

═══════════════════════════════════════════════════
LAYER 1 — QUERY ENTRY
═══════════════════════════════════════════════════

User query
  → Input guardrail (is this a medical question?)
  → Level 1: Exact match cache check (Redis)
      hit → return instantly, skip everything
  → Level 2: Semantic cache check
      embed query → cosine similarity > 0.92 against cached queries
      hit → return cached answer instantly
  → Query classifier
      simple (<8 words, no comparative/causal language)
        → Route A: direct LLM answer, skip RAG
      complex
        → Route B: full RAG pipeline

═══════════════════════════════════════════════════
LAYER 2 — QUERY PROCESSING
═══════════════════════════════════════════════════

  → Query decomposition (Groq, replaces FLAN-T5)
      break complex query into focused subquestions
      remove duplicates
      resolve pronouns and medical context
  → HyDE (selective — only for complex multi-concept queries)
      generate hypothetical answer → embed it → use for retrieval
      skip for simple single-concept queries
  → Query router
      which books/collections to search?
      single book or multi-book?

═══════════════════════════════════════════════════
LAYER 3 — RETRIEVAL
═══════════════════════════════════════════════════

For each subquery:
  → Dense retrieval (MedCPT Query Encoder)
      search across sentence + paragraph + section levels
      metadata pre-filter by book/chapter/content_type
      top 10 per level
  → Sparse retrieval (BM25)
      keyword match — critical for drug names, anatomical terms
      top 10
  → RRF fusion
      combines dense + sparse by rank position (not raw scores)
      handles different score scales correctly
  → BGE reranker
      rerank fused results
  → Dynamic N selection via token budget
      token budget = 3000 (configurable)
      take top chunks until budget exhausted
      no fixed K or N — self-regulating
  → Multi-hop check (LangGraph agent)
      answer sufficient? → proceed to generation
      not sufficient? → identify missing concepts → re-query
      supports cross-chapter and cross-book retrieval
      max 3 hops before forcing generation

═══════════════════════════════════════════════════
LAYER 4 — GENERATION
═══════════════════════════════════════════════════

  → Context builder
      compress verbose chunks (LLMLingua)
      rank chunks by relevance score
      build final context within token budget
  → Prompt builder
      versioned prompts (v1.0, v1.1, v1.2...)
      medical system prompt with strict grounding instruction
      conversation history (last 3 Q&A pairs)
  → Groq inference (Llama 3.1 8B)
      token streaming → user sees text immediately
      deterministic (temperature 0.2)
  → Output guardrail
      does answer contain information not in retrieved context?
      hallucination detected → regenerate with stricter prompt
  → Citation builder
      map answer sentences to source chunks
      attribution: Book Title → Chapter → Page number

═══════════════════════════════════════════════════
LAYER 5 — STORAGE
═══════════════════════════════════════════════════

  Qdrant Cloud     → vector embeddings (persistent, per-book collections)
  Document store   → raw chunks with full metadata (S3 or disk)
  Redis (Upstash)  → exact match cache + semantic cache
  Session store    → conversation memory (last 3 turns, cleared on session end)

═══════════════════════════════════════════════════
LAYER 6 — OBSERVABILITY
═══════════════════════════════════════════════════

  → Langfuse tracing
      every LLM call logged: latency, tokens, cost, prompt version
  → RAGAS eval pipeline (async, does not block response)
      faithfulness score (answer grounded in context?)
      answer relevancy score (does it address the question?)
      context precision (were retrieved chunks useful?)
  → Metrics dashboard
      p50/p95 latency, cost per query, cache hit rate,
      RAGAS scores over time, failure query log
  → Feedback loop
      thumbs up/down from users
      negative feedback → flagged for manual review
      all interactions → growing eval dataset
```

---

## All Possible Upgrades — Numbered

### Parsing & Ingestion

1. **Docling for PDF parsing** — tables, diagrams, equations, two-column layouts
2. **PySBD sentence splitter** — medical abbreviation aware, replaces NLTK punkt
3. **Multi-level chunking** — sentence (80t) + paragraph (250t) + section (1000t)
4. **LLM-based metadata tagging at ingestion** — context-aware, replaces keyword rules
5. **Automatic TOC parsing for chapter mapping** — works for any new book automatically
6. **Figure caption extraction** — link diagram captions to surrounding text context
7. **Table-to-text conversion** — convert markdown tables to retrievable natural language

### Embeddings

8. **MedCPT asymmetric encoders** — separate query + article encoders, best for medical QA retrieval
9. **Fine-tuned embeddings on real query data** — after 500+ real queries, fine-tune on (query, chunk, negative) triplets
10. **BGE-M3 as alternative** — supports hybrid dense+sparse in single model

### Retrieval

11. **HyDE** — hypothetical document embeddings for complex queries
12. **Dynamic N via token budgeting** — replaces fixed top-K selection
13. **Multi-hop LangGraph agent** — cross-chapter, cross-book retrieval with planning
14. **Multi-level simultaneous search** — query all chunk sizes, reranker selects best granularity
15. **Metadata pre-filtering** — filter by book/chapter/content_type before vector search
16. **Contextual compression (LLMLingua)** — trim chunks to relevant sentences only

### Query Processing

17. **Groq-based query decomposition** — replaces weak FLAN-T5
18. **Query classifier** — route simple vs complex queries differently
19. **Query router** — decide which book collections to search per query
20. **Conversation-aware query rewriting** — use session history to resolve pronouns and context

### Caching

21. **Exact match cache (Redis)** — instant return for identical repeated queries
22. **Semantic cache** — cosine similarity threshold on query embeddings
23. **Book-level popular questions cache** — pre-cache most asked questions per book
24. **Chapter-level summary cache** — pre-generate chapter summaries for overview questions

### Generation

25. **Groq inference** — 800+ tokens/second vs 20 tokens/second local
26. **Token streaming** — real-time word-by-word display
27. **Prompt versioning** — track which prompt version gives better RAGAS scores
28. **Output guardrails** — hallucination detection before answer reaches user
29. **Citation builder** — book + chapter + page attribution in every answer
30. **Medical-specific system prompt** — strict grounding, preserve medical terminology
31. **Together AI OpenBioLLM-70B** — larger medical-specific model via API if quality needs boost

### Evaluation

32. **RAGAS eval pipeline** — faithfulness, answer relevancy, context precision, context recall
33. **Golden dataset from LLM-generated QA pairs** — auto-generate from your own chunks
34. **MBBS exam question evaluation** — use past university papers as ground truth
35. **Domain expert evaluation** — structured feedback collection from MBBS student users
36. **A/B testing prompt versions** — split traffic between prompt versions, compare RAGAS scores

### Observability

37. **Langfuse tracing** — every call logged with latency, cost, prompt version
38. **Metrics dashboard** — p95 latency, cache hit rate, RAGAS over time
39. **Failure query log** — auto-flag queries where system said "I don't know"
40. **User feedback loop** — thumbs up/down → growing eval dataset

### Scale

41. **Per-book Qdrant collections** — clean separation, easy add/remove books
42. **Global collection index** — tracks which collection contains which book
43. **Persistent document store** — raw chunks in S3 separate from embeddings
44. **Session-based conversation memory** — last 3 turns, cleared on session end
45. **Multi-user session isolation** — separate memory and context per user

### Future / Advanced

46. **Fine-tuned generation model** — fine-tune Llama 3.1 8B on medical QA pairs via LoRA
47. **Multimodal retrieval** — embed diagram images alongside text for visual anatomy questions
48. **Confidence scoring** — express uncertainty when retrieval confidence is low
49. **Cross-lingual support** — Hindi medical query handling for broader student base
50. **Personalization** — track per-student weak topics, surface relevant content proactively

---

## Version Roadmap

---

### Version 1.0 — Ship It (Target: 1 week)
**Goal: Live product with real users**

Fixes from current notebook:
- Replace local Zephyr-7B with **Groq API** (upgrade #25)
- Add **token streaming** (upgrade #26)
- Migrate Qdrant from in-memory to **Qdrant Cloud** persistent (upgrade #41)
- Build **Streamlit UI** with text input, answer display, source display
- Deploy on **HuggingFace Spaces**
- Add basic **1-10 rating feedback** scale (upgrade #40)

Pipeline additions:
- Basic **citation** showing book + chapter in answer (upgrade #29)
- **Session conversation memory** — last 3 turns (upgrade #44)

What stays the same:
- PyMuPDF parsing (good enough for v1)
- S-PubMedBert embeddings (good enough for v1)
- Existing hybrid search + RRF + BGE reranker
- Existing FLAN-T5 decomposition (imperfect but functional)

**Outcome: Sister and friends can use it on phone. Real feedback starts flowing.**

---

### Version 1.5 — Quality Pass (Target: 2-3 weeks after v1 launch)
**Goal: Fix what real users actually complain about**

Based on real usage data from v1:

- Replace FLAN-T5 with **Groq query decomposition** (upgrade #17)
- Add **exact match cache** via Redis/Upstash (upgrade #21)
- Add **semantic cache** with cosine threshold (upgrade #22)
- Add **query classifier** — skip RAG for simple definitional questions (upgrade #18)
- Add **RAGAS eval pipeline** running async (upgrade #32)
- Add **Langfuse tracing** (upgrade #37)
- Fix **PySBD sentence splitter** replacing NLTK punkt (upgrade #2)
- Add **output guardrail** — basic hallucination check (upgrade #28)
- Add **prompt versioning** (upgrade #27)

**Outcome: Noticeably faster, measurably better quality, you can track what's working.**

---

### Version 2.0 — Production Grade (Target: 1-2 months after v1)
**Goal: Handle 50 books, proper evaluation, interview-worthy architecture**

Major additions:

- **Docling PDF parsing** — tables and diagrams now visible (upgrade #1)
- **Multi-level chunking** — sentence + paragraph + section (upgrade #3)
- **LLM metadata tagging at ingestion** — replaces keyword rules (upgrade #4)
- **Automatic TOC chapter mapping** — works for any book (upgrade #5)
- **MedCPT embeddings** — replace S-PubMedBert (upgrade #8)
- **HyDE for complex queries** — selective, only when needed (upgrade #11)
- **Dynamic N via token budget** — replaces fixed top-2 (upgrade #12)
- **LLMLingua context compression** (upgrade #16)
- **Per-book Qdrant collections** — scale architecture (upgrade #41)
- **Global collection index** (upgrade #42)
- **Persistent document store** (upgrade #43)
- **Golden dataset** from LLM-generated QA pairs (upgrade #33)
- **MBBS exam question evaluation** (upgrade #34)
- **Metrics dashboard** (upgrade #38)

**Outcome: Handles 50 books cleanly. Full eval pipeline. Strong system design interview story.**

---

### Version 3.0 — Research Grade (Target: post-placement)
**Goal: Novel contributions, publishable improvements**

Advanced additions:

- **Multi-hop LangGraph agent** — cross-book, cross-chapter planning (upgrade #13)
- **Fine-tuned MedCPT embeddings** on real query data from sister's friends (upgrade #9)
- **Fine-tuned Llama 3.1 8B** via LoRA on medical QA pairs (upgrade #46)
- **A/B testing prompt versions** with RAGAS as judge (upgrade #36)
- **Multimodal retrieval** — anatomy diagrams embedded alongside text (upgrade #47)
- **Confidence scoring** — express uncertainty explicitly (upgrade #48)
- **Personalization** — per-student weak topic tracking (upgrade #50)
- **Cross-lingual Hindi support** (upgrade #49)

**Outcome: Novel research contributions. Publishable. Far beyond any student project.**

---

## Interview Framing — How To Talk About This

### The Novel Contribution (say this confidently)

> "Most RAG projects use pre-cleaned datasets with ready-made QA pairs. QAMed uses real MBBS textbooks — messy PDFs with tables, equations, two-column layouts, medical abbreviations. Because no off-the-shelf eval dataset existed for our corpus, we built a custom golden evaluation set using LLM-generated QA pairs validated by actual MBBS students. We tracked faithfulness and answer relevancy using RAGAS alongside domain expert human evaluation. This is closer to how production RAG systems are actually evaluated."

### The Architecture Story (for system design rounds)

> "We built a six-layer production RAG architecture — ingestion, storage, query processing, retrieval, generation, and observability. Key design decisions: asymmetric medical encoders (MedCPT) for query-document matching, multi-level chunking to let the reranker self-select optimal granularity, RRF fusion for combining dense and sparse retrieval without score scale mismatch, and a LangGraph multi-hop agent for cross-book questions. The system scales from 1 to 50 books via per-collection Qdrant architecture with a global index."

### The Impact Story (for behavioural rounds)

> "After deploying on HuggingFace Spaces, 10+ MBBS students at [college] used it during exam preparation. Their real queries became our evaluation dataset — we discovered failure patterns we never anticipated from synthetic testing, which directly drove Version 2 improvements."

---

*Last updated: March 2026*
*Project: QAMed — Medical RAG System for MBBS Textbooks*
