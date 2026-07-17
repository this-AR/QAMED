"""
QAMed — PDF Ingestion Pipeline (Hierarchical v2)

Loads a medical PDF, extracts TOC and structure using Docling,
chunks at the paragraph level (tracking leaf offset), stores parent
sections in SQLite, and uploads paragraph chunks to Qdrant Cloud.

Usage:
    python -m data.ingest --pdf path/to/book.pdf [--clean] [--force]
"""

import argparse
import os
import uuid
import hashlib
import json
import re
import concurrent.futures
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from groq import Groq

import pysbd
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_docling.loader import DoclingLoader, ExportType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, FilterSelector, Filter, FieldCondition, MatchValue
from langchain_core.documents import Document

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PARAGRAPH_CHUNK_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    DOC_STORE_PATH,
)
from data.doc_store import DocumentStore

SENTENCE_SEGMENTER = pysbd.Segmenter(language="en", clean=False)

# 🔥 Page offset (VERY IMPORTANT)
PAGE_OFFSET = 15  # Book page 1 = PDF page 16

# Chapter mapping (PDF pages)
CHAPTER_MAP = {
    (16, 24): "Introduction and Overview of the Abdomen",
    (25, 38): "Osteology of the Abdomen",
    (39, 60): "Anterior Abdominal Wall",
    (61, 73): "Inguinal Region/Groin",
    (74, 88): "Male External Genital Organs",
    (89, 107): "Abdominal Cavity and Peritoneum",
    (108, 123): "Abdominal Part of Esophagus, Stomach, and Spleen",
    (124, 140): "Liver and Extrahepatic Biliary Apparatus",
    (141, 158): "Duodenum, Pancreas, and Portal Vein",
    (159, 179): "Small and Large Intestines",
    (180, 199): "Kidneys, Ureters, and Suprarenal Glands",
    (200, 216): "Posterior Abdominal Wall and Associated Structures",
    (217, 226): "Pelvis",
    (227, 239): "Pelvic Walls and Associated Soft Tissue Structures",
    (240, 252): "Perineum",
    (253, 265): "Urinary Bladder and Urethra",
    (266, 274): "Male Genital Organs",
    (275, 293): "Female Genital Organs",
    (294, 305): "Rectum and Anal Canal",
    (306, 313): "Introduction to the Lower Limb",
    (314, 342): "Bones of the Lower Limb",
    (343, 358): "Front of the Thigh",
    (359, 367): "Medial Side of the Thigh",
    (368, 378): "Gluteal Region",
    (379, 391): "Back of the Thigh and Popliteal Fossa",
    (392, 400): "Hip Joint",
    (401, 414): "Front of the Leg and Dorsum of the Foot",
    (415, 421): "Lateral and Medial Sides of the Leg",
    (422, 434): "Back of the Leg",
    (435, 446): "Sole of the Foot",
    (447, 453): "Arches of the Foot",
    (454, 472): "Joints of the Lower Limb",
    (473, 481): "Venous and Lymphatic Drainage of the Lower Limb",
    (482, 493): "Innervation of the Lower Limb",
}


def get_chapter_by_page(pdf_page):
    for (start, end), title in CHAPTER_MAP.items():
        if start <= pdf_page <= end:
            return title
    return "Unknown Chapter"

# ── Contextual Micro-Headers (v2.5 — Multi-Provider) ─────────────────────────
import threading
import time as _time
import google.generativeai as genai

from config import CONTEXT_API_KEYS, CONTEXT_WORKERS

CONTEXT_PROMPT_TEMPLATE = (
    "You are an expert medical librarian. Below is a section from a medical textbook, "
    "followed by a specific chunk extracted from that section.\n\n"
    "<document>\n{parent_text}\n</document>\n\n"
    "<chunk>\n{chunk_text}\n</chunk>\n\n"
    "Analyze the chunk in the context of the document and extract metadata as a JSON object.\n"
    "Return ONLY a valid JSON object matching this exact schema:\n"
    "{{\n"
    '  "summary": "Concise 1-2 sentence situating summary providing anatomical/clinical context for the chunk so it can be understood in isolation.",\n'
    '  "breadcrumbs": "The full hierarchical path, e.g. [Abdomen > Anterior Abdominal Wall > Inguinal Canal].",\n'
    '  "semantic_label": "One of: @definition, @procedure, @clinical_correlation, @embryology, @anatomy_structure, @table.",\n'
    '  "clinical_category": "The broad clinical topic, e.g. surgical pathology, diagnostic imaging, pharmacotherapy, regional anatomy.",\n'
    '  "prerequisite_concepts": ["List", "of", "foundational", "concepts", "needed", "to", "understand", "this", "chunk"],\n'
    '  "demographic": "Target demographic, e.g. neonatal, pediatric, adult_general."\n'
    "}}\n"
)


class APIKeyRotator:
    """Round-robin across multiple LLM API keys with RPM-aware delays."""

    # RPM limits per provider (requests per minute)
    RPM = {"groq": 30, "gemini": 15}

    def __init__(self, keys_csv: str):
        self.providers = []
        for entry in keys_csv.split(","):
            entry = entry.strip()
            if not entry or ":" not in entry:
                continue
            provider, key = entry.split(":", 1)
            self.providers.append((provider.lower().strip(), key.strip()))
        self.index = 0
        self.lock = threading.Lock()
        # Per-key request timestamps for RPM tracking
        self._timestamps: dict[int, list[float]] = {i: [] for i in range(len(self.providers))}

    def _wait_for_rpm(self, key_idx: int) -> None:
        """Block if this key has hit its RPM limit in the last 60 seconds."""
        provider = self.providers[key_idx][0]
        rpm_limit = self.RPM.get(provider, 30)
        now = _time.time()
        # Prune timestamps older than 60s
        self._timestamps[key_idx] = [t for t in self._timestamps[key_idx] if now - t < 60]
        if len(self._timestamps[key_idx]) >= rpm_limit:
            wait = 60 - (now - self._timestamps[key_idx][0]) + 0.5
            if wait > 0:
                _time.sleep(wait)
        self._timestamps[key_idx].append(_time.time())

    def _next_key(self) -> tuple[int, str, str]:
        with self.lock:
            idx = self.index % len(self.providers)
            self.index += 1
        provider, key = self.providers[idx]
        return idx, provider, key

    def smoke_test(self) -> list[tuple[str, str, bool]]:
        """Test each key with a trivial request. Returns [(provider, key_suffix, ok)]."""
        results = []
        for i, (provider, key) in enumerate(self.providers):
            key_suffix = f"...{key[-6:]}"
            try:
                if provider == "groq":
                    client = Groq(api_key=key)
                    client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": "Say OK"}],
                        max_tokens=5,
                    )
                elif provider == "gemini":
                    genai.configure(api_key=key)
                    model = genai.GenerativeModel("gemini-3.1-flash-lite")
                    model.generate_content("Say OK")
                results.append((provider, key_suffix, True))
                print(f"  [OK] {provider} key {key_suffix} - OK")
            except Exception as e:
                results.append((provider, key_suffix, False))
                print(f"  [FAIL] {provider} key {key_suffix} - FAILED: {e}")
        return results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def _call_groq(self, key: str, prompt: str) -> str:
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def _call_gemini(self, key: str, prompt: str) -> str:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-3.1-flash-lite", generation_config={"response_mime_type": "application/json"})
        resp = model.generate_content(prompt)
        return resp.text.strip()

    def generate(self, prompt: str) -> str:
        """Try providers in round-robin until one succeeds."""
        attempts = len(self.providers)
        for _ in range(attempts):
            idx, provider, key = self._next_key()
            try:
                self._wait_for_rpm(idx)
                if provider == "groq":
                    return self._call_groq(key, prompt)
                elif provider == "gemini":
                    return self._call_gemini(key, prompt)
            except Exception:
                continue  # Skip to next key on any error
        return ""  # All keys exhausted — chunk stays unenriched


# Global rotator instance (initialized lazily)
_rotator: APIKeyRotator | None = None

def get_rotator() -> APIKeyRotator | None:
    global _rotator
    if _rotator is None and CONTEXT_API_KEYS:
        _rotator = APIKeyRotator(CONTEXT_API_KEYS)
    return _rotator


import json

def generate_chunk_context(parent_text: str, chunk_text: str) -> dict:
    """Generate a situating summary and metadata for a chunk using multi-key rotation."""
    rotator = get_rotator()
    if not rotator:
        return {}
    prompt = CONTEXT_PROMPT_TEMPLATE.format(
        parent_text=parent_text[:3000],  # Truncate parent to avoid token limits
        chunk_text=chunk_text
    )
    raw_response = rotator.generate(prompt)
    if not raw_response:
        return {}
    
    try:
        # Strip potential markdown formatting if the model still wrapped it
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        print(f"Failed to parse JSON context for chunk. Raw: {raw_response[:100]}")
        return {}


# ── Coreference Resolution ────────────────────────────────────────────────────
def extract_subject_from_heading(heading: str) -> str:
    """Extract the anatomical subject from a section heading.
    
    e.g., 'The Femoral Nerve — Course and Relations' → 'The Femoral Nerve'
    e.g., 'Inguinal Canal' → 'The Inguinal Canal'
    """
    if not heading:
        return ""
    # Remove common suffixes
    for sep in ["—", "–", "-", ":"]:
        heading = heading.split(sep)[0].strip()
    # Remove markdown
    heading = heading.lstrip("# ").strip()
    # Skip overly generic headings
    if len(heading.split()) > 6 or len(heading) < 4:
        return ""
    if heading.lower() in ("introduction", "summary", "clinical anatomy", "review questions"):
        return ""
    # Add "The" if missing for natural text flow
    if not heading.lower().startswith("the "):
        heading = f"The {heading}"
    return heading


def resolve_coreferences(text: str, section_heading: str) -> str:
    """Replace common anatomical dangling pronouns with the section subject."""
    subject = extract_subject_from_heading(section_heading)
    if not subject:
        return text
    
    patterns = [
        # "It passes/runs/crosses..." → "The Femoral Nerve passes/runs/crosses..."
        (r'\bIt (is |passes |runs |crosses |supplies |innervates |enters |exits |pierces |arises |descends |ascends )',
         f'{subject} \\1'),
        # "This structure/organ/muscle..." → "The Femoral Nerve"
        (r'\bThis (structure|organ|muscle|nerve|vessel|canal|ligament|artery|vein|foramen) ',
         f'{subject} '),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, count=3, flags=re.IGNORECASE)
    return text


# ── Table Header Repetition ───────────────────────────────────────────────────
def ensure_table_headers(chunk_text: str, parent_text: str) -> str:
    """If a chunk starts mid-table (no header separator), prepend the table's header rows."""
    lines = chunk_text.strip().split("\n")
    if not lines or not lines[0].strip().startswith("|"):
        return chunk_text  # Not a table chunk
    
    # Check if chunk already has a header separator (e.g., |---|---|)
    for line in lines[:3]:
        if re.search(r'\|[-:]+\|', line):
            return chunk_text  # Already has headers
    
    # Find the nearest table header in the parent text before this chunk's content
    chunk_start = parent_text.find(lines[0][:60])
    if chunk_start <= 0:
        return chunk_text
    
    # Search backward from chunk_start for a header separator line
    preceding = parent_text[:chunk_start]
    preceding_lines = preceding.split("\n")
    
    header_lines = []
    for i in range(len(preceding_lines) - 1, -1, -1):
        line = preceding_lines[i].strip()
        if re.search(r'\|[-:]+\|', line):
            # Found separator — grab it and the line above (column names)
            header_lines = []
            if i > 0:
                header_lines.append(preceding_lines[i - 1])
            header_lines.append(preceding_lines[i])
            break
    
    if header_lines:
        return "\n".join(header_lines) + "\n" + chunk_text
    return chunk_text



print(f"Loading Tokenizer {EMBEDDING_MODEL_NAME}...")
try:
    TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"Warning: Could not load tokenizer ({e}). Falling back to gpt2.")
    TOKENIZER = AutoTokenizer.from_pretrained("gpt2")

def token_length(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def make_chunk_id(book_name: str, section: str, chunk_index: int, text: str) -> str:
    raw = f"{book_name}|{section}|{chunk_index}|{hashlib.sha256(text.encode()).hexdigest()[:8]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def make_parent_id(book_name: str, section_heading: str) -> str:
    raw = f"{book_name}|{section_heading}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def detect_labels(text: str) -> list:
    labels = []
    t = text.lower()

    if "is defined as" in t or "refers to" in t or "means" in t:
        labels.append("@definition")
    if "symptoms include" in t or "signs are" in t or "manifestations" in t:
        labels.append("@symptoms")
    if "diagnosis is based on" in t or "diagnosed by" in t or "investigations include" in t:
        labels.append("@diagnosis")
    if "treatment includes" in t or "managed by" in t or "therapy" in t:
        labels.append("@treatment")
    if "relations include" in t or "borders are" in t:
        labels.append("@anatomy_structure")
    if "supplied by" in t or "innervated by" in t:
        labels.append("@supply")

    return labels if labels else ["@general"]


def segment_text_with_pysbd(text: str) -> str:
    if not text:
        return ""
    sentences = SENTENCE_SEGMENTER.segment(text)
    return "\n".join(s.strip() for s in sentences if s and s.strip())

class ChunkData:
    def __init__(self, text, offset):
        self.text = text
        self.offset = offset

def split_with_offset_tracking(text: str, target: int, overlap: int, base_offset: int) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target,
        chunk_overlap=overlap,
        length_function=token_length,
    )
    pieces = splitter.split_text(text)
    
    chunks = []
    search_start = 0
    for piece in pieces:
        # Match piece exactly to find its offset within the text
        # Using a prefix search to be robust against slight splitter normalizations
        prefix = piece[:80]
        idx = text.find(prefix, search_start)
        if idx == -1:
            idx = search_start  # fallback
        
        chunks.append(ChunkData(piece, base_offset + idx))
        # advance search start, allowing for overlap
        search_start = max(0, idx + len(piece) - (overlap * 5))
        
    return chunks

def extract_pages(doc) -> set:
    metadata = doc.metadata
    dl_meta = metadata.get("dl_meta", {})
    doc_items = dl_meta.get("doc_items", [])
    pages = set()
    for item in doc_items:
        prov_list = item.get("prov", [])
        for prov in prov_list:
            p = prov.get("page_no")
            if p is not None:
                pages.add(p)
    return pages

def extract_book_page(doc) -> int:
    # A robust extraction would use regex on header/footer, but for now we fallback
    # to pdf_page if not implemented, or use the first page found in docling prov.
    pages = extract_pages(doc)
    if pages:
        return sorted(list(pages))[0]
    return doc.metadata.get("page", 1)

def hierarchical_chunk(docs, book_name: str, doc_store: DocumentStore) -> list:
    chapter = "Unknown Chapter"
    section = "Unknown Section"
    
    # Group elements into sections
    sections_dict = {}
    
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue
            
        label = doc.metadata.get("dl_meta", {}).get("doc_items", [{}])[0].get("label", "")
        
        pdf_page = extract_book_page(doc)
        chapter = get_chapter_by_page(pdf_page)
        
        if content.startswith("## ") or label == "section_header":
            section = content.lstrip("# ").strip()
            
        parent_id = make_parent_id(book_name, section)
        if parent_id not in sections_dict:
            sections_dict[parent_id] = {
                'chapter': chapter,
                'section': section,
                'pages': set(),
                'elements': [],
                'text_blocks': []
            }
            
        sections_dict[parent_id]['elements'].append(doc)
        sections_dict[parent_id]['text_blocks'].append(content)
        sections_dict[parent_id]['pages'].update(extract_pages(doc))
        
    chunked_docs = []
    
    # Process each section
    for parent_id, sec_data in sections_dict.items():
        section_text = "\n\n".join(sec_data['text_blocks'])
        
        # Apply coreference resolution on the full section before chunking
        section_text = resolve_coreferences(section_text, sec_data['section'])
        
        # Store parent (with resolved coreferences)
        doc_store.store_parent(parent_id, {
            "text": section_text,
            "chapter": sec_data['chapter'],
            "section": sec_data['section'],
            "book_name": book_name,
            "pages": sorted(list(sec_data['pages'])),
            "token_count": token_length(section_text)
        })
        
        # Generate leaf chunks
        chunk_index = 0
        for doc in sec_data['elements']:
            content = doc.page_content.strip()
            # Apply coreference resolution on element text too
            content = resolve_coreferences(content, sec_data['section'])
            label = doc.metadata.get("dl_meta", {}).get("doc_items", [{}])[0].get("label", "")
            
            # Find offset of this element within the section text
            element_offset = section_text.find(content[:80])
            if element_offset == -1: element_offset = 0
            
            content_type = "text"
            is_table = "table" in label or "|" in content[:20]  # simple table detection if label fails
            if is_table:
                content_type = "table"
            elif "figure" in label or "caption" in label:
                content_type = "figure_caption"
                
            pdf_page = extract_book_page(doc)
            book_page = pdf_page - PAGE_OFFSET
            
            if content_type == "table":
                # Tables are indexed as ONE chunk, never split
                chunk_id = make_chunk_id(book_name, sec_data['section'], chunk_index, content)
                chunk_metadata = {
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "leaf_offset": element_offset,
                    "book_name": book_name,
                    "chapter": sec_data['chapter'],
                    "section": sec_data['section'],
                    "chunk_level": "paragraph",
                    "pdf_page": pdf_page,
                    "book_page": book_page,
                    "labels": detect_labels(content),
                    "content_type": content_type,
                }
                chunked_docs.append(Document(page_content=content, metadata=chunk_metadata))
                chunk_index += 1
            else:
                # Normal text: paragraph-level split with offset tracking
                segmented = segment_text_with_pysbd(content)
                para_chunks = split_with_offset_tracking(
                    segmented, 
                    target=PARAGRAPH_CHUNK_TOKENS, 
                    overlap=CHUNK_OVERLAP_TOKENS, 
                    base_offset=element_offset
                )
                
                for pc in para_chunks:
                    # Apply table header repetition if chunk starts mid-table
                    pc_text = ensure_table_headers(pc.text, section_text)
                    chunk_id = make_chunk_id(book_name, sec_data['section'], chunk_index, pc_text)
                    chunk_metadata = {
                        "chunk_id": chunk_id,
                        "parent_id": parent_id,
                        "leaf_offset": pc.offset,
                        "book_name": book_name,
                        "chapter": sec_data['chapter'],
                        "section": sec_data['section'],
                        "chunk_level": "paragraph",
                        "pdf_page": pdf_page,
                        "book_page": book_page,
                        "labels": detect_labels(pc_text),
                        "content_type": content_type,
                    }
                    chunked_docs.append(Document(page_content=pc_text, metadata=chunk_metadata))
                    chunk_index += 1

    # ── Contextual Enrichment (v2.5 — Multi-Provider) ─────────────────────
    rotator = get_rotator()
    if rotator and rotator.providers:
        print(f"\nSmoke-testing {len(rotator.providers)} API keys...")
        smoke_results = rotator.smoke_test()
        working_keys = sum(1 for _, _, ok in smoke_results if ok)
        if working_keys == 0:
            print("⚠️  No working API keys — skipping contextual enrichment.")
            return chunked_docs
        
        print(f"\n{working_keys}/{len(rotator.providers)} keys working. "
              f"Generating contextual micro-headers for {len(chunked_docs)} chunks...\n")
        
        parent_texts = {pid: "\n\n".join(sec['text_blocks']) for pid, sec in sections_dict.items()}
        
        def enrich_chunk(doc: Document) -> Document:
            # Tables don't need semantic contextualization, skip to save tokens
            if doc.metadata.get("content_type") == "table":
                return doc
                
            parent_id = doc.metadata.get("parent_id")
            parent_text = parent_texts.get(parent_id, "")
            if parent_text and len(doc.page_content.strip()) > 50:
                try:
                    context = generate_chunk_context(parent_text, doc.page_content)
                    if context and isinstance(context, dict):
                        breadcrumbs = context.get("breadcrumbs", "")
                        summary = context.get("summary", "")
                        
                        # Prepend hierarchical breadcrumbs and summary to content
                        prefix = ""
                        if breadcrumbs:
                            prefix += f"{breadcrumbs}\n"
                        if summary:
                            prefix += f"Context: {summary}\n"
                            
                        if prefix:
                            doc.page_content = f"{prefix}\n{doc.page_content}"
                            
                        # Add structured medical tags to Qdrant metadata payload
                        if "semantic_label" in context:
                            doc.metadata["semantic_label"] = context["semantic_label"]
                        if "clinical_category" in context:
                            doc.metadata["clinical_category"] = context["clinical_category"]
                        if "prerequisite_concepts" in context:
                            doc.metadata["prerequisite_concepts"] = context["prerequisite_concepts"]
                        if "demographic" in context:
                            doc.metadata["demographic"] = context["demographic"]
                            
                except Exception as e:
                    pass  # Fallback to unenriched chunk on persistent failure
            return doc
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONTEXT_WORKERS) as executor:
            enriched_docs = list(tqdm(
                executor.map(enrich_chunk, chunked_docs),
                total=len(chunked_docs),
                desc="Contextualizing"
            ))
        return enriched_docs
    else:
        print("No CONTEXT_API_KEYS configured — skipping contextual enrichment.")
        return chunked_docs

def ensure_collection(client, collection_name, vector_size):
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="book_name",
            field_schema="keyword"
        )
    except Exception:
        pass

def reingest_book(qdrant_client, collection_name, book_name, doc_store):
    """Delete all existing chunks for this book before inserting fresh."""
    print(f"Deleting existing Qdrant points for book: {book_name}")
    try:
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=Filter(must=[
                    FieldCondition(key="book_name", match=MatchValue(value=book_name))
                ])
            )
        )
    except Exception as e:
        print(f"No existing points found or error during deletion: {e}")
        
    print(f"Clearing existing doc store entries for book: {book_name}")
    doc_store.clear_book(book_name)

def clean_all(qdrant_client, collection_name, doc_store):
    """Delete entire collection and clear doc store."""
    print(f"Deleting entire Qdrant collection: {collection_name}")
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception as e:
        print(f"Collection might not exist yet: {e}")
        
    print("Clearing doc store...")
    doc_store.clear()


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest a PDF into Qdrant Cloud.")
    parser.add_argument("--pdf", required=True, help="Path to the medical PDF")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--clean", action="store_true", help="Delete collection and doc store before ingestion")
    parser.add_argument("--force", action="store_true", help="Delete existing chunks for this book before insertion")
    return parser.parse_args()


def main():
    args = parse_args()

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in .env")

    print("Connecting to Qdrant...")
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    
    doc_store = DocumentStore(DOC_STORE_PATH)

    if args.clean:
        clean_all(qdrant_client, args.collection, doc_store)
        
    ensure_collection(qdrant_client, args.collection, vector_size=768)

    book_name = os.path.basename(args.pdf)
    
    if args.force:
        reingest_book(qdrant_client, args.collection, book_name, doc_store)

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Loading and extracting PDF with Docling...")
    
    # Accelerated processing for large medical PDFs
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options.num_threads = 1
    
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
    )

    loader = DoclingLoader(
        file_path=args.pdf,
        export_type=ExportType.DOC_CHUNKS,
        converter=converter
    )
    docs = loader.load()

    print("Chunking hierarchically...")
    chunks = hierarchical_chunk(docs, book_name, doc_store)

    print(f"Generated {len(chunks)} paragraph chunks. Uploading to Qdrant...")

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=args.collection,
        embedding=embedding_model,
    )
    
    # Use deterministic chunk_id for UUIDs
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, doc.metadata["chunk_id"])) for doc in chunks]
    vectorstore.add_documents(chunks, ids=ids)

    print(f"[OK] Successfully added {len(chunks)} chunks to '{args.collection}'")


if __name__ == "__main__":
    main()
