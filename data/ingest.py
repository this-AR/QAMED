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

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PARAGRAPH_CHUNK_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    DOC_STORE_PATH
)
from data.doc_store import DocumentStore

SENTENCE_SEGMENTER = pysbd.Segmenter(language="en", clean=False)

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
        
        if content.startswith("# ") or label == "title":
            chapter = content.lstrip("# ").strip()
        elif content.startswith("## ") or label == "section_header":
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
        
        # Store parent
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
                    "book_page": pdf_page,  # extracted from page
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
                    chunk_id = make_chunk_id(book_name, sec_data['section'], chunk_index, pc.text)
                    chunk_metadata = {
                        "chunk_id": chunk_id,
                        "parent_id": parent_id,
                        "leaf_offset": pc.offset,
                        "book_name": book_name,
                        "chapter": sec_data['chapter'],
                        "section": sec_data['section'],
                        "chunk_level": "paragraph",
                        "pdf_page": pdf_page,
                        "book_page": pdf_page,
                        "labels": detect_labels(pc.text),
                        "content_type": content_type,
                    }
                    chunked_docs.append(Document(page_content=pc.text, metadata=chunk_metadata))
                    chunk_index += 1

    return chunked_docs

def ensure_collection(client, collection_name, vector_size):
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

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
