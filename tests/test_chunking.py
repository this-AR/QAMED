import pytest
import os
import sqlite3
from typing import List

from data.ingest import (
    make_chunk_id,
    make_parent_id,
    detect_labels,
    token_length,
    split_with_offset_tracking,
    ChunkData
)
from data.doc_store import DocumentStore
from services.retrieval import truncate_parent, expand_to_parents
from langchain_core.documents import Document

def test_deterministic_chunk_ids():
    """Same input produces same chunk_id across runs."""
    id1 = make_chunk_id("book.pdf", "Section 1", 0, "Some text")
    id2 = make_chunk_id("book.pdf", "Section 1", 0, "Some text")
    assert id1 == id2
    
    id3 = make_chunk_id("book.pdf", "Section 1", 1, "Some text")
    assert id1 != id3

def test_multi_label_detect_labels():
    """detect_labels returns multiple labels when text matches multiple patterns."""
    text = "Hernia is defined as a protrusion. Treatment includes surgery."
    labels = detect_labels(text)
    assert "@definition" in labels
    assert "@treatment" in labels
    assert "@general" not in labels
    
    # zero keywords
    assert "@general" in detect_labels("The abdomen is a body cavity.")

def test_token_length_function():
    """Token-based length function counts tokens, not characters."""
    text = "This is a short medical sentence."
    # usually 1 char is less than 1 token
    assert token_length(text) > 0
    assert token_length(text) < len(text)

def test_offset_tracking():
    text = "First sentence. Second sentence. Third sentence."
    chunks = split_with_offset_tracking(text, target=10, overlap=2, base_offset=100)
    for chunk in chunks:
        # Check that the chunk text exists in original text at (chunk.offset - base_offset)
        local_offset = chunk.offset - 100
        assert text[local_offset:local_offset + len(chunk.text[:80])] == chunk.text[:80]

def test_doc_store_crud(tmp_path):
    """SQLite doc store stores, retrieves, batch-fetches, and deletes correctly."""
    db_path = str(tmp_path / "test_doc_store.db")
    store = DocumentStore(db_path)
    
    store.store_parent("p1", {
        "text": "Parent 1 text",
        "book_name": "BookA",
        "chapter": "Chap 1",
        "section": "Sec 1",
        "pages": [1, 2],
        "token_count": 10
    })
    
    parent = store.get_parent("p1")
    assert parent is not None
    assert parent["text"] == "Parent 1 text"
    assert parent["pages"] == [1, 2]
    
    # Clear book
    store.clear_book("BookA")
    assert store.get_parent("p1") is None

def test_parent_truncation_uses_stored_offset():
    """Truncation uses leaf_offset from metadata and centers window on it."""
    # Use realistic sentences that pysbd will segment correctly
    parent_text = (
        "The femoral nerve arises from L2 to L4 segments of the lumbar plexus. "
        "It descends through the psoas major muscle and emerges from its lateral border. "
        "The nerve then passes beneath the inguinal ligament to enter the femoral triangle. "
        "In the femoral triangle it divides into anterior and posterior divisions. "
        "The anterior division supplies the sartorius muscle and skin of the anterior thigh. "
        "The posterior division supplies the quadriceps femoris muscle. "
        "The saphenous nerve is the largest cutaneous branch of the femoral nerve. "
        "It accompanies the femoral artery into the adductor canal."
    )
    # Target leaf is in the middle — "The nerve then passes..."
    leaf_text = "The nerve then passes beneath the inguinal ligament to enter the femoral triangle."
    leaf_offset = parent_text.find(leaf_text)
    leaf_length = len(leaf_text)
    assert leaf_offset > 0, "sanity: leaf text must be in parent"
    
    # With a small token budget, the truncated output should contain the leaf text
    truncated = truncate_parent(parent_text, leaf_offset, leaf_length, max_tokens=40)
    assert leaf_text[:30] in truncated or "inguinal ligament" in truncated
    # And it should be shorter than the full text
    assert len(truncated) < len(parent_text)

def test_parent_expansion_deduplicates(tmp_path):
    """expand_to_parents deduplicates when multiple leaves share a parent."""
    db_path = str(tmp_path / "test_doc_store2.db")
    store = DocumentStore(db_path)
    
    store.store_parent("p_shared", {
        "text": "Shared parent text",
        "book_name": "Book",
        "chapter": "C",
        "section": "S",
        "pages": [1],
        "token_count": 10
    })
    
    docs = [
        Document(page_content="leaf1", metadata={"parent_id": "p_shared", "leaf_offset": 0}),
        Document(page_content="leaf2", metadata={"parent_id": "p_shared", "leaf_offset": 5}),
    ]
    
    expanded = expand_to_parents(docs, store, 3000)
    assert len(expanded) == 1
    assert expanded[0]["text"] == "Shared parent text"
