"""retrieve() pipeline and truncate_parent, using fakes (no Qdrant/models needed)."""
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from data.doc_store import DocumentStore
from services.retrieval import retrieve, truncate_parent, token_length


def _doc(cid, text):
    return Document(page_content=text, metadata={"chunk_id": cid, "parent_id": f"p_{cid}", "leaf_offset": 0})


class FakeVectorstore:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def similarity_search(self, query, k):
        self.queries.append(query)
        return self.docs[:k]


class FakeReranker:
    def predict(self, pairs):
        return [len(text) for _, text in pairs]  # longer text ranks higher


def _setup(tmp_path):
    dense = [_doc("d1", "inguinal canal dense hit")]
    corpus = [_doc("b1", "spermatic cord passes through the inguinal canal"), _doc("x", "kidney"), _doc("y", "liver")]
    bm25 = BM25Okapi([d.page_content.lower().split() for d in corpus])
    store = DocumentStore(str(tmp_path / "r.db"))
    for d in dense + corpus:
        store.store_parent(d.metadata["parent_id"], {"text": f"parent of {d.page_content}", "book_name": "B", "token_count": 5})
    return FakeVectorstore(dense), bm25, corpus, store


def test_retrieve_hybrid_includes_bm25_hits(tmp_path):
    vs, bm25, corpus, store = _setup(tmp_path)
    top, parents = retrieve("spermatic cord", vs, FakeReranker(), store, bm25, corpus, hybrid=True)
    ids = [d.metadata["chunk_id"] for d in top]
    assert ids[0] == "b1" and "d1" in ids
    assert parents[0]["text"] == "parent of spermatic cord passes through the inguinal canal"


def test_retrieve_dense_only_skips_bm25(tmp_path):
    vs, bm25, corpus, store = _setup(tmp_path)
    top, parents = retrieve("spermatic cord", vs, FakeReranker(), store, bm25, corpus, hybrid=False)
    assert [d.metadata["chunk_id"] for d in top] == ["d1"]
    assert vs.queries == ["query: spermatic cord"]


def test_truncate_parent_keeps_leaf_when_trimming():
    """Trimming must alternate toward the leaf, not always cut the same side through it."""
    before = " ".join(f"Filler sentence number {i} about the abdominal wall." for i in range(6))
    leaf = "The deep inguinal ring lies above the midpoint of the inguinal ligament."
    after = " ".join(f"Trailing sentence number {i} about the thigh." for i in range(6))
    parent = f"{before} {leaf} {after}"
    truncated = truncate_parent(parent, parent.find(leaf), len(leaf), max_tokens=token_length(leaf) + 4)
    assert leaf in truncated
