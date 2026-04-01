import argparse
import os
import uuid

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_documents")

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


def detect_labels(text):
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

    return labels or ["@general"]


def load_and_chunk_pdf(pdf_path, chunk_size, chunk_overlap):
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_docs = []
    book_name = os.path.basename(pdf_path)

    for i, page in enumerate(pages):
        pdf_page = i + 1
        book_page = pdf_page - PAGE_OFFSET

        chapter = get_chapter_by_page(pdf_page)

        chunks = splitter.split_documents([page])

        for j, chunk in enumerate(chunks):
            chunk.metadata = {
                "chapter": chapter,
                "pdf_page": pdf_page,
                "book_page": book_page,
                "labels": detect_labels(chunk.page_content),
                "book_name": book_name,
                "chunk_id": f"{book_name}-{pdf_page}-{j}",
            }
            chunked_docs.append(chunk)

    return chunked_docs


def ensure_collection(client, collection_name, vector_size):
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest a PDF into Qdrant Cloud.")
    parser.add_argument("--pdf", required=True, help="Path to the medical PDF")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in .env")

    print("Connecting to Qdrant...")

    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO"
    )

    ensure_collection(qdrant_client, args.collection, vector_size=768)

    print("Processing PDF...")

    chunks = load_and_chunk_pdf(
        args.pdf,
        args.chunk_size,
        args.chunk_overlap
    )

    print(f"Generated {len(chunks)} chunks")

    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, doc.metadata["chunk_id"])) for doc in chunks]

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=args.collection,
        embedding=embedding_model,
    )

    print("Uploading to Qdrant...")

    vectorstore.add_documents(chunks, ids=ids)

    print(f"[OK] Successfully added {len(chunks)} chunks to '{args.collection}'")


if __name__ == "__main__":
    main()