import argparse
import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_documents")


# TOC-based chapter mapping (adjust for your textbook)
CHAPTER_MAP = {
    (1, 9): "Introduction and Overview of the Abdomen",
    (10, 23): "Osteology of the Abdomen",
    (24, 45): "Anterior Abdominal Wall",
    (46, 58): "Inguinal Region/Groin",
    (59, 73): "Male External Genital Organs",
    (74, 92): "Abdominal Cavity and Peritoneum",
    (93, 108): "Abdominal Part of Esophagus, Stomach, and Spleen",
    (109, 125): "Liver and Extrahepatic Biliary Apparatus",
    (126, 143): "Duodenum, Pancreas, and Portal Vein",
    (144, 164): "Small and Large Intestines",
    (165, 184): "Kidneys, Ureters, and Suprarenal Glands",
    (185, 201): "Posterior Abdominal Wall and Associated Structures",
    (202, 211): "Pelvis",
    (212, 224): "Pelvic Walls and Associated Soft Tissue Structures",
    (225, 237): "Perineum",
    (238, 250): "Urinary Bladder and Urethra",
    (251, 259): "Male Genital Organs",
    (260, 278): "Female Genital Organs",
    (279, 290): "Rectum and Anal Canal",
    (291, 298): "Introduction to the Lower Limb",
    (299, 327): "Bones of the Lower Limb",
    (328, 343): "Front of the Thigh",
    (344, 352): "Medial Side of the Thigh",
    (353, 363): "Gluteal Region",
    (364, 376): "Back of the Thigh and Popliteal Fossa",
    (377, 385): "Hip Joint",
    (386, 399): "Front of the Leg and Dorsum of the Foot",
    (400, 406): "Lateral and Medial Sides of the Leg",
    (407, 419): "Back of the Leg",
    (420, 431): "Sole of the Foot",
    (432, 438): "Arches of the Foot",
    (439, 457): "Joints of the Lower Limb",
    (458, 466): "Venous and Lymphatic Drainage of the Lower Limb",
    (467, 478): "Innervation of the Lower Limb",
}


def get_chapter_by_page(page_number):
    for (start, end), title in CHAPTER_MAP.items():
        if start <= page_number <= end:
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
        page_number = i + 1
        chapter = get_chapter_by_page(page_number)
        chunks = splitter.split_documents([page])
        for j, chunk in enumerate(chunks):
            chunk.metadata = {
                "chapter": chapter,
                "page_number": page_number,
                "labels": detect_labels(chunk.page_content),
                "book_name": book_name,
                "chunk_id": f"{chapter[:20].replace(' ', '_')}-{page_number}-{j}",
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
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name")
    parser.add_argument("--chunk-size", type=int, default=250, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in characters")
    return parser.parse_args()


def main():
    args = parse_args()

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")

    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    embedding_model = SentenceTransformerEmbeddings(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO"
    )

    ensure_collection(qdrant_client, args.collection, vector_size=768)

    chunks = load_and_chunk_pdf(args.pdf, args.chunk_size, args.chunk_overlap)
    ids = [doc.metadata.get("chunk_id") for doc in chunks]

    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=args.collection,
        embeddings=embedding_model,
    )

    vectorstore.add_documents(chunks, ids=ids)
    print(f"[OK] Added {len(chunks)} chunks to '{args.collection}'")


if __name__ == "__main__":
    main()
