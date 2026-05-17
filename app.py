import os
import re
import time
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

load_dotenv()

# ── Environment Variables ──────────────────────────────────────────────────────
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medical_documents")
DEFAULT_PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")


PROMPT_TEMPLATES = {
    "v1": {
        "name": "Strict Grounded (Original)",
        "system": (
            "You are a precise medical assistant trained on MBBS textbooks. "
            "Answer strictly using the numbered source excerpts provided. "
            "Use precise medical terminology as it appears in the sources. "
            "Do not simplify or paraphrase medical terms. "
            "Cite source numbers inline like [1] or [2] where relevant. "
            "If the answer is not in the sources, say: 'Not found in provided sources.'"
        ),
        "user": (
            "Sources:\n{sources}\n\n"
            "Question: {question}\n\n"
            "Give a complete and accurate answer based only on the sources. "
            "Include all clinically relevant information. "
            "Do not omit important anatomical or clinical details."
        ),
    },
    "v2": {
        "name": "Structured Clinical Answer",
        "system": (
            "You are a precise medical assistant trained on MBBS textbooks. "
            "Use only the numbered source excerpts. "
            "If evidence is insufficient, say: 'Not found in provided sources.' "
            "Preserve exact medical terminology from sources and add inline citations like [1]."
        ),
        "user": (
            "Sources:\n{sources}\n\n"
            "Question: {question}\n\n"
            "Answer in this order when information is available: "
            "Definition, Key anatomical/clinical details, Important exceptions/variations, Clinical significance. "
            "Keep the answer concise but complete and source-grounded."
        ),
    },
}

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="QAMed — Medical Assistant", layout="wide")
st.title("QAMed — Medical RAG Assistant")
st.caption("Streamed answers grounded in MBBS textbook chunks.")

# ── Session State Init ────────────────────────────────────────────────────────
if "ratings" not in st.session_state:
    st.session_state["ratings"] = {}
if "prompt_runs" not in st.session_state:
    st.session_state["prompt_runs"] = []


def resolve_prompt_version(requested_version: str) -> str:
    return requested_version if requested_version in PROMPT_TEMPLATES else "v1"


# ── Query Decomposition via Groq ──────────────────────────────────────────────
def extract_subquestions(query: str) -> list[str]:
    response = st.session_state["groq_client"].chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical assistant. "
                    "Break the question into focused subquestions, one concept each. "
                    "Return only a numbered list. Maximum 4 subquestions. "
                    "Do not add explanations or preamble."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\nSubquestions:",
            },
        ],
        temperature=0,
        max_tokens=200,
    )

    output = response.choices[0].message.content
    raw    = re.findall(r"\d+\.\s*(.+)", output)
    result = [q.strip() for q in raw if q.strip()]
    return result if result else [query]


# ── Prompt Builder ─────────────────────────────────────────────────────────────
def build_prompt(subquery: str, docs: list, prompt_version: str) -> tuple[str, str, str]:
    context_blocks = []
    for i, doc in enumerate(docs, 1):
        passage = doc.page_content.strip().replace("\n", " ")
        context_blocks.append(f"[{i}] {passage}")

    version = resolve_prompt_version(prompt_version)
    template = PROMPT_TEMPLATES[version]

    system_msg = template["system"]
    user_msg = template["user"].format(
        sources=chr(10).join(context_blocks),
        question=subquery,
    )

    return system_msg, user_msg, version


# ── Groq Streaming ─────────────────────────────────────────────────────────────
def stream_groq_answer(system_msg: str, user_msg: str):
    response = st.session_state["groq_client"].chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,
        top_p=0.9,
        max_tokens=512,
        stream=True,
    )

    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── Model + Client Loading ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models_and_clients():
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY in environment variables.")

    groq_client   = Groq(api_key=GROQ_API_KEY)
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    embedding_model = HuggingFaceEmbeddings(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO"
    )

    reranker = CrossEncoder("BAAI/bge-reranker-base")

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )

    return groq_client, vectorstore, reranker


# ── Load on Startup ────────────────────────────────────────────────────────────
with st.spinner("Loading models and connections..."):
    try:
        groq_client, vectorstore, reranker = load_models_and_clients()
        st.session_state["groq_client"] = groq_client
        st.session_state["vectorstore"] = vectorstore
        st.session_state["reranker"]    = reranker
    except Exception as exc:
        st.error(str(exc))
        st.stop()


# ── UI ─────────────────────────────────────────────────────────────────────────
query             = st.text_input("Ask a medical question", placeholder="e.g. What is the inguinal canal and what are its contents?")
use_decomposition = st.checkbox("Use query decomposition", value=True)

prompt_options = list(PROMPT_TEMPLATES.keys())
default_version = resolve_prompt_version(DEFAULT_PROMPT_VERSION)
default_index = prompt_options.index(default_version) if default_version in prompt_options else 0
active_prompt_version = st.selectbox(
    "Prompt version",
    options=prompt_options,
    index=default_index,
    format_func=lambda version: f"{version} - {PROMPT_TEMPLATES[version]['name']}",
)
st.caption(f"Active prompt version: {active_prompt_version}")

if st.button("Ask", type="primary") and query.strip():
    start_time = time.time()
    st.write("---")

    subqueries = extract_subquestions(query) if use_decomposition else [query]

    if not subqueries:
        st.warning("No subquestions generated. Try rephrasing your query.")
        st.stop()

    retriever = st.session_state["vectorstore"].as_retriever(
        search_kwargs={"k": 15}
    )

    for idx, subquery in enumerate(subqueries, start=1):
        st.subheader(f"Subquestion {idx}: {subquery}")

        docs   = retriever.invoke("query: " + subquery)
        scores = st.session_state["reranker"].predict(
            [(subquery, doc.page_content) for doc in docs]
        )
        top_docs = [
            doc for doc, _ in
            sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:4]
        ]

        system_msg, user_msg, used_prompt_version = build_prompt(
            subquery,
            top_docs,
            active_prompt_version,
        )

        answer_box  = st.empty()
        answer_text = ""
        for token in stream_groq_answer(system_msg, user_msg):
            answer_text += token
            answer_box.markdown(answer_text)

        st.caption(f"Prompt version used: {used_prompt_version}")

        st.session_state["prompt_runs"].append(
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "query": query,
                "subquery": subquery,
                "prompt_version": used_prompt_version,
                "model": GROQ_MODEL,
                "sources_used": len(top_docs),
            }
        )

        rating_key = f"rating_{idx}_{hash(subquery)}"
        col1, col2 = st.columns([3, 1])
        with col1:
            rating = st.slider("Rate this answer", 1, 10, 7, key=rating_key)
        with col2:
            if st.button("Submit", key=f"submit_{rating_key}"):
                st.session_state["ratings"][subquery] = rating
                st.success(f"Saved {rating}/10")

        with st.expander("View sources"):
            for doc in top_docs:
                meta = doc.metadata or {}
                st.markdown(
                    f"**Chapter:** {meta.get('chapter', 'N/A')} &nbsp;|&nbsp; "
                    f"**Page:** {meta.get('page_number', 'N/A')} &nbsp;|&nbsp; "
                    f"**Book:** {meta.get('book_name', 'N/A')} &nbsp;|&nbsp; "
                    f"**Labels:** {', '.join(meta.get('labels', [])) if meta.get('labels') else 'N/A'}"
                )
                st.write(doc.page_content)
                st.write("---")

    st.caption(f"Completed in {time.time() - start_time:.2f}s")

    with st.expander("Prompt run history"):
        for item in reversed(st.session_state["prompt_runs"][-20:]):
            st.markdown(
                f"- **{item['timestamp']}** | version={item['prompt_version']} | "
                f"model={item['model']} | sources={item['sources_used']}"
            )