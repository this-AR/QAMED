import os
import re
import time

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from transformers import pipeline


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medical_documents")


st.set_page_config(page_title="Medical RAG", layout="wide")

st.title("Medical RAG with Qdrant + Groq")
st.caption("Streamed answers grounded in medical textbook chunks.")


if "ratings" not in st.session_state:
    st.session_state["ratings"] = {}


def extract_subquestions(query):
    noun_match = re.search(r"what\s+is\s+(.*?)(?:\s*(,|and|\.|\?)|$)", query, re.IGNORECASE)
    subject = noun_match.group(1).strip() if noun_match else None

    prompt = f"""
You are a helpful medical assistant.

Break down each medical question into smaller subquestions that cover one clear medical concept at a time.

Examples:

Question: Tell me about lung surfaces, borders and structures surrounding it.
Subquestions:
1. What are the surfaces of the lungs?
2. What are the borders of the lungs?
3. What are the structures surrounding the lungs?

Question: Tell me about liver anatomy and function.
Subquestions:
1. What is the anatomy of the liver?
2. What are the functions of the liver?

Question: Tell me about the surfaces, borders and relations of the liver.
Subquestions:
1. What are the surfaces of the liver?
2. What are the borders of the liver?
3. What are the relations of the liver?

Question: {query}
Subquestions:
"""

    output = st.session_state["subq_pipe"](prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
    raw = re.findall(r"\d+\.\s*([^0-9]+(?:\?.*?)?)", output)

    cleaned = []
    seen_normalized = set()

    for q in raw:
        q_clean = q.strip()
        if subject:
            q_clean = re.sub(r"\bits\b", f"the {subject}", q_clean, flags=re.IGNORECASE)
            q_clean = re.sub(r"\bit\b", f"the {subject}", q_clean, flags=re.IGNORECASE)
            q_clean = re.sub(r"\btheir\b", f"the {subject}'s", q_clean, flags=re.IGNORECASE)

        q_clean = re.sub(r"\bthe\s+the\b", "the", q_clean, flags=re.IGNORECASE)

        norm = re.sub(r"[^a-z]", "", q_clean.lower())
        if norm not in seen_normalized:
            cleaned.append(q_clean)
            seen_normalized.add(norm)

    stopwords = {"what", "are", "is", "the", "of", "in", "its", "a", "an", "and", "on"}

    def keyword_set(text):
        tokens = re.findall(r"\w+", text.lower())
        return set(t for t in tokens if t not in stopwords)

    final = []
    seen_keywords = []

    for q in cleaned:
        q_keywords = keyword_set(q)
        if any(q_keywords == existing for existing in seen_keywords):
            continue
        final.append(q)
        seen_keywords.append(q_keywords)

    return final


# ── UPDATED: proper system/user role split instead of inline <|system|> tags ──

def build_prompt(subquery, docs):
    context_blocks = []
    for i, doc in enumerate(docs, 1):
        passage = doc.page_content.strip().replace("\n", " ")
        context_blocks.append(f"[{i}] {passage}")

    system_msg = (
    "You are a precise medical assistant. "
    "Answer strictly using the numbered source excerpts below. "
    "If the answer isn't in the sources, say: 'Not found in provided sources.'"
    )

    user_msg = (
    f"Sources:\n{chr(10).join(context_blocks)}\n\n"
    f"Question: {subquery}\n\n"
    "Give a complete and accurate answer based only on the sources. "
    "Be as detailed as needed do not omit clinically relevant information. "
    "Cite source numbers inline where relevant."
    )

    return system_msg, user_msg


def stream_groq_answer(system_msg, user_msg):
    response = st.session_state["groq_client"].chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,      # Lower = more deterministic for medical facts
        top_p=0.9,
        max_tokens=300,       # Tightened to match 2–4 sentence output
        stream=True,
    )

    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_models_and_clients():
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY in environment variables.")

    groq_client = Groq(api_key=GROQ_API_KEY)
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    embedding_model = SentenceTransformerEmbeddings(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO"
    )
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
    )

    subq_pipe = pipeline("text2text-generation", model="google/flan-t5-base")

    return groq_client, vectorstore, reranker, subq_pipe


with st.spinner("Loading models and connections..."):
    try:
        groq_client, vectorstore, reranker, subq_pipe = load_models_and_clients()
        st.session_state["groq_client"] = groq_client
        st.session_state["vectorstore"] = vectorstore
        st.session_state["reranker"] = reranker
        st.session_state["subq_pipe"] = subq_pipe
    except Exception as exc:
        st.error(str(exc))
        st.stop()


query = st.text_input("Ask a medical question", value="What is the inguinal canal, and what are its contents?")
use_decomposition = st.checkbox("Use query decomposition", value=True)

if st.button("Ask"):
    start_time = time.time()
    st.write("---")

    if use_decomposition:
        subqueries = extract_subquestions(query)
    else:
        subqueries = [query]

    if not subqueries:
        st.warning("No subquestions generated. Try rephrasing your query.")
        st.stop()

    retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 15})

    for idx, subquery in enumerate(subqueries, start=1):
        st.subheader(f"Subquestion {idx}: {subquery}")
        docs = retriever.get_relevant_documents("query: " + subquery)
        scores = st.session_state["reranker"].predict([(subquery, doc.page_content) for doc in docs])
        top_docs = [doc for doc, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:2]]

        # ── UPDATED: unpack tuple returned by build_prompt ──
        system_msg, user_msg = build_prompt(subquery, top_docs)
        answer_box = st.empty()
        answer_text = ""
        for token in stream_groq_answer(system_msg, user_msg):
            answer_text += token
            answer_box.markdown(answer_text)

        rating_key = f"rating_{idx}_{hash(subquery)}"
        rating = st.slider("Rate this answer", 1, 10, 7, key=rating_key)
        if st.button("Submit rating", key=f"submit_{rating_key}"):
            st.session_state["ratings"][subquery] = rating
            st.success(f"Saved rating: {rating}/10")

        with st.expander("Sources"):
            for doc in top_docs:
                meta = doc.metadata or {}
                st.markdown(
                    f"- **Chapter:** {meta.get('chapter', 'N/A')}\n"
                    f"- **Page:** {meta.get('page_number', 'N/A')}\n"
                    f"- **Book:** {meta.get('book_name', 'N/A')}\n"
                    f"- **Labels:** {', '.join(meta.get('labels', [])) if meta.get('labels') else 'N/A'}"
                )
                st.write(doc.page_content)

    st.caption(f"Completed in {time.time() - start_time:.2f} seconds")
