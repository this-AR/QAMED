import os
import sys
import json
import time
from datetime import datetime

# Make sure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GROQ_MODEL, GROQ_API_KEY, EMBEDDING_MODEL_NAME, MAX_PARENT_CONTEXT_TOKENS, BM25_TOP_K, DENSE_TOP_K, RRF_K
from services.retrieval import load_models_and_clients, bm25_search, rrf_fusion, rerank_docs, expand_to_parents
from services.llm import build_prompt, stream_groq_answer
from services.guardrails import check_hallucination
from evaluation.test_questions import TEST_QUESTIONS

# RAGAS imports
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

def run_benchmark():
    print("Loading models and clients...")
    groq_client, vectorstore, reranker, doc_store, bm25_index, bm25_corpus = load_models_and_clients()
    
    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": DENSE_TOP_K})
    
    methods = ["hybrid", "dense"]
    prompts = ["v1", "v2", "v3"]
    
    results = []
    
    for q_idx, query in enumerate(TEST_QUESTIONS, 1):
        print(f"\n--- Question {q_idx}/{len(TEST_QUESTIONS)}: {query} ---")
        
        # We can pre-calculate the retrieved chunks for both methods since they don't depend on the prompt
        t0 = time.time()
        dense_docs = retriever.invoke("query: " + query)
        dense_top = rerank_docs(reranker, query, dense_docs, top_n=6)
        dense_parents = expand_to_parents(dense_top, doc_store, MAX_PARENT_CONTEXT_TOKENS)
        t_dense_retrieval = time.time() - t0
        
        t0 = time.time()
        sparse_docs = bm25_search(query, bm25_index, bm25_corpus, top_k=BM25_TOP_K)
        fused_docs = rrf_fusion(dense_docs, sparse_docs, k=RRF_K)
        hybrid_top = rerank_docs(reranker, query, fused_docs, top_n=6)
        hybrid_parents = expand_to_parents(hybrid_top, doc_store, MAX_PARENT_CONTEXT_TOKENS)
        t_hybrid_retrieval = time.time() - t0
        
        # Calculate overlap
        dense_ids = {doc.metadata.get("chunk_id", str(hash(doc.page_content))) for doc in dense_top}
        hybrid_ids = {doc.metadata.get("chunk_id", str(hash(doc.page_content))) for doc in hybrid_top}
        overlap = len(dense_ids.intersection(hybrid_ids))
        overlap_ratio = overlap / max(len(hybrid_ids), 1)
        
        for method in methods:
            is_hybrid = method == "hybrid"
            top_docs = hybrid_top if is_hybrid else dense_top
            parent_sections = hybrid_parents if is_hybrid else dense_parents
            retrieval_latency = t_hybrid_retrieval if is_hybrid else t_dense_retrieval
            context_chunks = [doc.page_content for doc in top_docs]
            
            for prompt_ver in prompts:
                print(f"Running config: {method} + {prompt_ver}")
                sys_msg, usr_msg, resolved_version = build_prompt(query, parent_sections, top_docs, prompt_ver)
                
                t0 = time.time()
                answer = "".join(list(stream_groq_answer(groq_client, GROQ_MODEL, sys_msg, usr_msg)))
                gen_latency = time.time() - t0
                
                guardrail = check_hallucination(answer, context_chunks, groq_client, GROQ_MODEL)
                
                # RAGAS evaluation
                dataset = Dataset.from_dict({
                    "question": [query],
                    "answer": [answer],
                    "contexts": [context_chunks]
                })
                
                print("Running RAGAS eval...")
                try:
                    eval_result = evaluate(
                        dataset,
                        metrics=[faithfulness, answer_relevancy],
                        llm=llm,
                        embeddings=embeddings,
                        raise_exceptions=False
                    )
                    eval_df = eval_result.to_pandas()
                    f_score = float(eval_df.iloc[0].get("faithfulness", 0.0) or 0.0)
                    ar_score = float(eval_df.iloc[0].get("answer_relevancy", 0.0) or 0.0)
                except Exception as e:
                    print(f"RAGAS eval failed: {e}")
                    f_score = 0.0
                    ar_score = 0.0
                
                result_entry = {
                    "question": query,
                    "method": method,
                    "prompt_version": resolved_version,
                    "answer": answer,
                    "retrieval_latency_ms": round(retrieval_latency * 1000, 2),
                    "generation_latency_ms": round(gen_latency * 1000, 2),
                    "overlap_ratio": round(overlap_ratio, 2),
                    "sources_used": len(top_docs),
                    "guardrail_is_grounded": guardrail.is_grounded,
                    "guardrail_explanation": guardrail.explanation,
                    "faithfulness": round(f_score, 4),
                    "answer_relevancy": round(ar_score, 4)
                }
                results.append(result_entry)
                
    # Save results
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    out_file = os.path.join(os.path.dirname(__file__), "results", "benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {out_file}")
    
if __name__ == "__main__":
    run_benchmark()
