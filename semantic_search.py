#!/usr/bin/env python3
"""
semantic_search.py
- Load embedding JSON file
- Build hybrid retriever: BM25 + dense (cosine)
- Accept console queries
- Retrieve top-K chunks, show sources, and synthesize answer with FLAN-T5
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# CONFIG
EMBED_FILE = r"C:\Users\ganni\Downloads\Knowledge-base Search Engine\docs_embeddings.json"
LLM_MODEL = "google/flan-t5-base"  # will use cache automatically
TOP_K = 3
ALPHA = 0.6         # weight for dense (0..1)
BM25_PENALTY = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SUMMARY_TOKENS = 200

# Load embedding model (for query encoding)
print("Loading embedding model...")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)

# Load LLM
print("Loading FLAN-T5 model (from cache)...")
tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL)
llm = T5ForConditionalGeneration.from_pretrained(LLM_MODEL).to(DEVICE)
print("Models loaded.\n")

# Load embeddings JSON
def load_embeddings(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [d['text'] for d in data]
    docs = [d['doc_name'] for d in data]
    embeddings = np.vstack([np.array(d['embedding'], dtype=float) for d in data])
    tokenized = [text.lower().split() for text in texts]  # simple tokenization
    bm25 = BM25Okapi(tokenized)
    return {
        "entries": data,
        "texts": texts,
        "docs": docs,
        "embeddings": embeddings,
        "tokenized": tokenized,
        "bm25": bm25
    }

# Normalize numpy scores to 0..1
def normalize_scores(scores):
    min_s = np.min(scores)
    max_s = np.max(scores)
    if abs(max_s - min_s) < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)

# Hybrid search: BM25 + dense embeddings
def hybrid_search(index, query, top_k=TOP_K, alpha=ALPHA, bm25_penalty=BM25_PENALTY):
    q_tokens = query.lower().split()
    bm25_scores = index['bm25'].get_scores(q_tokens) * bm25_penalty
    bm25_norm = normalize_scores(bm25_scores)

    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    dense_scores = cosine_similarity([q_emb], index['embeddings'])[0]
    dense_norm = normalize_scores(dense_scores)

    min_len = min(len(bm25_norm), len(dense_norm))
    hybrid = alpha * dense_norm[:min_len] + (1 - alpha) * bm25_norm[:min_len]
    top_idx = np.argsort(hybrid)[::-1][:top_k]
    return top_idx, hybrid[top_idx]

# Generate answer with FLAN-T5
def synthesize_answer(retrieved_texts, question):
    context = "\n\n".join(retrieved_texts)
    prompt = (
    f"Extract the exact factual answer to the question below from the given context. "
    f"If the answer is clearly stated, repeat it exactly as written. "
    f"Do not add explanations or unrelated information.\n\n"
    f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(DEVICE)
    out = llm.generate(enc, max_new_tokens=MAX_SUMMARY_TOKENS, num_beams=4, early_stopping=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    if not os.path.exists(EMBED_FILE):
        print(f"Embedding file not found: {EMBED_FILE}")
        return

    print("Loading embeddings...")
    index = load_embeddings(EMBED_FILE)
    print(f"Loaded {len(index['entries'])} chunks from {EMBED_FILE}")

    while True:
        q = input("\nEnter your question (or 'exit'): ").strip()
        if not q or q.lower() in ['exit', 'quit']:
            print("Goodbye.")
            break

        top_idx, scores = hybrid_search(index, q)
        retrieved_texts = [index['texts'][i] for i in top_idx]
        retrieved_docs = [index['docs'][i] for i in top_idx]

        print("\nRetrieved (top chunks):")
        for rank, i in enumerate(top_idx, start=1):
            snippet = index['texts'][i][:400].replace("\n", " ").strip()
            print(f"[{rank}] doc: {index['docs'][i]}  chunk_id: {index['entries'][i].get('chunk_id','N/A')}")
            print(f"    {snippet} ...")

        print("\nSynthesizing answer (LLM)...")
        answer = synthesize_answer(retrieved_texts, q)
        print("\n===== ANSWER =====")
        print(answer)
        print("\nSources:")
        seen = set()
        for d in retrieved_docs:
            if d not in seen:
                print(" -", d)
                seen.add(d)

if __name__ == "__main__":
    main()
