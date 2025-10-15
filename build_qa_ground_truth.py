#!/usr/bin/env python3
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
LLM_MODEL = "google/flan-t5-base"
TOP_K = 3
ALPHA = 0.6
BM25_PENALTY = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SUMMARY_TOKENS = 200
OUTPUT_FILE = "qa_ground_truth.json"

questions = [
    "What are the main new features introduced in the iPhone 17 Pro?",
    "How does the iPhone 17 Pro Max differ from the iPhone 17 Pro in terms of camera?",
    "What processor powers the iPhone 17 Pro series?",
    "What battery-life improvements were made in the iPhone 17 Pro models?",
    "What new display technology does the iPhone 17 Pro use compared to older models?",
    "What are the major software updates introduced with iOS 26?",
    "How does iOS 26 improve privacy or security for users?",
    "What AI-based features were added to iOS 26?",
    "Does iOS 26 bring any new Siri or voice-assistant capabilities?",
    "What are the new connectivity or charging improvements in iPhone 17 devices?"
]

# Load embedding model
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)

# Load LLM
tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL)
llm = T5ForConditionalGeneration.from_pretrained(LLM_MODEL).to(DEVICE)

# Load embeddings JSON
def load_embeddings(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [d['text'] for d in data]
    docs = [d['doc_name'] for d in data]
    embeddings = np.vstack([np.array(d['embedding'], dtype=float) for d in data])
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    return {"entries": data, "texts": texts, "docs": docs, "embeddings": embeddings, "tokenized": tokenized, "bm25": bm25}

def normalize_scores(scores):
    min_s = np.min(scores)
    max_s = np.max(scores)
    if abs(max_s - min_s) < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)

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

    index = load_embeddings(EMBED_FILE)
    qa_pairs = []

    for q in questions:
        print(f"Processing question: {q}")
        top_idx, _ = hybrid_search(index, q)
        retrieved_texts = [index['texts'][i] for i in top_idx]
        answer = synthesize_answer(retrieved_texts, q)
        qa_pairs.append({"question": q, "answer": answer})
        print(f"Answer: {answer}\n")

    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"QA ground truth saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
