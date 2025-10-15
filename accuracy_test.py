# #!/usr/bin/env python3
# """
# accuracy_test.py
# - Evaluate retrieval accuracy of semantic_search_console system
# - Uses Recall@K, MRR, and semantic similarity metrics
# """

# import json
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi
# from semantic_search import load_embeddings, hybrid_search, embedder

# # CONFIG
# EMBED_FILE = r"C:\Users\ganni\Downloads\Knowledge-base Search Engine\docs_embeddings.json"
# QA_FILE = "qa_ground_truth.json"
# TOP_K = 5
# SIM_THRESHOLD = 0.65  # similarity threshold to consider match

# def evaluate_retrieval(index, qa_pairs):
#     """Evaluate retrieval accuracy on QA pairs."""
#     model = embedder
#     correct_at_k = 0
#     total = len(qa_pairs)
#     reciprocal_ranks = []
#     similarities = []

#     for item in qa_pairs:
#         q, true_answer = item["question"], item["answer"]
#         top_idx, _ = hybrid_search(index, q, top_k=TOP_K)
#         retrieved_texts = [index["texts"][i] for i in top_idx]

#         # Encode true answer and retrieved chunks for similarity
#         ans_emb = model.encode([true_answer], normalize_embeddings=True)
#         retr_embs = model.encode(retrieved_texts, normalize_embeddings=True)
#         sims = cosine_similarity(ans_emb, retr_embs)[0]

#         # Highest similarity among top-K
#         best_sim = np.max(sims)
#         similarities.append(best_sim)

#         # Determine rank of first relevant retrieval
#         match_rank = np.argmax(sims >= SIM_THRESHOLD) + 1 if np.any(sims >= SIM_THRESHOLD) else None
#         if match_rank:
#             reciprocal_ranks.append(1.0 / match_rank)
#             if match_rank <= TOP_K:
#                 correct_at_k += 1
#         else:
#             reciprocal_ranks.append(0)

#         print(f"\nQ: {q}")
#         print(f"Best sim: {best_sim:.3f} | Match rank: {match_rank if match_rank else 'none'}")

#     recall_at_k = correct_at_k / total
#     mrr = np.mean(reciprocal_ranks)
#     avg_sim = np.mean(similarities)

#     print("\n=== EVALUATION RESULTS ===")
#     print(f"Total questions: {total}")
#     print(f"Recall@{TOP_K}: {recall_at_k:.3f}")
#     print(f"MRR: {mrr:.3f}")
#     print(f"Avg Cosine Similarity: {avg_sim:.3f}")

# def main():
#     print("Loading embeddings and QA pairs...")
#     index = load_embeddings(EMBED_FILE)
#     with open(QA_FILE, "r", encoding="utf-8") as f:
#         qa_pairs = json.load(f)
#     evaluate_retrieval(index, qa_pairs)

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
#!/usr/bin/env python3
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# CONFIG
EMBED_FILE = r"C:\Users\ganni\Downloads\Knowledge-base Search Engine\docs_embeddings.json"
LLM_MODEL = "google/flan-t5-base"
TOP_K = 3
ALPHA = 0.6
BM25_PENALTY = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SUMMARY_TOKENS = 200
GROUND_TRUTH_FILE = "qa_ground_truth.json"

# Load embedding model
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)

# Load LLM
tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL)
llm = T5ForConditionalGeneration.from_pretrained(LLM_MODEL).to(DEVICE)

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

def similarity(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def main():
    if not os.path.exists(EMBED_FILE):
        print(f"Embedding file not found: {EMBED_FILE}")
        return
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Ground truth file not found: {GROUND_TRUTH_FILE}")
        return

    index = load_embeddings(EMBED_FILE)
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    total = len(ground_truth)
    correct = 0

    print("\n=== Differences Found ===\n")

    for entry in ground_truth:
        question = entry['question']
        true_answer = entry['answer']

        top_idx, _ = hybrid_search(index, question)
        retrieved_texts = [index['texts'][i] for i in top_idx]
        pred_answer = synthesize_answer(retrieved_texts, question)

        sim = similarity(pred_answer, true_answer)
        if sim > 0.9:
            correct += 1
        else:
            # print(f"Q: {question}")
            # print(f"Ground Truth: {true_answer}")
            print(f"Predicted: {pred_answer}")
            print(f"Similarity: {sim:.2f}\n")

    accuracy = (correct / total) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()
