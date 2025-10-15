# 🧠 Knowledge-Base Semantic Search Engine  
### Two-Stage Hybrid Retrieval + LLM-Based Question Answering  

This repository implements a **two-step knowledge-base semantic search pipeline** that combines **BM25 sparse retrieval** and **SentenceTransformer dense embeddings** for accurate document chunk retrieval. The retrieved chunks are then passed to **FLAN-T5** for concise, context-aware answer synthesis.

---

## 🚀 Features

- **Multi-document ingestion:** Supports multiple `.pdf` and `.txt` files.  
- **Automatic text chunking:** Splits documents into overlapping word-level chunks for better retrieval.  
- **Language detection:** Detects language per chunk (no translation performed).  
- **Dual retrieval:** Combines BM25 (lexical) and MiniLM (semantic) similarity.  
- **LLM answer generation:** Uses `google/flan-t5-base` to synthesize precise factual answers.  
- **Lightweight & extensible:** Modular code for easy scaling to other datasets or models.

---

## 🧩 Project Structure

```

├── embed_docs.py          # Step 1: Build embeddings and BM25 tokens from docs
├── semantic_search.py     # Step 2: Run hybrid retrieval and FLAN-T5 QA
├── docs_embeddings.json   # Output file containing chunk metadata + embeddings
├── requirements.txt       # Dependencies (see below)
└── sample_docs/           # Folder containing PDF/TXT files

````

---

## ⚙️ Step 1: Document Embedding (`embed_docs.py`)

### Description
This script:
- Extracts text from multiple PDFs or TXT files.  
- Splits content into overlapping chunks (default: 300 words with 50 overlap).  
- Detects language of each chunk.  
- Creates BM25 tokens and **SentenceTransformer** embeddings.  
- Saves everything to a JSON file for later retrieval.

### Run
```bash
python embed_docs.py
````

### Output

A JSON file (default: `docs_embeddings.json`) with entries like:

```json
{
  "doc_name": "iPhone_17_Pro_and_iPhone_17_Pro_Max_PER_Sept2025.pdf",
  "chunk_id": 4,
  "orig_text": "Apple introduces ...",
  "tokens": ["apple", "introduces", ...],
  "embedding": [0.123, -0.045, ...]
}
```

---

## 🔍 Step 2: Semantic Search & QA (`semantic_search.py`)

### Description

This script:

* Loads the `docs_embeddings.json` file.
* Builds a **hybrid retriever** combining BM25 + cosine similarity on dense vectors.
* Accepts **natural-language questions** from the console.
* Retrieves the most relevant text chunks.
* Synthesizes concise answers using **FLAN-T5**.

### Run

```bash
python semantic_search.py
```

### Example

```
Enter your question (or 'exit'): What are the new features of iPhone 17 Pro?
```

**Output:**

```
Retrieved (top chunks):
[1] doc: iPhone_17_Pro_and_iPhone_17_Pro_Max_PER_Sept2025.pdf chunk_id: 3
    The iPhone 17 Pro introduces a new A19 Bionic chip with enhanced thermal efficiency ...

===== ANSWER =====
The iPhone 17 Pro features the A19 Bionic chip, improved thermal system, and redesigned camera array.

Sources:
 - iPhone_17_Pro_and_iPhone_17_Pro_Max_PER_Sept2025.pdf
```

---

## 🧮 Hybrid Scoring

| Component      | Description                           | Weight  |
| -------------- | ------------------------------------- | ------- |
| BM25           | Lexical relevance using token overlap | 1 - α   |
| Dense (MiniLM) | Semantic similarity (cosine)          | α = 0.6 |

**Final score:**
`s = α * dense_score + (1 - α) * bm25_score`

---

## 🧠 Models Used

| Model                                                         | Purpose           | Source                                                                                                     |
| ------------------------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------- |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Dense embeddings  | [SentenceTransformers](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| `google/flan-t5-base`                                         | Answer generation | [Hugging Face](https://huggingface.co/google/flan-t5-base)                                                 |

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
torch
numpy
tqdm
langdetect
sentence-transformers
transformers
rank-bm25
PyPDF2
scikit-learn
```

---

## 🧰 Configuration

You can modify these parameters inside each script:

| Parameter       | Default       | Description                     |
| --------------- | ------------- | ------------------------------- |
| `CHUNK_SIZE`    | 300           | Words per chunk                 |
| `CHUNK_OVERLAP` | 50            | Overlap between chunks          |
| `ALPHA`         | 0.6           | Weight for dense vs BM25 scores |
| `TOP_K`         | 3             | Number of chunks to retrieve    |
| `EMBED_MODEL`   | MiniLM-L12-v2 | Embedding model name            |
| `LLM_MODEL`     | flan-t5-base  | LLM for answer generation       |

---

## 🧪 Example Use-Case

You can use this system for:

* Company knowledge-base search
* Research document Q&A
* Product manuals or specification lookup
* Multi-language document retrieval

---

## ✨ Author

**Prateek Ganni**
Hybrid Retrieval + LLM Knowledge-Base System (2025)

```
