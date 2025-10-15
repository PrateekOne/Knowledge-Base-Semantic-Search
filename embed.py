#!/usr/bin/env python3
"""
embed_docs.py
- Ingest multiple text / PDF files in a folder
- Chunk text, detect language, no translation (remove translation part)
- Build SentenceTransformer embeddings (normalized)
- Tokenize chunks for BM25
- Save JSON with: doc_name, chunk_id, text, tokens, embedding
"""

import os
import json
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import List
from langdetect import detect
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from PyPDF2 import PdfReader
import re
from rank_bm25 import BM25Okapi
import math

# CONFIG
DOCS = [
    "iPhone_17_Pro_and_iPhone_17_Pro_Max_PER_Sept2025.pdf",
    "All_New_Features_iOS_26_Sept_2025.pdf"
]
OUT_FILENAME_DEFAULT = "docs_embeddings.json"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # good multilingual model
CHUNK_SIZE = 300           # words per chunk
CHUNK_OVERLAP = 50         # words overlap
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Utilities
def extract_text_from_pdf(path: str) -> str:
    text = []
    try:
        with open(path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
    return "\n".join(text)

def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT {path}: {e}")
        return ""

def split_into_chunks(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    # Split by whitespace tokens (words)
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def detect_language_safe(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"

def tokenize_for_bm25(text: str):
    # very simple tokenizer - lowercase alnum tokens
    toks = [w.lower() for w in re.findall(r"\w+", text) if len(w) > 0]
    return toks

def build_embeddings(chunks: List[str], model_name=EMBED_MODEL):
    model = SentenceTransformer(model_name)
    # normalized embeddings for cosine similarity
    embs = model.encode(chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embs)

def main(docs: List[str], out_file: str, chunk_size: int, chunk_overlap: int):
    all_chunks = []
    meta = []  # metadata for each chunk (doc, chunk_id, orig_text, lang)
    print(f"Processing {len(docs)} files. Extracting and chunking...")

    for p in docs:
        print(f"Processing: {p}")
        text = extract_text_from_pdf(p)
        
        # Basic cleanup: collapse long whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = split_into_chunks(text, chunk_size=chunk_size, overlap=chunk_overlap)
        for i, c in enumerate(chunks):
            lang = detect_language_safe(c[:1000])  # detect on first 1000 chars
            all_chunks.append(c)
            meta.append({
                "doc_name": Path(p).name,
                "chunk_id": i,
                "language": lang
            })

    if not all_chunks:
        print("No chunks extracted from documents.")
        return

    # Build BM25 tokenized corpus
    tokenized_corpus = [tokenize_for_bm25(t) for t in all_chunks]

    # Build embeddings (dense)
    print("Encoding chunks to embeddings (this may take time)...")
    dense_embeddings = build_embeddings(all_chunks)

    # Save everything to JSON
    out_data = []
    for i, m in enumerate(meta):
        out_data.append({
            "doc_name": m["doc_name"],
            "chunk_id": m["chunk_id"],
            "orig_text": all_chunks[i],
            "text": all_chunks[i],  # no translation, use the original text
            "tokens": tokenized_corpus[i],
            "embedding": dense_embeddings[i].tolist()
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(out_data)} chunks to {out_file}")

if __name__ == "__main__":
    main(DOCS, OUT_FILENAME_DEFAULT, CHUNK_SIZE, CHUNK_OVERLAP)
