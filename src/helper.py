import fitz
import re
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import time
import os
import json

# Initialize models once to avoid reloading on every function call
# model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
# model = SentenceTransformer(
#     "all-MiniLM-L6-v2",
#     device="cuda"  # use GPU
# )
model = SentenceTransformer("intfloat/e5-small", device="cuda")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device = "cuda")

def extract_pdf_text(pdf_path):
    """Extracts text from PDF pages."""
    doc = fitz.open(pdf_path)
    pages_data = []
    for page_num in range(len(doc)):
        text = doc.load_page(page_num).get_text("text")
        if text.strip():
            pages_data.append({
                "pdf_name": Path(pdf_path).name,
                "page_number": page_num + 1,
                "text": text.strip()
            })
    doc.close()
    return pages_data

def chunk_text(pages_data, chunk_size=200, overlap=40):
    """Splits text into overlapping chunks for better retrieval."""
    all_chunks = []
    for page in pages_data:
        paragraphs = re.split(r'\n\s*\n', page["text"])
        current_chunk = []
        for para in paragraphs:
            words = para.split()
            if len(current_chunk) + len(words) <= chunk_size:
                current_chunk.extend(words)
            else:
                if current_chunk:
                    all_chunks.append({
                        "pdf_name": page["pdf_name"],
                        "page_number": page["page_number"],
                        "text": " ".join(current_chunk)
                    })
                    current_chunk = current_chunk[-overlap:]
                current_chunk.extend(words)
        if current_chunk:
            all_chunks.append({
                "pdf_name": page["pdf_name"],
                "page_number": page["page_number"],
                "text": " ".join(current_chunk)
            })
    return all_chunks

def create_search_index(chunks):
    """Generates embeddings and initializes FAISS and BM25."""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs("storage", exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, "storage/faiss.index")

    # Save embeddings
    np.save("storage/embeddings.npy", embeddings)

    # Save metadata
    with open("storage/metadata.json", "w") as f:
        json.dump(chunks, f)
    
    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return index, bm25, embeddings

def load_existing_index():
    if not os.path.exists("storage/faiss.index"):
        return None, None, None

    index = faiss.read_index("storage/faiss.index")

    with open("storage/metadata.json", "r") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]
    tokenized_corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    return index, chunks, bm25