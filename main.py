from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os

from src.helper import (
    extract_pdf_text, chunk_text, create_search_index,
    load_existing_index, search_with_latency, compute_metrics,
    compute_paraphrase_robustness, compute_false_positive_rate,
    compute_ambiguous_metrics,
    result_cache, embedding_cache
)

app = FastAPI(title="Smart PDF RAG API", version="2.1.0")

# ── Global state ─────────────────────────────────────────────────────────────
db: dict = {"chunks": [], "index": None, "bm25": None}

loaded_index, loaded_chunks, loaded_bm25 = load_existing_index()
if loaded_index is not None:
    db["index"]  = loaded_index
    db["chunks"] = loaded_chunks
    db["bm25"]   = loaded_bm25
    print(f"[Startup] Loaded existing index with {len(loaded_chunks)} chunks.")


# ═══════════════════════════════════════════════════════════════════════════
# Upload
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_pdf(files: list[UploadFile] = File(...)):
    os.makedirs("data", exist_ok=True)
    all_new_chunks = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(400, f"{file.filename} is not a PDF.")
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        pages = extract_pdf_text(file_path)
        all_new_chunks.extend(chunk_text(pages))

    db["chunks"].extend(all_new_chunks)
    db["index"], db["bm25"], _ = create_search_index(db["chunks"])

    return {
        "message":      f"{len(files)} PDF(s) indexed successfully.",
        "total_chunks": len(db["chunks"])
    }


# ═══════════════════════════════════════════════════════════════════════════
# Search
# ═══════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    query:       str
    top_k:       int  = 3
    use_rerank:  bool = True
    candidate_k: int  = 8


@app.post("/search")
async def search(req: SearchRequest):
    if db["index"] is None:
        raise HTTPException(400, "No PDF indexed yet.")

    out = search_with_latency(
        query=req.query, index=db["index"], chunks=db["chunks"],
        bm25=db["bm25"], top_k=req.top_k,
        candidate_k=req.candidate_k, use_rerank=req.use_rerank,
    )
    return {
        "cache_hit":         out["cache_hit"],
        "latency_breakdown": out["latency"],
        "results":           out["results"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main evaluation endpoint  (accepts ground_truth text, not chunk IDs)
# ═══════════════════════════════════════════════════════════════════════════

class EvalRequest(BaseModel):
    eval_data:               list[dict]   # {"query", "ground_truth": [str], "section"?}
    top_k:                   int  = 3
    use_rerank:              bool = True
    relevance_threshold:     float = 0.72


@app.post("/evaluate")
async def evaluate(req: EvalRequest):
    if db["index"] is None:
        raise HTTPException(400, "No PDF indexed yet.")

    metrics = compute_metrics(
        eval_data           = req.eval_data,
        index               = db["index"],
        chunks              = db["chunks"],
        bm25                = db["bm25"],
        top_k               = req.top_k,
        use_rerank          = req.use_rerank,
        relevance_threshold = req.relevance_threshold,
    )
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Paraphrase Robustness (Section F)
# ═══════════════════════════════════════════════════════════════════════════

class ParaphraseRequest(BaseModel):
    topics:      list[dict]   # [{"topic": str, "variants": [str]}]
    top_k:       int  = 3
    use_rerank:  bool = True


@app.post("/evaluate/paraphrase")
async def eval_paraphrase(req: ParaphraseRequest):
    if db["index"] is None:
        raise HTTPException(400, "No PDF indexed yet.")

    return compute_paraphrase_robustness(
        paraphrase_topics = req.topics,
        index             = db["index"],
        chunks            = db["chunks"],
        bm25              = db["bm25"],
        top_k             = req.top_k,
        use_rerank        = req.use_rerank,
    )


# ═══════════════════════════════════════════════════════════════════════════
# False Positive Rate (Section G)
# ═══════════════════════════════════════════════════════════════════════════

class FPRRequest(BaseModel):
    negative_queries:    list[str]
    top_k:               int   = 3
    use_rerank:          bool  = True
    relevance_threshold: float = 0.85   # cosine similarity cutoff (0–1)


@app.post("/evaluate/false-positive-rate")
async def eval_fpr(req: FPRRequest):
    if db["index"] is None:
        raise HTTPException(400, "No PDF indexed yet.")

    return compute_false_positive_rate(
        negative_queries    = req.negative_queries,
        index               = db["index"],
        chunks              = db["chunks"],
        bm25                = db["bm25"],
        top_k               = req.top_k,
        use_rerank          = req.use_rerank,
        relevance_threshold = req.relevance_threshold,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Ambiguous Query Evaluation (Section H)
# ═══════════════════════════════════════════════════════════════════════════

class AmbiguousRequest(BaseModel):
    ambiguous_queries: list[str]
    top_k:             int  = 7
    use_rerank:        bool = True


@app.post("/evaluate/ambiguous")
async def eval_ambiguous(req: AmbiguousRequest):
    if db["index"] is None:
        raise HTTPException(400, "No PDF indexed yet.")

    return compute_ambiguous_metrics(
        ambiguous_queries = req.ambiguous_queries,
        index             = db["index"],
        chunks            = db["chunks"],
        bm25              = db["bm25"],
        top_k             = req.top_k,
        use_rerank        = req.use_rerank,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Cache management
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/cache/stats")
async def cache_stats():
    return {
        "embedding_cache_size": len(embedding_cache),
        "result_cache_size":    len(result_cache),
    }


@app.delete("/cache/clear")
async def clear_cache():
    embedding_cache.clear()
    result_cache.clear()
    return {"message": "All caches cleared."}


# ═══════════════════════════════════════════════════════════════════════════
# Index management & health
# ═══════════════════════════════════════════════════════════════════════════

@app.delete("/index/reset")
async def reset_index():
    db["index"]  = None
    db["chunks"] = []
    db["bm25"]   = None
    storage_path = "storage"
    if os.path.exists(storage_path):
        for f in os.listdir(storage_path):
            os.remove(os.path.join(storage_path, f))
    return {"message": "Index cleared."}


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "index_loaded": db["index"] is not None,
        "num_chunks":   len(db["chunks"]),
    }