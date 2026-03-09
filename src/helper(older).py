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
import hashlib
from diskcache import Cache

# ── Models ──────────────────────────────────────────────────────────────────
model    = SentenceTransformer("intfloat/e5-small", device="cuda")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")

# ── Disk-backed caches ───────────────────────────────────────────────────────
os.makedirs("cache", exist_ok=True)
embedding_cache = Cache("cache/embeddings")
result_cache    = Cache("cache/results")
RESULT_TTL      = 3600  # seconds

# A retrieved chunk is "relevant" if cosine-sim to ground truth >= this value.
RELEVANCE_THRESHOLD = 0.72


# ═══════════════════════════════════════════════════════════════════════════
# PDF helpers
# ═══════════════════════════════════════════════════════════════════════════

def extract_pdf_text(pdf_path: str) -> list[dict]:
    doc   = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc.load_page(i).get_text("text")
        if text.strip():
            pages.append({
                "pdf_name":    Path(pdf_path).name,
                "page_number": i + 1,
                "text":        text.strip()
            })
    doc.close()
    return pages


def chunk_text(pages_data: list[dict],
               chunk_size: int = 200,
               overlap:    int = 40) -> list[dict]:
    all_chunks = []
    for page in pages_data:
        paragraphs    = re.split(r'\n\s*\n', page["text"])
        current_chunk: list[str] = []
        for para in paragraphs:
            words = para.split()
            if len(current_chunk) + len(words) <= chunk_size:
                current_chunk.extend(words)
            else:
                if current_chunk:
                    all_chunks.append({
                        "pdf_name":    page["pdf_name"],
                        "page_number": page["page_number"],
                        "text":        " ".join(current_chunk)
                    })
                    current_chunk = current_chunk[-overlap:]
                current_chunk.extend(words)
        if current_chunk:
            all_chunks.append({
                "pdf_name":    page["pdf_name"],
                "page_number": page["page_number"],
                "text":        " ".join(current_chunk)
            })
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════
# Index creation / loading
# ═══════════════════════════════════════════════════════════════════════════

def create_search_index(chunks: list[dict]):
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=True, batch_size=64)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs("storage", exist_ok=True)
    faiss.write_index(index, "storage/faiss.index")
    np.save("storage/embeddings.npy", embeddings)
    with open("storage/metadata.json", "w") as f:
        json.dump(chunks, f)

    tokenized = [t.lower().split() for t in texts]
    bm25      = BM25Okapi(tokenized)
    result_cache.clear()
    return index, bm25, embeddings


def load_existing_index():
    if not os.path.exists("storage/faiss.index"):
        return None, None, None
    index = faiss.read_index("storage/faiss.index")
    with open("storage/metadata.json") as f:
        chunks = json.load(f)
    texts     = [c["text"] for c in chunks]
    tokenized = [t.lower().split() for t in texts]
    bm25      = BM25Okapi(tokenized)
    return index, chunks, bm25


# ═══════════════════════════════════════════════════════════════════════════
# Embedding cache helper
# ═══════════════════════════════════════════════════════════════════════════

def _query_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


def get_query_embedding(query: str) -> np.ndarray:
    key = _query_key(query)
    if key in embedding_cache:
        return embedding_cache[key]
    emb = model.encode([query], normalize_embeddings=True)
    embedding_cache[key] = emb
    return emb


# ═══════════════════════════════════════════════════════════════════════════
# Core search with stage-wise latency breakdown
# ═══════════════════════════════════════════════════════════════════════════

def search_with_latency(query:       str,
                        index,
                        chunks:      list[dict],
                        bm25,
                        top_k:       int  = 3,
                        candidate_k: int  = 8,
                        use_rerank:  bool = True) -> dict:
    cache_key = f"{_query_key(query)}::{top_k}::{use_rerank}"
    if cache_key in result_cache:
        cached = result_cache[cache_key]
        cached["cache_hit"] = True
        return cached

    timings: dict[str, float] = {}

    # Stage 1 – Embedding
    t0    = time.perf_counter()
    q_emb = get_query_embedding(query)
    timings["embedding_s"] = time.perf_counter() - t0

    # Stage 2 – Hybrid Retrieval
    t0 = time.perf_counter()
    dense_scores, dense_indices = index.search(q_emb, candidate_k)
    bm25_scores   = np.array(bm25.get_scores(query.lower().split()))
    bm25_top_idx  = np.argsort(bm25_scores)[::-1][:candidate_k]
    candidate_idx = list(set(dense_indices[0].tolist()) | set(bm25_top_idx.tolist()))
    candidate_idx = [i for i in candidate_idx if i != -1]
    timings["retrieval_s"] = time.perf_counter() - t0

    # Stage 3 – Re-ranking
    t0 = time.perf_counter()
    if use_rerank and candidate_idx:
        pairs         = [(query, chunks[i]["text"]) for i in candidate_idx]
        rerank_scores = reranker.predict(pairs)
        ranked        = sorted(zip(candidate_idx, rerank_scores),
                               key=lambda x: x[1], reverse=True)
    else:
        score_map = {int(dense_indices[0][j]): float(dense_scores[0][j])
                     for j in range(len(dense_indices[0]))}
        ranked    = sorted([(i, score_map.get(i, 0.0)) for i in candidate_idx],
                           key=lambda x: x[1], reverse=True)
    timings["reranking_s"] = time.perf_counter() - t0
    timings["total_s"]     = sum(timings.values())

    results = [
        {
            "score":    float(s),
            "text":     chunks[i]["text"],
            "page":     chunks[i]["page_number"],
            "pdf_name": chunks[i]["pdf_name"],
            "chunk_id": i
        }
        for i, s in ranked[:top_k]
    ]

    payload = {"latency": timings, "results": results, "cache_hit": False}
    result_cache.set(cache_key, payload, expire=RESULT_TTL)
    return payload


# ═══════════════════════════════════════════════════════════════════════════
# Semantic relevance helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sem_sim(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using the loaded model."""
    embs = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def _chunk_is_relevant(chunk_text: str, ground_truths: list[str],
                       threshold: float = RELEVANCE_THRESHOLD) -> bool:
    """True if chunk is semantically similar to ANY ground truth answer."""
    for gt in ground_truths:
        if _sem_sim(chunk_text, gt) >= threshold:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Entity Coverage Score
# ═══════════════════════════════════════════════════════════════════════════

def _extract_entities(text: str) -> list[str]:
    """
    Lightweight entity extractor — no external NLP dependency needed.
    Captures: numbers+units, capitalised proper nouns, acronyms.
    """
    nums   = re.findall(
        r'\b[\d,]+(?:\.\d+)?\s*(?:sq\.?\s*ft\.?|acres?|floors?|km|m|%|COP)?\b',
        text, re.IGNORECASE)
    caps   = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    abbrev = re.findall(r'\b[A-Z]{2,}\b', text)
    return list({e.strip().lower() for e in nums + caps + abbrev if e.strip()})


def entity_coverage_score(retrieved_text: str, ground_truth: str) -> float:
    """Fraction of key entities from ground_truth present in retrieved_text."""
    required = _extract_entities(ground_truth)
    if not required:
        return 1.0
    rt_lower = retrieved_text.lower()
    covered  = sum(1 for e in required if e in rt_lower)
    return round(covered / len(required), 4)


# ═══════════════════════════════════════════════════════════════════════════
# Hallucination Rate — retrieval mismatch (no LLM required)
#
# Definition: a "hallucination" occurs when the top-1 retrieved chunk is NOT
# semantically relevant to the ground truth answer (i.e. the retrieval
# failed for that query).  This is purely retrieval-based — no generative
# model is involved, consistent with the system design.
#
# Hallucination Rate = % of queries where top-1 chunk sim < threshold
# ═══════════════════════════════════════════════════════════════════════════

def is_retrieval_mismatch(top1_chunk_text: str,
                          ground_truths:   list[str],
                          threshold:       float = RELEVANCE_THRESHOLD) -> bool:
    """
    Returns True when the top-1 retrieved chunk does NOT match the ground
    truth (i.e. cosine-sim of top-1 chunk vs every ground truth < threshold).
    In a retrieval-only RAG system this IS the hallucination signal because
    the returned chunk itself is the answer — a wrong chunk = wrong answer.
    """
    return not _chunk_is_relevant(top1_chunk_text, ground_truths, threshold)


# ═══════════════════════════════════════════════════════════════════════════
# Full evaluation — all 11 required metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(eval_data:  list[dict],
                    index,
                    chunks:     list[dict],
                    bm25,
                    top_k:      int  = 3,
                    use_rerank: bool = True,
                    relevance_threshold: float = RELEVANCE_THRESHOLD) -> dict:
    """
    eval_data items must have:
        "query"        : str
        "ground_truth" : list[str]   ← free-text expected answers
        "section"      : str         ← optional, for per-section breakdown

    Covered metrics
    ───────────────
     1. Recall@1
     2. Recall@3
     3. Recall@K          (K = top_k)
     4. Top-1 Accuracy    (= Recall@1)
     5. Top-3 Accuracy    (= Recall@3)
     6. MRR
     7. nDCG@K
     8. Entity Coverage Score
     9. Hallucination Rate  (retrieval mismatch: top-1 chunk sim < threshold)
    10. False Positive Rate → computed separately via compute_false_positive_rate()
    11. Stage-wise Latency  → embedded in every search result
    """
    recall_1 = recall_3 = recall_k = 0
    mrr_sum = ndcg_sum = 0.0
    entity_scores:      list[float] = []
    hallucination_flags: list[bool] = []
    latencies:          list[float] = []
    per_query_results:  list[dict]  = []
    section_buckets:    dict        = {}

    for item in eval_data:
        query         = item["query"]
        ground_truths = item.get("ground_truth", [])
        section       = item.get("section", "unknown")
        if not ground_truths:
            continue

        out          = search_with_latency(query, index, chunks, bm25,
                                           top_k=top_k, use_rerank=use_rerank)
        latencies.append(out["latency"]["total_s"])
        ranked_texts = [r["text"] for r in out["results"]]
        top1_text    = ranked_texts[0] if ranked_texts else ""

        # Build relevance mask for top-K results
        rel_mask = [
            _chunk_is_relevant(t, ground_truths, relevance_threshold)
            for t in ranked_texts
        ]
        first_rel = next((i + 1 for i, r in enumerate(rel_mask) if r), 0)

        # ── Recall / Accuracy ────────────────────────────────────────────
        hit1 = int(bool(rel_mask) and rel_mask[0])
        hit3 = int(any(rel_mask[:3]))
        hitK = int(any(rel_mask[:top_k]))
        recall_1 += hit1
        recall_3 += hit3
        recall_k += hitK

        # ── MRR ─────────────────────────────────────────────────────────
        mrr_q = (1.0 / first_rel) if first_rel else 0.0
        mrr_sum += mrr_q

        # ── nDCG@K ──────────────────────────────────────────────────────
        dcg = idcg = 0.0
        for rank, rel in enumerate(rel_mask[:top_k], 1):
            if rel:
                dcg += 1.0 / np.log2(rank + 1)
        ideal = min(sum(rel_mask), top_k)
        for rank in range(1, ideal + 1):
            idcg += 1.0 / np.log2(rank + 1)
        ndcg_q = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_sum += ndcg_q

        # ── Entity Coverage ──────────────────────────────────────────────
        ecs = entity_coverage_score(top1_text, ground_truths[0]) if top1_text else 0.0
        entity_scores.append(ecs)

        # ── Hallucination = retrieval mismatch on top-1 chunk ────────────
        hall = is_retrieval_mismatch(top1_text, ground_truths, relevance_threshold) \
               if top1_text else True
        hallucination_flags.append(hall)

        # ── Per-query record ─────────────────────────────────────────────
        per_query_results.append({
            "query":             query,
            "section":           section,
            "relevant_rank":     first_rel,
            "recall@1":          hit1,
            "recall@3":          hit3,
            f"recall@{top_k}":   hitK,
            "mrr":               round(mrr_q, 4),
            f"ndcg@{top_k}":     round(ndcg_q, 4),
            "entity_coverage":   ecs,
            "hallucination":     hall,
            "latency_ms":        round(out["latency"]["total_s"] * 1000, 1),
            "top1_text_snippet": top1_text[:200],
            "ground_truth":      ground_truths[0],
        })

        # ── Section accumulation ─────────────────────────────────────────
        if section not in section_buckets:
            section_buckets[section] = {
                "total": 0, "r1": 0, "r3": 0, "rk": 0,
                "mrr": 0.0, "ndcg": 0.0, "ecs": [], "hall": 0
            }
        sb = section_buckets[section]
        sb["total"] += 1
        sb["r1"]    += hit1
        sb["r3"]    += hit3
        sb["rk"]    += hitK
        sb["mrr"]   += mrr_q
        sb["ndcg"]  += ndcg_q
        sb["ecs"].append(ecs)
        sb["hall"]  += int(hall)

    n = len(per_query_results)
    if n == 0:
        return {"error": "No valid eval items were processed."}

    # ── Section summary ──────────────────────────────────────────────────
    section_summary = {}
    for sec, sb in section_buckets.items():
        t = sb["total"]
        section_summary[sec] = {
            "num_queries":         t,
            "Recall@1":            round(sb["r1"] / t, 4),
            "Recall@3":            round(sb["r3"] / t, 4),
            f"Recall@{top_k}":     round(sb["rk"] / t, 4),
            "MRR":                 round(sb["mrr"] / t, 4),
            f"nDCG@{top_k}":       round(sb["ndcg"] / t, 4),
            "Avg_Entity_Coverage": round(sum(sb["ecs"]) / len(sb["ecs"]), 4),
            "Hallucination_Rate":  round(sb["hall"] / t, 4),
        }

    return {
        # ── 1-7: Retrieval metrics ───────────────────────────────────────
        "num_queries":          n,
        "Recall@1":             round(recall_1 / n, 4),
        "Recall@3":             round(recall_3 / n, 4),
        f"Recall@{top_k}":      round(recall_k / n, 4),
        "Top1_Accuracy":        round(recall_1 / n, 4),
        "Top3_Accuracy":        round(recall_3 / n, 4),
        "MRR":                  round(mrr_sum / n, 4),
        f"nDCG@{top_k}":        round(ndcg_sum / n, 4),
        # ── 8-9: Answer quality ─────────────────────────────────────────
        "Entity_Coverage_Score":  round(sum(entity_scores) / len(entity_scores), 4),
        "Hallucination_Rate":     round(sum(hallucination_flags) / len(hallucination_flags), 4),
        # ── 11: Latency ─────────────────────────────────────────────────
        "avg_latency_s":          round(float(np.mean(latencies)), 4),
        "p95_latency_s":          round(float(np.percentile(latencies, 95)), 4),
        # ── Breakdown ───────────────────────────────────────────────────
        "section_breakdown":      section_summary,
        "per_query_results":      per_query_results,
        "config": {
            "top_k":               top_k,
            "use_rerank":          use_rerank,
            "relevance_threshold": relevance_threshold,
            "hallucination_definition": (
                "retrieval mismatch: top-1 chunk cosine-sim < relevance_threshold"
            ),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# Paraphrase Robustness (Section F)
# ═══════════════════════════════════════════════════════════════════════════

def compute_paraphrase_robustness(paraphrase_topics: list[dict],
                                  index, chunks: list[dict], bm25,
                                  top_k: int = 3,
                                  use_rerank: bool = True) -> dict:
    """
    Measures robustness by computing pairwise semantic similarity between
    the top-1 retrieved chunks across all variants of the same question.

    Why not Jaccard on chunk IDs?
    ─────────────────────────────
    Different phrasings often retrieve different but semantically equivalent
    chunks (e.g. two chunks on the same page with the same answer).
    Jaccard penalises this as inconsistency even though the answer is correct.

    New approach — pairwise cosine similarity of top-1 chunk texts:
    ───────────────────────────────────────────────────────────────
    consistency(topic) = mean cosine_sim(top1_i, top1_j)
                         for all pairs (i, j) of variants

    Score of 1.0 → every rephrasing retrieves a semantically identical chunk.
    Score of 0.0 → retrieved chunks share no semantic content across variants.
    """
    from itertools import combinations

    topic_results = []
    all_scores    = []

    for td in paraphrase_topics:
        topic    = td["topic"]
        variants = td["variants"]

        top1_texts = []
        rows       = []

        for q in variants:
            res = search_with_latency(q, index, chunks, bm25,
                                      top_k=top_k, use_rerank=use_rerank)
            if res["results"]:
                top1_text = res["results"][0]["text"]
                chunk_ids = [r["chunk_id"] for r in res["results"]]
            else:
                top1_text = ""
                chunk_ids = []

            top1_texts.append(top1_text)
            rows.append({
                "query":          q,
                "topk_chunk_ids": chunk_ids,
                "snippet":        top1_text[:200]
            })

        # Pairwise cosine similarity between top-1 chunk texts
        overlaps = []
        for (i, text_a), (j, text_b) in combinations(enumerate(top1_texts), 2):
            if not text_a and not text_b:
                overlaps.append(1.0)   # both empty → consistently returning nothing
            elif not text_a or not text_b:
                overlaps.append(0.0)   # one empty, one not → inconsistent
            else:
                overlaps.append(_sem_sim(text_a, text_b))

        consistency = round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0
        all_scores.append(consistency)

        topic_results.append({
            "topic":       topic,
            "consistency": consistency,
            "variants":    rows,
        })

    return {
        "Paraphrase_Robustness_Score": round(sum(all_scores) / len(all_scores), 4)
                                       if all_scores else 0.0,
        "num_topics": len(paraphrase_topics),
        "topics":     topic_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# False Positive Rate (Section G — Metric 10)
# ═══════════════════════════════════════════════════════════════════════════

def compute_false_positive_rate(negative_queries: list[str],
                                index, chunks: list[dict], bm25,
                                top_k: int = 3,
                                use_rerank: bool = True,
                                relevance_threshold: float = 0.85) -> dict:
    """
    Metric 10: False Positive Rate for OOS queries.

    False positive occurs when a retrieved chunk is NOT semantically
    relevant to the query (cosine similarity < threshold).
    """

    rows = []
    fp_count = 0

    for q in negative_queries:

        res = search_with_latency(q, index, chunks, bm25,
                                  top_k=top_k, use_rerank=use_rerank)

        if res["results"]:
            text = res["results"][0]["text"]
            sim  = _sem_sim(q, text)
        else:
            text = ""
            sim  = 0.0

        is_fp = sim >= relevance_threshold
        if is_fp:
            fp_count += 1

        rows.append({
            "query": q,
            "semantic_similarity": round(sim, 4),
            "false_positive": is_fp,
            "snippet": text[:200]
        })

    n = len(negative_queries)
    fpr = round(fp_count / n, 4) if n else 0.0

    return {
        "False_Positive_Rate": fpr,
        "false_positive_count": fp_count,
        "total_negative_queries": n,
        "relevance_threshold": relevance_threshold,
        "rows": rows
    }


# ═══════════════════════════════════════════════════════════════════════════
# Ambiguous Query Evaluation (Section H)
#
# Ambiguous queries (e.g. "What is the total area?") have no single correct
# answer because they don't specify which property.  The ideal system
# behaviour is to return results that SPAN multiple documents/properties,
# surfacing the ambiguity rather than silently committing to one source.
#
# Key metric: Multi-Source Coverage
#   = % of ambiguous queries where top-K results come from ≥ 2 distinct PDFs
#   Higher is better — it means the system is not blindly anchoring to one doc.
# ═══════════════════════════════════════════════════════════════════════════

def compute_ambiguous_metrics(ambiguous_queries: list[str],
                              index, chunks: list[dict], bm25,
                              top_k:      int  = 7,
                              use_rerank: bool = True) -> dict:
    """
    Evaluates system behaviour on intentionally ambiguous queries.

    For each query returns:
      - distinct_sources : number of unique PDFs in top-K results
      - multi_source     : True if results span ≥ 2 PDFs (desired behaviour)
      - results          : the actual retrieved snippets for manual inspection

    Aggregate metric:
      - Multi_Source_Coverage : % of queries with ≥ 2 distinct source PDFs
    """
    rows              = []
    multi_source_count = 0

    for q in ambiguous_queries:
        res     = search_with_latency(q, index, chunks, bm25,
                                      top_k=top_k, use_rerank=use_rerank)
        results = res.get("results", [])

        distinct_sources = list({r["pdf_name"] for r in results})
        is_multi         = len(distinct_sources) >= 2
        if is_multi:
            multi_source_count += 1

        rows.append({
            "query":            q,
            "distinct_sources": distinct_sources,
            "num_sources":      len(distinct_sources),
            "multi_source":     is_multi,
            "results": [
                {
                    "pdf_name": r["pdf_name"],
                    "page":     r["page"],
                    "score":    round(r["score"], 4),
                    "snippet":  r["text"][:200],
                }
                for r in results
            ],
        })

    n = len(ambiguous_queries)
    return {
        "total_ambiguous_queries": n,
        "Multi_Source_Coverage":   round(multi_source_count / n, 4) if n else 0.0,
        "multi_source_count":      multi_source_count,
        "note": (
            "Multi_Source_Coverage = % of queries where top-K results span "
            "≥ 2 distinct PDFs. Higher means the system correctly surfaces "
            "ambiguity instead of anchoring to a single document."
        ),
        "rows": rows,
    }
