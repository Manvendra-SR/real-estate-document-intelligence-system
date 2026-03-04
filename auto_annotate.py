"""
auto_annotate.py
================
After indexing your PDFs, run this script to automatically find the most
relevant chunk IDs for every factual question (Sections A–E).

It produces:  eval_dataset.json
which can be directly POSTed to /evaluate or used in benchmark.py.

Usage:
    python auto_annotate.py --url http://localhost:8000 --top_k 1

Then manually review eval_dataset.json and correct any wrong chunk IDs
before running the full evaluation.
"""

import argparse
import json
import requests

BASE_URL       = "http://localhost:8000"
QUESTIONS_FILE = "eval_questions.json"
OUT_FILE       = "eval_dataset.json"

SKIP_SECTIONS  = {"F_Paraphrase_Robustness", "G_Negative_Adversarial", "H_Ambiguous"}


def search(query: str, top_k: int, url: str) -> dict:
    r = requests.post(f"{url}/search", json={"query": query, "top_k": top_k,
                                              "use_rerank": True}, timeout=60)
    return r.json() if r.status_code == 200 else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",   default=BASE_URL)
    parser.add_argument("--top_k", type=int, default=1,
                        help="Number of top chunks to assign per question")
    args = parser.parse_args()

    with open(QUESTIONS_FILE) as f:
        qs = json.load(f)["sections"]

    eval_data = []
    total = 0

    for section_key, questions in qs.items():
        if section_key in SKIP_SECTIONS:
            continue
        print(f"\n[{section_key}]")
        for q in questions:
            res     = search(q, args.top_k, args.url)
            results = res.get("results", [])
            chunk_ids = [r["chunk_id"] for r in results]
            eval_data.append({
                "query":               q,
                "section":             section_key,
                "relevant_chunk_ids":  chunk_ids,
                "top_texts":           [r["text"][:120] for r in results],
            })
            total += 1
            print(f"  ✓  chunk_ids={chunk_ids}  | {q[:80]}")

    with open(OUT_FILE, "w") as f: 
        json.dump(eval_data, f, indent=2)

    print(f"\n✅  Saved {total} annotated questions → {OUT_FILE}")
    print("👉  Review chunk IDs, then run:  python benchmark.py")


if __name__ == "__main__":
    main()
