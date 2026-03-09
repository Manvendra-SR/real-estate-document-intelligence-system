import streamlit as st
import requests
import json
import time
import pandas as pd

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="📄 Smart PDF RAG", layout="wide")
st.title("📄 Smart PDF Search & Retrieval  v2.1")

tab_upload, tab_search, tab_eval, tab_cache = st.tabs(
    ["📤 Upload", "🔍 Search", "📊 Full Evaluate", "⚙️ Cache"]
)


# ═══════════════════════════════════════════════════════════════════════════
# Tab 1 – Upload
# ═══════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("Upload & Index PDFs")
    uploaded_files = st.file_uploader("Choose one or more PDFs",
                                       type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Process & Index PDFs"):
        files = [("files", (f.name, f.getvalue(), "application/pdf"))
                 for f in uploaded_files]
        with st.spinner("Indexing …"):
            res = requests.post(f"{BASE_URL}/upload", files=files)
        if res.status_code == 200:
            d = res.json()
            st.success(f"✅ {d['message']}  |  Chunks: **{d['total_chunks']}**")
        else:
            st.error(f"Upload failed: {res.text}")

    try:
        h = requests.get(f"{BASE_URL}/health", timeout=3).json()
        if h["index_loaded"]:
            st.info(f"ℹ️ Active index: **{h['num_chunks']}** chunks ready.")
    except Exception:
        st.warning("⚠️ Backend not reachable at localhost:8000")

    st.markdown("---")
    st.subheader("⚠️ Index Management")
    if st.button("🗑️ Reset Entire Index"):
        res = requests.delete(f"{BASE_URL}/index/reset")

        if res.status_code == 200:
            st.success(res.json()["message"])
        else:
            st.error("Failed to reset index.")


# ═══════════════════════════════════════════════════════════════════════════
# Tab 2 – Search
# ═══════════════════════════════════════════════════════════════════════════
with tab_search:
    st.subheader("Ask a Question")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("Your question:")
    with col2:
        top_k = st.number_input("Top-K results", 1, 10, 3)
    with col3:
        use_rerank = st.checkbox("Re-ranking", value=True)

    if query and st.button("Search"):
        payload = {"query": query, "top_k": int(top_k), "use_rerank": use_rerank}
        with st.spinner("Searching …"):
            res = requests.post(f"{BASE_URL}/search", json=payload)

        if res.status_code == 200:
            data = res.json()
            lat  = data["latency_breakdown"]

            if data.get("cache_hit"):
                st.success("⚡ **Cache hit!** Result served instantly.")
            else:
                st.info("🔄 Fresh search performed.")

            st.markdown("### ⏱ Stage-wise Latency Breakdown")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Embedding",   f"{lat['embedding_s']*1000:.1f} ms")
            c2.metric("Retrieval",   f"{lat['retrieval_s']*1000:.1f} ms")
            c3.metric("Re-ranking",  f"{lat['reranking_s']*1000:.1f} ms")
            c4.metric("Total",       f"{lat['total_s']*1000:.1f} ms")

            st.markdown("### 📑 Results")
            for item in data["results"]:
                with st.expander(
                    f"📄 {item['pdf_name']}  |  Page {item['page']}  "
                    f"(score: {item['score']:.3f})"
                ):
                    st.write(item["text"])
        else:
            st.error(res.json().get("detail", "Search failed."))


# ═══════════════════════════════════════════════════════════════════════════
# Tab 3 – Full Evaluate
# ═══════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("📊 Full Evaluation — All 11 Required Metrics")

    st.markdown("""
    Upload your **eval dataset JSON** or paste it below.
    Each entry must have:
    ```json
    {"query": "...", "ground_truth": ["expected answer text"], "section": "A_..."}
    ```
    This supports the `eval_dataset_modified.json` format directly.
    """)

    # ── Sub-tabs for the 3 eval types ─────────────────────────────────────
    ev_main, ev_para, ev_fpr, ev_ambig = st.tabs(
        ["📋 Main Eval (A–E)", "🔁 Paraphrase Robustness (F)",
         "🚫 False Positive Rate (G)", "❓ Ambiguous Queries (H)"]
    )

    # ────────────────────────────────────────────────────────────────────
    # Sub-tab 3a: Main eval (Sections A–E)
    # ────────────────────────────────────────────────────────────────────
    with ev_main:
        st.markdown("**Covers Metrics 1–9 + 11:** Recall@K, Top-K Accuracy, MRR, nDCG, "
                    "Entity Coverage, Hallucination Rate, Latency")

        uploaded_eval = st.file_uploader("Upload eval JSON file", type="json",
                                          key="eval_upload")
        eval_text = st.text_area("Or paste eval JSON here:", height=120,
                                  key="eval_text_input",
                                  placeholder='[{"query":"...", "ground_truth":["..."], "section":"A_..."}]')

        col_a, col_b, col_c = st.columns(3)
        ev_k        = col_a.number_input("Top-K", 1, 10, 3, key="ev_k")
        ev_rr       = col_b.checkbox("Re-ranking", True, key="ev_rr")
        ev_thresh   = col_c.number_input("Relevance threshold", 0.1, 0.9, 0.72,
                                          step=0.05, key="ev_thresh")

        if st.button("▶️ Run Evaluation", key="run_eval"):
            # Load data from file or text box
            raw_text = None
            if uploaded_eval:
                raw_text = uploaded_eval.read().decode()
            elif eval_text.strip():
                raw_text = eval_text.strip()

            if not raw_text:
                st.warning("Please upload a file or paste JSON data.")
            else:
                try:
                    eval_data = json.loads(raw_text)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
                    eval_data = None

                if eval_data:
                    payload = {
                        "eval_data":           eval_data,
                        "top_k":               int(ev_k),
                        "use_rerank":          ev_rr,
                        "relevance_threshold": float(ev_thresh),
                    }
                    with st.spinner(
                        f"Running full evaluation … ({len(eval_data)} queries)"
                    ):
                        res = requests.post(f"{BASE_URL}/evaluate", json=payload,
                                            timeout=600)

                    if res.status_code == 200:
                        m = res.json()
                        st.success(f"✅ Evaluation complete — {m['num_queries']} queries processed.")

                        # ── Top-level metric cards ──────────────────────────────
                        st.markdown("### 🏆 Overall Metrics")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Recall@1",
                                  f"{m['Recall@1']*100:.1f}%",
                                  delta="✅ ≥75%" if m['Recall@1'] >= 0.75 else "⚠️ <75% target")
                        c2.metric("Recall@3",
                                  f"{m['Recall@3']*100:.1f}%",
                                  delta="✅ ≥90%" if m['Recall@3'] >= 0.90 else "⚠️ <90% target")
                        c3.metric("MRR",   f"{m['MRR']:.4f}")
                        c4.metric(f"nDCG@{int(ev_k)}", f"{m.get(f'nDCG@{int(ev_k)}', 0):.4f}")

                        c5, c6, c7, c8 = st.columns(4)
                        c5.metric("Top-1 Accuracy", f"{m['Top1_Accuracy']*100:.1f}%")
                        c6.metric("Top-3 Accuracy", f"{m['Top3_Accuracy']*100:.1f}%")
                        c7.metric("Entity Coverage",
                                  f"{m['Entity_Coverage_Score']*100:.1f}%",
                                  help="Fraction of key entities from ground truth found in top-1 chunk")
                        c8.metric("Hallucination Rate",
                                  f"{m['Hallucination_Rate']*100:.1f}%",
                                  delta="✅ low" if m['Hallucination_Rate'] <= 0.1 else "⚠️ high",
                                  help="% of queries where top-1 chunk is NOT semantically relevant to ground truth (retrieval mismatch). Lower is better.")

                        c9, c10 = st.columns(2)
                        c9.metric("Avg Latency",  f"{m['avg_latency_s']*1000:.1f} ms")
                        c10.metric("P95 Latency", f"{m['p95_latency_s']*1000:.1f} ms")

                        # ── Section breakdown ───────────────────────────────────
                        if "section_breakdown" in m and m["section_breakdown"]:
                            st.markdown("### 📂 Section-wise Breakdown")
                            rows = []
                            for sec, s in m["section_breakdown"].items():
                                rows.append({
                                    "Section":      sec,
                                    "Queries":      s["num_queries"],
                                    "Recall@1":     f"{s['Recall@1']*100:.0f}%",
                                    "Recall@3":     f"{s['Recall@3']*100:.0f}%",
                                    "MRR":          f"{s['MRR']:.3f}",
                                    f"nDCG@{int(ev_k)}": f"{s.get(f'nDCG@{int(ev_k)}', 0):.3f}",
                                    "Entity Cov.":  f"{s['Avg_Entity_Coverage']*100:.0f}%",
                                    "Hall. Rate":   f"{s['Hallucination_Rate']*100:.0f}%",
                                })
                            st.dataframe(pd.DataFrame(rows), use_container_width=True)

                        # ── Per-query results ───────────────────────────────────
                        if st.checkbox("Show per-query results table"):
                            pq = m.get("per_query_results", [])
                            if pq:
                                pq_df = pd.DataFrame([{
                                    "Query":        r["query"][:70],
                                    "Section":      r["section"],
                                    "Rank":         r["relevant_rank"],
                                    "Recall@1":     "✅" if r["recall@1"] else "❌",
                                    "MRR":          r["mrr"],
                                    "Entity Cov.":  f"{r['entity_coverage']*100:.0f}%",
                                    "Hallucinated": "⚠️ Yes" if r["hallucination"] else "✅ No",
                                    "Latency ms":   r["latency_ms"],
                                } for r in pq])
                                st.dataframe(pq_df, use_container_width=True)

                        # ── Download full report ────────────────────────────────
                        st.download_button(
                            "⬇️ Download full JSON report",
                            data=json.dumps(m, indent=2),
                            file_name="eval_report.json",
                            mime="application/json"
                        )
                    else:
                        st.error(res.json().get("detail", "Evaluation failed."))

    # ────────────────────────────────────────────────────────────────────
    # Sub-tab 3b: Paraphrase Robustness (Section F)
    # ────────────────────────────────────────────────────────────────────
    with ev_para:
        st.markdown("""
        **Metric 8 — Paraphrase Robustness Score**

        Paste the `F_Paraphrase_Robustness` section from `eval_questions.json`:
        ```json
        [{"topic": "Residential vs Commercial", "variants": ["Q1", "Q2", "Q3"]}, ...]
        ```
        """)

        para_upload = st.file_uploader("Upload eval_questions.json", type="json",
                                        key="para_upload")
        para_text   = st.text_area("Or paste Section F JSON:", height=120,
                                    key="para_text")
        para_k      = st.number_input("Top-K", 1, 10, 3, key="para_k")
        para_rr     = st.checkbox("Re-ranking", True, key="para_rr")

        if st.button("▶️ Run Paraphrase Robustness", key="run_para"):
            raw = None
            if para_upload:
                full = json.loads(para_upload.read().decode())
                # Accept full eval_questions.json or just section F
                topics = full.get("sections", {}).get("F_Paraphrase_Robustness",
                         full if isinstance(full, list) else [])
                raw = topics
            elif para_text.strip():
                raw = json.loads(para_text.strip())

            if raw:
                payload = {"topics": raw, "top_k": int(para_k), "use_rerank": para_rr}
                with st.spinner("Running paraphrase robustness check …"):
                    res = requests.post(f"{BASE_URL}/evaluate/paraphrase",
                                        json=payload, timeout=120)
                if res.status_code == 200:
                    pr = res.json()
                    st.metric("Paraphrase Robustness Score",
                              f"{pr['Paraphrase_Robustness_Score']:.4f}",
                              help="1.0 = perfectly consistent across all rephrasings")
                    rows = [{
                        "Topic":       t["topic"],
                        "Consistency": f"{t['consistency']:.4f}",
                    } for t in pr["topics"]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    if st.checkbox("Show variant-level details", key="para_details"):
                        for t in pr["topics"]:
                            st.markdown(f"**{t['topic']}**")
                            for v in t["variants"]:
                                st.write(f"- `{v['query'][:80]}` → chunk {v['chunk_id']}")
                else:
                    st.error("Paraphrase eval failed.")

    # ────────────────────────────────────────────────────────────────────
    # Sub-tab 3c: False Positive Rate (Section G)
    # ────────────────────────────────────────────────────────────────────
    with ev_fpr:
        st.markdown("""
        **Metric 10 — False Positive Rate (Negative / Adversarial Queries)**

        Paste the `G_Negative_Adversarial` list from `eval_questions.json`
        or type queries that have NO answer in your documents.
        Lower FPR is better (ideally 0%).
        """)

        fpr_upload = st.file_uploader(
            "Upload eval_questions.json",
            type="json",
            key="fpr_upload"
        )

        fpr_text = st.text_area(
            "Or paste JSON list of negative queries:",
            height=100,
            key="fpr_text",
            placeholder='["Which property has a helipad?", "Is any property in Mumbai?"]'
        )

        col_fk, col_frr, col_fthr = st.columns(3)

        fpr_k = col_fk.number_input(
            "Top-K",
            min_value=1,
            max_value=10,
            value=3,
            key="fpr_k"
        )

        fpr_rr = col_frr.checkbox(
            "Re-ranking",
            True,
            key="fpr_rr"
        )

        fpr_threshold = col_fthr.number_input(
            "Relevance similarity threshold",
            value=0.85,
            step=0.05,
            key="fpr_thr"
        )

        if st.button("▶️ Run FPR Test", key="run_fpr"):

            neg_queries = None

            if fpr_upload:
                full = json.loads(fpr_upload.read().decode())
                neg_queries = full.get(
                    "sections", {}
                ).get(
                    "G_Negative_Adversarial",
                    full if isinstance(full, list) else []
                )

            elif fpr_text.strip():
                neg_queries = json.loads(fpr_text.strip())

            if neg_queries:

                payload = {
                    "negative_queries": neg_queries,
                    "top_k": int(fpr_k),
                    "use_rerank": fpr_rr,
                    "relevance_threshold": float(fpr_threshold),
                }

                with st.spinner("Running false-positive rate test …"):
                    res = requests.post(
                        f"{BASE_URL}/evaluate/false-positive-rate",
                        json=payload,
                        timeout=120
                    )

                if res.status_code == 200:

                    fp = res.json()

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "False Positive Rate",
                        f"{fp['False_Positive_Rate'] * 100:.1f}%",
                        delta="✅ low" if fp["False_Positive_Rate"] <= 0.1 else "⚠️ high"
                    )

                    col2.metric(
                        "FP Count",
                        f"{fp['false_positive_count']} / {fp['total_negative_queries']}"
                    )

                    rows = [{
                        "Query": r["query"],
                        "Similarity": f"{r['semantic_similarity']:.3f}",
                        "False Positive": r["false_positive"],
                    } for r in fp["rows"]]

                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                else:
                    st.error("FPR test failed.")

    # ────────────────────────────────────────────────────────────────────
    # Sub-tab 3d: Ambiguous Queries (Section H)
    # ────────────────────────────────────────────────────────────────────
    with ev_ambig:
        st.markdown("""
        **Section H — Ambiguous Query Behaviour**

        These queries (e.g. *"What is the total area?"*, *"How many floors does it have?"*)
        intentionally omit which property they refer to.

        **Desired behaviour:** the system should return results from **multiple
        documents**, surfacing the ambiguity rather than silently picking one property.

        **Key metric — Multi-Source Coverage:**
        % of queries where top-K results span ≥ 2 distinct PDFs. Higher is better.

        Paste the `H_Ambiguous` list from `eval_questions.json` or type your own queries.
        """)

        ambig_upload = st.file_uploader("Upload eval_questions.json", type="json",
                                         key="ambig_upload")
        ambig_text   = st.text_area("Or paste JSON list of ambiguous queries:", height=100,
                                     key="ambig_text",
                                     placeholder='["What is the total area?", "How many floors does it have?"]')
        col_ak, col_arr = st.columns(2)
        ambig_k  = col_ak.number_input("Top-K", 1, 10, 3, key="ambig_k")
        ambig_rr = col_arr.checkbox("Re-ranking", True, key="ambig_rr")

        if st.button("▶️ Run Ambiguous Query Eval", key="run_ambig"):
            ambig_queries = None
            if ambig_upload:
                full = json.loads(ambig_upload.read().decode())
                ambig_queries = full.get("sections", {}).get("H_Ambiguous",
                                full if isinstance(full, list) else [])
            elif ambig_text.strip():
                ambig_queries = json.loads(ambig_text.strip())

            if ambig_queries:
                payload = {
                    "ambiguous_queries": ambig_queries,
                    "top_k":             int(ambig_k),
                    "use_rerank":        ambig_rr,
                }
                with st.spinner("Running ambiguous query evaluation …"):
                    res = requests.post(f"{BASE_URL}/evaluate/ambiguous",
                                        json=payload, timeout=120)

                if res.status_code == 200:
                    am = res.json()

                    col1, col2 = st.columns(2)
                    col1.metric(
                        "Multi-Source Coverage",
                        f"{am['Multi_Source_Coverage']*100:.1f}%",
                        delta="✅ high" if am['Multi_Source_Coverage'] >= 0.6 else "⚠️ low",
                        help=am["note"]
                    )
                    col2.metric(
                        "Queries spanning ≥ 2 PDFs",
                        f"{am['multi_source_count']} / {am['total_ambiguous_queries']}"
                    )

                    # ── Summary table ────────────────────────────────────
                    st.markdown("### Per-query Results")
                    summary_rows = [{
                        "Query":           row["query"],
                        "# Sources":       row["num_sources"],
                        "PDFs returned":   ", ".join(row["distinct_sources"]),
                        "Multi-source?":   "✅ Yes" if row["multi_source"] else "⚠️ No",
                    } for row in am["rows"]]
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                    # ── Expandable per-query detail ──────────────────────
                    for row in am["rows"]:
                        sources_str = ", ".join(row["distinct_sources"])
                        label = (
                            f"{'✅' if row['multi_source'] else '⚠️'}  "
                            f"`{row['query']}`  →  "
                            f"{row['num_sources']} source(s): {sources_str}"
                        )
                        with st.expander(label):
                            for r in row["results"]:
                                st.markdown(
                                    f"**{r['pdf_name']}** | Page {r['page']} "
                                    f"| Score {r['score']}"
                                )
                                st.caption(r["snippet"])

                    # ── Downloads ────────────────────────────────────────
                    st.markdown("### ⬇️ Download Results")
                    dl_col1, dl_col2 = st.columns(2)

                    # Full JSON report
                    dl_col1.download_button(
                        label="📄 Download full JSON report",
                        data=json.dumps(am, indent=2),
                        file_name="ambiguous_eval_report.json",
                        mime="application/json"
                    )

                    # Summary CSV
                    csv_data = pd.DataFrame(summary_rows).to_csv(index=False)
                    dl_col2.download_button(
                        label="📊 Download summary CSV",
                        data=csv_data,
                        file_name="ambiguous_eval_summary.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Ambiguous query eval failed.")


# ═══════════════════════════════════════════════════════════════════════════
# Tab 4 – Cache
# ═══════════════════════════════════════════════════════════════════════════
with tab_cache:
    st.subheader("Cache Management")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📊 Refresh Cache Stats"):
            res = requests.get(f"{BASE_URL}/cache/stats")
            if res.status_code == 200:
                s = res.json()
                st.metric("Embedding cache entries", s["embedding_cache_size"])
                st.metric("Result cache entries",    s["result_cache_size"])
    with col_b:
        if st.button("🗑️ Clear All Caches"):
            res = requests.delete(f"{BASE_URL}/cache/clear")
            if res.status_code == 200:
                st.success(res.json()["message"])

    st.markdown("""
    ### How Caching Works
    | Layer | Key | TTL | Benefit |
    |---|---|---|---|
    | **Embedding cache** | MD5(query) | Persistent | Avoids re-encoding repeated queries |
    | **Result cache** | MD5(query)+top_k+rerank | 1 hour | Full search bypass on repeats |

    Repeated queries after the first call typically reduce latency by **80–98%**.
    """)
