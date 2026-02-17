from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from src.helper import extract_pdf_text, chunk_text, create_search_index, model, reranker, np, time, load_existing_index

app = FastAPI()

# Global state to store the index for the uploaded file
db = {"chunks": [], "index": None, "bm25": None}

loaded_index, loaded_chunks, loaded_bm25 = load_existing_index()

if loaded_index is not None:
    db["index"] = loaded_index
    db["chunks"] = loaded_chunks
    db["bm25"] = loaded_bm25

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    file_path = f"data/{file.filename}"
    os.makedirs("data", exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the file
    pages = extract_pdf_text(file_path)
    db["chunks"] = chunk_text(pages)
    db["index"], db["bm25"], _ = create_search_index(db["chunks"])
    
    return {"message": f"File {file.filename} indexed successfully", "chunks": len(db["chunks"])}

@app.post("/search")
async def search(query_data: dict):
    if db["index"] is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet")
    
    start_time = time.perf_counter()  # Start the high-res timer
    
    query = query_data["query"]
    top_k = query_data.get("top_k", 3)
    candidate_k = 8
    
    # --- Retrieval Logic ---
    q_emb = model.encode([query], normalize_embeddings=True)
    dense_scores, dense_indices = db["index"].search(q_emb, candidate_k)
    bm25_scores = np.array(db["bm25"].get_scores(query.lower().split()))
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
    
    candidate_indices = list(set(dense_indices[0]) | set(bm25_top_indices))
    pairs = [(query, db["chunks"][idx]["text"]) for idx in candidate_indices if idx != -1]
    rerank_scores = reranker.predict(pairs)
    
    results = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
    
    end_time = time.perf_counter()  # Stop the timer
    latency = end_time - start_time  # Calculate duration in seconds
    
    return {
        "latency_seconds": latency,
        "results": [
            {
                "score": float(s),
                "text": db["chunks"][i]["text"],
                "page": db["chunks"][i]["page_number"]
            } for i, s in results[:top_k]
        ]
    }