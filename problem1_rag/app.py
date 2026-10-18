"""HTTP API for the RAG service.

    uvicorn problem1_rag.app:app --port 8000

    curl -X POST localhost:8000/query -H "Content-Type: application/json" \
         -d '{"question": "What is the rate limit for the Pro plan?", "k": 5}'

Interactive docs (try it in the browser): http://localhost:8000/docs
"""
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from problem1_rag import rag

app = FastAPI(title="Cost-efficient RAG (Chroma)", version="1.0")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, description="The question to answer from the documents")
    k: int = Field(default=rag.DEFAULT_K, ge=1, le=20, description="How many chunks to retrieve")
    file_type: Optional[str] = Field(default=None, description="Metadata filter: md | pdf | html | txt")
    source: Optional[str] = Field(default=None, description="Metadata filter: exact file name")


@app.get("/health")
def health():
    return {"status": "ok", "vectors": rag.default_collection().count(),
            "embedding_model": rag.EMBED_MODEL, "embedding_dim": rag.EMBED_DIM}


@app.post("/query")
def query(req: QueryRequest):
    if rag.default_collection().count() == 0:
        raise HTTPException(status_code=409, detail="The index is empty. Run: python -m problem1_rag.ingest")
    try:
        return rag.answer(req.question, k=req.k, file_type=req.file_type, source=req.source)
    except ValueError as err:                      # bad config such as a missing API key
        raise HTTPException(status_code=500, detail=str(err))
    except Exception as err:                       # LLM provider problems
        raise HTTPException(status_code=502, detail=f"LLM call failed: {type(err).__name__}: {err}")
