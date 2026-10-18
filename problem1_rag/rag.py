"""Retrieval + grounded answering. Used by the API (app.py) and the evaluation (evaluate.py).

Vector store : ChromaDB (local, persistent, HNSW index, cosine distance)
Embeddings   : all-MiniLM-L6-v2 (384 dimensions) - Chroma's built-in local ONNX model, $0 per call
"""
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from llm import LLM  # importing llm also loads the .env file

HERE = Path(__file__).parent
DB_PATH = os.getenv("CHROMA_PATH", str(HERE / "chroma_db"))
COLLECTION = os.getenv("COLLECTION_NAME", "docs")
DEFAULT_K = int(os.getenv("DEFAULT_K", "5"))
# If the best chunk is less similar than this, we do not even call the LLM.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
NOT_FOUND = "I don't know based on the provided documents."
LOG_FILE = HERE / "logs" / "queries.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag")

_embedder = None
_collection = None
_llm = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = DefaultEmbeddingFunction()  # downloads the small model once, then cached
    return _embedder


def get_collection(path=None, name=None):
    """Open (or create) a Chroma collection. Defaults come from env vars."""
    client = chromadb.PersistentClient(path=str(path or DB_PATH), settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(
        name=name or COLLECTION,
        embedding_function=get_embedder(),
        metadata={"hnsw:space": "cosine"},
    )


def default_collection():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection


def get_llm():
    global _llm
    if _llm is None:
        _llm = LLM("GEN")
    return _llm


def build_where(file_type=None, source=None):
    """Metadata filter in Chroma syntax. Returns None when there is no filter."""
    conditions = []
    if file_type:
        conditions.append({"file_type": file_type})
    if source:
        conditions.append({"source": source})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def retrieve(question, k=DEFAULT_K, file_type=None, source=None, collection=None):
    """Return the top-k chunks (most similar first) as a list of dicts."""
    col = collection or default_collection()
    total = col.count()
    if total == 0:
        return []
    res = col.query(
        query_texts=[question],
        n_results=min(k, total),
        where=build_where(file_type, source),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        chunks.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "source": meta["source"],
            "file_type": meta["file_type"],
            "chunk_index": meta["chunk_index"],
            "similarity": round(1 - res["distances"][0][i], 4),  # cosine distance -> similarity
        })
    return chunks


def build_messages(question, chunks):
    context = "\n\n".join(f"[{i}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks, 1))
    system = (
        "You answer questions using ONLY the numbered context chunks provided.\n"
        "Rules:\n"
        "1. After every fact, cite the chunk number you used, like [1] or [2][3].\n"
        f"2. If the context does not contain the answer, reply with exactly: {NOT_FOUND}\n"
        "3. Never use outside knowledge. Never guess.\n"
        "4. Answer in at most 3 short sentences."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def answer(question, k=DEFAULT_K, file_type=None, source=None, collection=None):
    """Full RAG: retrieve -> (gate) -> LLM -> cited answer. Logs latency, chunks and tokens."""
    t0 = time.perf_counter()
    chunks = retrieve(question, k, file_type, source, collection)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    llm_ms, prompt_tokens, completion_tokens = 0.0, 0, 0
    if not chunks or chunks[0]["similarity"] < MIN_SIMILARITY:
        # Nothing relevant was found: refuse WITHOUT calling the LLM (cheaper + cannot hallucinate).
        text = NOT_FOUND
        reason = "no_chunks" if not chunks else "below_similarity_threshold"
    else:
        out = get_llm().chat(build_messages(question, chunks))
        text = out["text"].strip()
        llm_ms, prompt_tokens, completion_tokens = out["latency_ms"], out["prompt_tokens"], out["completion_tokens"]
        reason = "llm_refused" if NOT_FOUND.lower() in text.lower() else "answered"

    answerable = reason == "answered"
    cited_numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(chunks)})
    citations = [{"n": n, "source": chunks[n - 1]["source"], "chunk_index": chunks[n - 1]["chunk_index"],
                  "similarity": chunks[n - 1]["similarity"], "snippet": chunks[n - 1]["text"][:200]}
                 for n in cited_numbers] if answerable else []

    metrics = {
        "n_chunks": len(chunks),
        "retrieval_ms": round(retrieval_ms, 1),
        "llm_ms": round(llm_ms, 1),
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "question": question, "k": k,
              "file_type": file_type, "source": source, "answerable": answerable, "reason": reason,
              "top_similarity": chunks[0]["similarity"] if chunks else None, **metrics}
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    log.info("query k=%s chunks=%s retrieval=%.1fms llm=%.1fms total=%.1fms tokens=%s+%s answerable=%s (%s)",
             k, metrics["n_chunks"], retrieval_ms, llm_ms, metrics["total_ms"], prompt_tokens,
             completion_tokens, answerable, reason)

    retrieved = [{"n": i, **{key: c[key] for key in ("source", "file_type", "chunk_index", "similarity")},
                  "text": c["text"]} for i, c in enumerate(chunks, 1)]
    return {"question": question, "answer": text, "answerable": answerable, "reason": reason,
            "citations": citations, "retrieved": retrieved, "metrics": metrics}
