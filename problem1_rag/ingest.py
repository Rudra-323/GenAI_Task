"""Ingest a folder of PDF / HTML / Markdown files into Chroma.

    python -m problem1_rag.ingest                       # uses data/docs, size 500, overlap 50
    python -m problem1_rag.ingest --chunk-size 800 --overlap 100
    python -m problem1_rag.ingest --reset               # wipe the collection first

IDEMPOTENT: running it twice gives the same number of vectors. Two things guarantee this:
  1. each chunk id = hash(file name + chunk number + chunk text)  -> same chunk = same id
  2. before inserting a file we delete that file's old chunks      -> edited files leave no stale chunks
"""
import argparse
import hashlib
import os
import re
import time
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from problem1_rag import rag

HERE = Path(__file__).parent
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))      # characters (default 500)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # characters (default 50)

FILE_TYPES = {".md": "md", ".markdown": "md", ".txt": "txt", ".html": "html", ".htm": "html", ".pdf": "pdf"}


# ---------------------------------------------------------------- loading
def read_sections(path):
    """Return a list of (text, page_number_or_None) for one file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = PdfReader(str(path)).pages
        # PDF line breaks are just visual wrapping, so join lines with spaces
        return [(re.sub(r"\s*\n\s*", " ", p.extract_text() or ""), i + 1) for i, p in enumerate(pages)]
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in (".html", ".htm"):
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        blocks = [t.get_text(" ", strip=True) for t in soup.find_all(["h1", "h2", "h3", "p", "li"])]
        return [("\n\n".join(b for b in blocks if b), None)]
    return [(raw, None)]  # markdown / text


# ---------------------------------------------------------------- chunking
def _split_long(paragraph, size):
    """Split one very long paragraph at sentence ends (or at spaces if a sentence is huge)."""
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > size:  # hard split at the last space before `size`
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        if current and len(current) + len(sentence) + 1 > size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Greedy paragraph-aware chunking.

    Paragraphs are packed together until `size` characters. Each new chunk starts with the
    last `overlap` characters of the previous one so a fact on a boundary is not cut in half.
    A chunk can be at most about size + overlap characters long.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    text = re.sub(r"[ \t]+", " ", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces = []
    for p in paragraphs:
        pieces.extend([p] if len(p) <= size else _split_long(p, size))

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            if " " in tail:                       # start the overlap at a word boundary
                tail = tail[tail.index(" ") + 1:]
            current = f"{tail}\n\n{piece}".strip() if tail else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------- ingestion
def ingest_file(collection, path, size, overlap):
    """Chunk one file and upsert its chunks. Returns the number of chunks."""
    file_type = FILE_TYPES[path.suffix.lower()]
    chunks = []  # (text, page)
    for text, page in read_sections(path):
        chunks.extend((c, page) for c in chunk_text(text, size, overlap))

    ids, documents, metadatas = [], [], []
    for i, (text, page) in enumerate(chunks):
        ids.append(hashlib.sha1(f"{path.name}|{i}|{text}".encode("utf-8")).hexdigest()[:20])
        documents.append(text)
        meta = {"source": path.name, "file_type": file_type, "chunk_index": i}
        if page:
            meta["page"] = page
        metadatas.append(meta)

    collection.delete(where={"source": path.name})  # remove the old version of this file (if any)
    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def ingest_dir(collection, docs_dir, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Ingest every supported file in a folder. Returns {file name: chunk count}."""
    files = sorted(p for p in Path(docs_dir).iterdir() if p.suffix.lower() in FILE_TYPES)
    return {p.name: ingest_file(collection, p, size, overlap) for p in files}


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF/HTML/MD files into Chroma")
    parser.add_argument("--docs", default=str(HERE / "data" / "docs"), help="folder with the documents")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="characters per chunk (default 500)")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="overlap characters (default 50)")
    parser.add_argument("--reset", action="store_true", help="delete the collection before ingesting")
    args = parser.parse_args()

    collection = rag.get_collection()
    if args.reset:
        old_ids = collection.get()["ids"]
        if old_ids:
            collection.delete(ids=old_ids)
    before = collection.count()

    start = time.perf_counter()
    per_file = ingest_dir(collection, args.docs, args.chunk_size, args.overlap)
    seconds = time.perf_counter() - start

    for name, n in per_file.items():
        print(f"  {name:28s} {n:4d} chunks")
    print(f"Embedding model : {rag.EMBED_MODEL} ({rag.EMBED_DIM} dimensions)")
    print(f"Chunk size/overlap: {args.chunk_size} / {args.overlap} characters")
    print(f"Vectors before  : {before}")
    print(f"Vectors after   : {collection.count()}   (took {seconds:.1f}s)")


if __name__ == "__main__":
    main()
