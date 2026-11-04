"""
Ingestion pipeline:  load -> smart-chunk -> embed -> store in pgvector.

Run:  uv run python ingest.py sample.txt
"""

import sys
from pypdf import PdfReader
from db import get_conn, init_db
from embeddings import embed


def load_file(path):
    """Read text from a .pdf or a plain text/markdown file."""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, encoding="utf-8") as f:
        return f.read()


def chunk_text(text, chunk_size=800, overlap=150):
    """
    Structure-aware chunking: pack whole paragraphs together up to ~chunk_size
    characters (so we don't cut sentences in half), with a sliding overlap so
    context isn't lost at chunk boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= chunk_size:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            if len(p) > chunk_size:          # a single huge paragraph -> hard split
                for i in range(0, len(p), chunk_size - overlap):
                    chunks.append(p[i:i + chunk_size])
                current = ""
            else:
                current = p
    if current:
        chunks.append(current)

    # add overlap: prepend the tail of the previous chunk to each chunk
    overlapped = []
    for i, c in enumerate(chunks):
        if i > 0:
            tail = chunks[i - 1][-overlap:]
            c = tail + "\n\n" + c
        overlapped.append(c)
    return overlapped


def ingest(path):
    init_db()
    text = load_file(path)
    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks. Embedding...")
    vectors = embed(chunks)
    with get_conn() as conn:
        with conn.cursor() as cur:
            for content, vec in zip(chunks, vectors):
                cur.execute(
                    "INSERT INTO chunks (source, content, embedding, tsv) "
                    "VALUES (%s, %s, %s, to_tsvector('english', %s))",
                    (path, content, vec, content),
                )
        conn.commit()
    print(f"Ingested {len(chunks)} chunks from {path}.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python ingest.py <file>")
        sys.exit(1)
    ingest(sys.argv[1])