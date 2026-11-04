"""
Database layer — PostgreSQL + pgvector.

Stores each document chunk with BOTH:
  - embedding : a 384-dim vector (for dense / semantic search)
  - tsv       : a tsvector (for sparse / keyword search, used in hybrid search)
"""

import psycopg
from pgvector.psycopg import register_vector

DB_URL = "postgresql://postgres:postgres@localhost:5433/ragdb"
EMBED_DIM = 384  # dimension of the all-MiniLM-L6-v2 embedding model


def _ensure_extension():
    # The vector extension must exist before we can register the vector type.
    with psycopg.connect(DB_URL) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()


def get_conn():
    """A connection that understands pgvector's `vector` type."""
    conn = psycopg.connect(DB_URL)
    register_vector(conn)
    return conn


def init_db():
    _ensure_extension()
    with get_conn() as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id        SERIAL PRIMARY KEY,
                source    TEXT NOT NULL,
                content   TEXT NOT NULL,
                embedding vector({EMBED_DIM}),
                tsv       tsvector
            );
        """)
        # HNSW index = fast approximate nearest-neighbour search for vectors.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
            "ON chunks USING hnsw (embedding vector_cosine_ops);"
        )
        # GIN index = fast keyword search over the tsvector.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);"
        )
        conn.commit()
    print("Database ready (pgvector + indexes created).")


if __name__ == "__main__":
    init_db()