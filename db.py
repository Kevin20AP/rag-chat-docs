"""
Database layer — PostgreSQL + pgvector.

The connection URL comes from the DATABASE_URL env var so the same code works
both locally (localhost:5433) and in Docker (the 'db' service).
"""

import os
import psycopg
from pgvector.psycopg import register_vector

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/ragdb"
)
EMBED_DIM = 384


def _ensure_extension():
    with psycopg.connect(DB_URL) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()


def get_conn():
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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
            "ON chunks USING hnsw (embedding vector_cosine_ops);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);"
        )
        conn.commit()
    print("Database ready (pgvector + indexes created).")


if __name__ == "__main__":
    init_db()