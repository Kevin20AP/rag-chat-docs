"""
Hybrid retrieval:
  dense (pgvector semantic)  +  sparse (Postgres full-text keyword)
  -> fused with Reciprocal Rank Fusion
  -> reranked with a cross-encoder for final relevance.
"""

from sentence_transformers import CrossEncoder
from db import get_conn
from embeddings import embed

# A cross-encoder scores (query, passage) pairs directly — far more accurate
# than embedding similarity, but slower, so we only run it on the shortlist.
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def dense_search(query, k=10):
    """Semantic search: closest chunks by vector cosine distance (<=>)."""
    vec = embed(query)
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (vec, k),
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def sparse_search(query, k=10):
    """Keyword search: Postgres full-text ranking over the tsvector column."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, content FROM chunks "
            "WHERE tsv @@ plainto_tsquery('english', %s) "
            "ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %s)) DESC "
            "LIMIT %s",
            (query, query, k),
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def reciprocal_rank_fusion(result_lists, k=60):
    """
    Merge several ranked lists. Each item scores 1/(k+rank) in each list it
    appears in; we sum those. Items ranked high in BOTH lists float to the top.
    """
    scores, content = {}, {}
    for lst in result_lists:
        for rank, (cid, text) in enumerate(lst):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            content[cid] = text
    ranked = sorted(scores, key=scores.get, reverse=True) # type: ignore
    return [(cid, content[cid]) for cid in ranked]


def rerank(query, candidates, top_n=4):
    """Re-score the fused candidates with the cross-encoder; keep the top_n."""
    if not candidates:
        return []
    pairs = [(query, text) for _, text in candidates]
    scores = _reranker.predict(pairs)
    order = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in order[:top_n]]


def hybrid_search(query, top_n=4):
    """Full pipeline: dense + sparse -> RRF -> cross-encoder rerank."""
    dense = dense_search(query)
    sparse = sparse_search(query)
    fused = reciprocal_rank_fusion([dense, sparse])
    return rerank(query, fused, top_n=top_n)