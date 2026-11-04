"""
Phase 3 — Query transformation for smarter retrieval.

Two LLM-powered techniques, feeding into your hybrid search + reranking:
  - Multi-query: rewrite the question into several variations, retrieve for each.
  - HyDE: generate a hypothetical answer, then search with ITS embedding.

Run:  uv run python query_transform.py "Can I work from home at Nimbus?"
"""

import os
import sys
from anthropic import Anthropic
from dotenv import load_dotenv

from embeddings import embed
from db import get_conn
from retrieval import dense_search, sparse_search, reciprocal_rank_fusion, rerank

load_dotenv()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip())
MODEL = "claude-sonnet-5"


def multi_query(question, n=3):
    """Ask the LLM for n alternative phrasings of the question."""
    prompt = (
        f"Generate {n} alternative ways to phrase the following question for "
        "document search. Vary the wording and angle. Return ONLY the rephrasings, "
        "one per line, no numbering or bullets.\n\nQuestion: " + question
    )
    resp = client.messages.create(model=MODEL, max_tokens=300,
                                  messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    variants = [l.strip("-*0123456789. ").strip() for l in text.splitlines() if l.strip()]
    return [question] + variants[:n]


def hyde(question):
    """Generate a hypothetical answer paragraph (used only to improve search)."""
    prompt = (
        "Write a short, plausible paragraph that could answer this question, as if "
        "taken from a company document. Accuracy doesn't matter; it's only used to "
        "improve search.\n\nQuestion: " + question
    )
    resp = client.messages.create(model=MODEL, max_tokens=200,
                                  messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in resp.content if b.type == "text")


def dense_by_vector(vec, k=10):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (vec, k)).fetchall()
    return [(r[0], r[1]) for r in rows]


def enhanced_search(question, top_n=4, verbose=False):
    queries = multi_query(question)
    hyde_doc = hyde(question)
    if verbose:
        print("\n-- Query variations the LLM generated --")
        for q in queries:
            print("   -", q)
        print("\n-- HyDE hypothetical answer (searched with this) --")
        print("  ", hyde_doc[:220].replace("\n", " "), "...\n")

    lists = []
    for q in queries:
        lists.append(dense_search(q))
        lists.append(sparse_search(q))
    lists.append(dense_by_vector(embed(hyde_doc)))   # HyDE dense search

    fused = reciprocal_rank_fusion(lists)
    return rerank(question, fused, top_n=top_n)      # rerank vs the ORIGINAL question


PROMPT = """Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I don't know based on the provided documents."
After your answer, cite the id(s) of the chunk(s) you used, like [3].

Context:
{context}

Question: {question}
"""


def answer(question, verbose=True):
    chunks = enhanced_search(question, top_n=4, verbose=verbose)
    context = "\n\n".join(f"[{cid}] {text}" for cid, text in chunks)
    resp = client.messages.create(
        model=MODEL, max_tokens=500,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, [cid for cid, _ in chunks]


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Can I work from home at Nimbus?"
    ans, cites = answer(q)
    print("\nQ:", q)
    print("\nA:", ans)
    print("\n(retrieved chunk ids:", cites, ")")