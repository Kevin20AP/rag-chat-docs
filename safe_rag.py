"""
Phase 5 (Part 1) — Guardrails.

Two safety layers on top of the RAG pipeline:
  1. Retrieval gate  : if no chunks come back, refuse instead of guessing.
  2. Groundedness    : after generating, an LLM verifies the answer is actually
                       supported by the context. If the score is below a
                       threshold, we return a safe fallback (no hallucinations).

Run:  uv run python safe_rag.py "How many PTO days do new employees get?"
"""

import os
import re
import sys
from anthropic import Anthropic
from dotenv import load_dotenv
from retrieval import hybrid_search

load_dotenv()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip())
MODEL = "claude-sonnet-5"
GROUNDEDNESS_THRESHOLD = 0.7   # answers below this are rejected

GEN_PROMPT = """Answer using ONLY the context below. If the answer is not present,
say exactly: "I don't know based on the provided documents."

Context:
{context}

Question: {q}
"""


def generate(q, context):
    r = client.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role": "user", "content": GEN_PROMPT.format(context=context, q=q)}])
    return "".join(b.text for b in r.content if b.type == "text")


def groundedness(answer, context):
    """LLM check: is every claim in the answer supported by the context? (0-1)"""
    prompt = (
        "Is every factual claim in the ANSWER supported by the CONTEXT? "
        "Reply with ONLY a number from 0 to 1 (1 = fully supported, 0 = not supported).\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nScore:"
    )
    r = client.messages.create(model=MODEL, max_tokens=10,
                               messages=[{"role": "user", "content": prompt}])
    t = "".join(b.text for b in r.content if b.type == "text")
    m = re.search(r"[01](?:\.\d+)?", t)
    return float(m.group()) if m else 0.0


def safe_answer(question):
    chunks = hybrid_search(question, top_n=4)

    # Guardrail 1: retrieval gate
    if not chunks:
        return "I couldn't find anything relevant in the documents.", None

    context = "\n\n".join(text for _, text in chunks)
    answer = generate(question, context)

    # Guardrail 2: groundedness check
    score = groundedness(answer, context)
    if score < GROUNDEDNESS_THRESHOLD:
        return ("I'm not able to answer that reliably from the provided documents.",
                score)

    return answer, score


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "How many PTO days do new employees get?"
    ans, score = safe_answer(q)
    print(f"\nQ: {q}")
    print(f"A: {ans}")
    print(f"(groundedness score: {score})")