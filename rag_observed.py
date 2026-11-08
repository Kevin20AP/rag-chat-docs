"""
Phase 5 (Part 2) — Observability with Langfuse.

@observe() wraps each step so Langfuse records a nested trace
(answer -> retrieve -> generate) with inputs, outputs, timing and token usage,
all viewable in the Langfuse dashboard.

.env needs:
  LANGFUSE_PUBLIC_KEY=pk-lf-...
  LANGFUSE_SECRET_KEY=sk-lf-...
  LANGFUSE_HOST=https://us.cloud.langfuse.com   (or https://cloud.langfuse.com)

Run:  uv run python rag_observed.py "How many PTO days do new employees get?"
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# @observe exists in both Langfuse v2 and v3; the import path differs.
try:
    from langfuse import observe              # Langfuse v3
except ImportError:
    from langfuse.decorators import observe   # Langfuse v2

from anthropic import Anthropic
from retrieval import hybrid_search

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip())
MODEL = "claude-sonnet-5"

GEN_PROMPT = """Answer using ONLY the context below. If the answer is not present,
say exactly: "I don't know based on the provided documents."

Context:
{context}

Question: {q}
"""


@observe()
def retrieve(question):
    chunks = hybrid_search(question, top_n=4)
    return [{"id": cid, "text": text} for cid, text in chunks]


@observe()
def generate(question, context):
    r = client.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role": "user", "content": GEN_PROMPT.format(context=context, q=question)}])
    return "".join(b.text for b in r.content if b.type == "text")


@observe()
def answer(question):
    chunks = retrieve(question)
    context = "\n\n".join(c["text"] for c in chunks)
    return generate(question, context)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "How many PTO days do new employees get?"
    print("\nQ:", q)
    print("A:", answer(q))

    # Send the trace to Langfuse (handle both SDK versions).
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        try:
            from langfuse.decorators import langfuse_context
            langfuse_context.flush()
        except Exception:
            pass
    print("\n(trace sent to Langfuse — open your dashboard to see it)")