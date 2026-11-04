"""
Ask a question against your ingested documents.

  hybrid retrieve -> build a grounded prompt -> Claude answers WITH citations.

Run:  uv run python query.py "How many PTO days do new employees get?"
"""

import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic
from retrieval import hybrid_search
load_dotenv() 
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip())
MODEL = "claude-sonnet-5"

PROMPT = """Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I don't know based on the provided documents."
After your answer, cite the id(s) of the chunk(s) you used, in square brackets like [3].

Context:
{context}

Question: {question}
"""


def answer(question):
    chunks = hybrid_search(question, top_n=4)
    context = "\n\n".join(f"[{cid}] {text}" for cid, text in chunks)
    prompt = PROMPT.format(context=context, question=question)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, [cid for cid, _ in chunks]


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "How many PTO days do new employees get?"
    ans, cites = answer(q)
    print("\nQ:", q)
    print("\nA:", ans)
    print("\n(retrieved chunk ids:", cites, ")")