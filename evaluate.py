"""
Phase 4 — RAG Evaluation harness (LLM-as-judge).

This is the core idea behind RAGAS: run the pipeline on a test set, then use a
strong LLM to SCORE the output on several metrics (0 to 1).

Metrics:
  faithfulness      : is the ANSWER supported by the retrieved CONTEXT?  (no hallucination)
  answer_relevance  : does the ANSWER address the QUESTION?
  context_relevance : were the retrieved CHUNKS relevant to the QUESTION? (retrieval quality)
  correctness       : does the ANSWER match the known GROUND TRUTH?

Run:  uv run python evaluate.py
"""

import os
import re
from statistics import mean
from anthropic import Anthropic
from dotenv import load_dotenv
from retrieval import hybrid_search

load_dotenv()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip())
MODEL = "claude-sonnet-5"

# --- the test set: questions + the correct (ground-truth) answers ---
DATASET = [
    {"q": "How many PTO days do new employees get?",
     "truth": "18 days per year during the first two years."},
    {"q": "What is the top speed of the Drift D3?",
     "truth": "2.1 meters per second."},
    {"q": "How much parental leave does Nimbus offer?",
     "truth": "16 weeks of fully paid parental leave."},
    {"q": "How many days per week must engineers be on-site?",
     "truth": "Two days per week: Tuesday and Thursday."},
    {"q": "How much does a replacement badge cost?",
     "truth": "25 dollars."},
]

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


def judge_score(instruction, **fields):
    """Ask the LLM to score something from 0 to 1 and parse the number out."""
    body = "\n\n".join(f"{k.upper()}:\n{v}" for k, v in fields.items())
    prompt = (f"{instruction}\nReply with ONLY a number between 0 and 1 "
              f"(e.g. 0.8), nothing else.\n\n{body}\n\nScore:")
    r = client.messages.create(model=MODEL, max_tokens=10,
                               messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in r.content if b.type == "text")
    m = re.search(r"[01](?:\.\d+)?", text)
    return float(m.group()) if m else 0.0


def evaluate():
    rows = []
    for item in DATASET:
        q, truth = item["q"], item["truth"]
        chunks = hybrid_search(q, top_n=4)
        context = "\n\n".join(text for _, text in chunks)
        ans = generate(q, context)

        faith = judge_score(
            "How well is the ANSWER supported by the CONTEXT? "
            "1 = every claim is supported, 0 = unsupported / hallucinated.",
            context=context, answer=ans)
        relev = judge_score(
            "How well does the ANSWER address the QUESTION? "
            "1 = directly answers it, 0 = irrelevant.",
            question=q, answer=ans)
        ctx = judge_score(
            "How relevant is the retrieved CONTEXT to the QUESTION? "
            "1 = highly relevant, 0 = irrelevant.",
            question=q, context=context)
        corr = judge_score(
            "Does the ANSWER match the GROUND TRUTH? "
            "1 = matches, 0 = wrong.",
            ground_truth=truth, answer=ans)

        rows.append((faith, relev, ctx, corr))
        print(f"\nQ: {q}")
        print(f"  A: {ans[:110]}")
        print(f"  faithfulness={faith}  answer_relevance={relev}  "
              f"context_relevance={ctx}  correctness={corr}")

    print("\n===================  AVERAGE SCORES  ===================")
    print(f"  Faithfulness      : {mean(r[0] for r in rows):.2f}")
    print(f"  Answer relevance  : {mean(r[1] for r in rows):.2f}")
    print(f"  Context relevance : {mean(r[2] for r in rows):.2f}")
    print(f"  Correctness       : {mean(r[3] for r in rows):.2f}")
    print("========================================================")


if __name__ == "__main__":
    evaluate()