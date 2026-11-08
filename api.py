"""
Phase 6/7 — FastAPI RAG service, instrumented with Langfuse observability.

Every /ask call produces a nested trace in Langfuse:
    ask -> run_rag -> retrieve -> generate

Run locally:  uv run uvicorn api:app --reload
In Docker:    docker compose up --build
Docs:         http://127.0.0.1:8000/docs
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

# @observe works in both Langfuse v2 and v3; the import path differs.
# If Langfuse isn't configured, @observe simply no-ops (the API still works).
try:
    from langfuse import observe              # Langfuse v3
except ImportError:
    from langfuse.decorators import observe   # Langfuse v2

from retrieval import hybrid_search
from ingest import chunk_text
from embeddings import embed
from db import get_conn, init_db
from safe_rag import groundedness, GROUNDEDNESS_THRESHOLD

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip())
MODEL = "claude-sonnet-5"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="RAG Chat Docs API", version="1.1.0", lifespan=lifespan)

GEN_PROMPT = """Answer using ONLY the context below. If the answer is not present,
say exactly: "I don't know based on the provided documents."
Cite the chunk id(s) you used in square brackets like [3].

Context:
{context}

Question: {q}
"""


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    citations: list[int]
    groundedness: float


class IngestRequest(BaseModel):
    source: str = Field(..., description="A label for where this text came from")
    text: str = Field(..., min_length=1)


# ---- traced pipeline steps (each becomes a span in Langfuse) ----
@observe()
def retrieve(question):
    return hybrid_search(question, top_n=4)


@observe()
def generate(question, context):
    resp = client.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role": "user", "content": GEN_PROMPT.format(context=context, q=question)}])
    return "".join(b.text for b in resp.content if b.type == "text")


@observe()
def run_rag(question):
    chunks = retrieve(question)
    if not chunks:
        return "I couldn't find anything relevant in the documents.", [], 0.0
    context = "\n\n".join(f"[{cid}] {text}" for cid, text in chunks)
    answer = generate(question, context)
    score = groundedness(answer, context)
    if score < GROUNDEDNESS_THRESHOLD:
        answer = "I'm not able to answer that reliably from the provided documents."
    return answer, [cid for cid, _ in chunks], score


# ---- endpoints ----
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answer, citations, score = run_rag(req.question)
    return AskResponse(answer=answer, citations=citations, groundedness=score)


@app.post("/ingest")
def ingest(req: IngestRequest):
    chunks = chunk_text(req.text)
    vectors = embed(chunks)
    with get_conn() as conn:
        with conn.cursor() as cur:
            for content, vec in zip(chunks, vectors):
                cur.execute(
                    "INSERT INTO chunks (source, content, embedding, tsv) "
                    "VALUES (%s, %s, %s, to_tsvector('english', %s))",
                    (req.source, content, vec, content))
        conn.commit()
    return {"status": "ingested", "chunks_added": len(chunks)}