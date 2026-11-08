# Chat with Your Docs — Advanced RAG System

A production-grade **Retrieval-Augmented Generation (RAG)** system that answers questions about your documents — grounded in the source, with citations, and engineered with the techniques that separate a real RAG pipeline from a basic one.

Built to learn and demonstrate the full advanced-RAG stack: hybrid retrieval, reranking, query transformation, evaluation, guardrails, observability, and containerized deployment.

---

## Why this is more than a basic RAG

A basic RAG embeds chunks and does a single vector search. This system adds the techniques that actually matter in production:

- **Hybrid retrieval** — combines **dense** (semantic / pgvector) and **sparse** (keyword / Postgres full-text) search, fused with **Reciprocal Rank Fusion (RRF)**. Catches both meaning *and* exact terms.
- **Cross-encoder reranking** — re-scores retrieved candidates with a cross-encoder for a major precision boost.
- **Query transformation** — **multi-query** (LLM rewrites the question several ways) and **HyDE** (searches with a hypothetical answer) to bridge the gap between how users ask and how documents are written.
- **Grounded generation with citations** — answers use only retrieved context and cite their sources.
- **Guardrails** — a post-generation **groundedness check** rejects unsupported answers (returns a safe fallback instead of hallucinating), plus a retrieval gate.
- **Evaluation** — an **LLM-as-judge** harness scoring faithfulness, answer/context relevance, and correctness against a ground-truth set (the idea behind RAGAS).
- **Observability** — **Langfuse** tracing of every retrieval and generation step.

## Tech stack

`Python 3.13` · `FastAPI` · `PostgreSQL + pgvector` · `sentence-transformers` (embeddings + cross-encoder) · `Anthropic Claude` · `Langfuse` · `Docker` · `uv`

## Architecture

```
          question
             │
             ▼
   ┌────────────────────┐
   │ Query transform    │  multi-query + HyDE (LLM)
   └─────────┬──────────┘
             ▼
   ┌────────────────────┐   dense (pgvector)  ┐
   │ Hybrid retrieval   │─────────────────────┤──► Reciprocal Rank Fusion
   │                    │   sparse (full-text)┘
   └─────────┬──────────┘
             ▼
   ┌────────────────────┐
   │ Cross-encoder rerank│
   └─────────┬──────────┘
             ▼
   ┌────────────────────┐
   │ Grounded generation│  Claude, with citations
   └─────────┬──────────┘
             ▼
   ┌────────────────────┐
   │ Groundedness guard │  reject if unsupported
   └─────────┬──────────┘
             ▼
     answer + citations
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/ask` | Ask a question → grounded answer, citations, groundedness score |
| `POST` | `/ingest` | Add a document (raw text) to the vector store at runtime |

## Running it

**Prerequisites:** Docker Desktop and an Anthropic API key.

1. Create a `.env` in the project root:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   # optional, for tracing:
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://us.cloud.langfuse.com
   ```
2. Start the whole stack (API + pgvector database):
   ```
   docker compose up --build
   ```
3. Open the interactive docs: **http://localhost:8000/docs**
4. `POST /ingest` a document, then `POST /ask` a question about it.

To stop: `docker compose down` (ingested data persists in the `ragdata` volume).

## Key components

| File | Role |
|------|------|
| `ingest.py` | Load → structure-aware chunking → embed → store |
| `db.py` | pgvector schema + connection (vector + full-text indexes) |
| `embeddings.py` | Local embedding model |
| `retrieval.py` | Hybrid search + RRF + cross-encoder reranking |
| `query_transform.py` | Multi-query + HyDE |
| `safe_rag.py` | Groundedness guardrail |
| `evaluate.py` | LLM-as-judge evaluation suite |
| `rag_observed.py` | Langfuse-instrumented pipeline |
| `api.py` | FastAPI service |

## What I learned

Building this covered the real advanced-RAG stack: why hybrid search beats pure vector search, how reranking and query transformation lift retrieval quality, how to *measure* a RAG system with faithfulness/relevance metrics, how to guard against hallucination, and how to make the whole thing observable and deployable with Docker.