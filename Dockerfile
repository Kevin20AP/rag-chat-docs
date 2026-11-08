# Dockerfile — builds the RAG API image.
FROM python:3.13-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy everything, then install (simple + reliable with a src-layout project).
COPY . .
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]