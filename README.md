# Aura Brain

FastAPI RAG microservice for Sydney lifestyle discovery and insider tips.

## Architecture

Production uses Pinecone integrated inference with `llama-text-embed-v2`.
Aura Brain sends raw text records plus flat string metadata to Pinecone via
`upsert_records()` and searches with `search_records()`. It does not use
LangChain's Pinecone vector store wrapper in production.

Local development can use FAISS. That path is isolated and is the only code path
that creates `OpenAIEmbeddings`.

## Key Environment

```bash
VECTOR_STORE=pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=aura-brain
PINECONE_NAMESPACE=aura-brain
PINECONE_TEXT_FIELD=text
PINECONE_EMBED_MODEL=llama-text-embed-v2

OPENAI_API_KEY=...   # used by the chat model, and by FAISS embeddings locally
CHAT_MODEL=gpt-4o
```

For local FAISS:

```bash
VECTOR_STORE=faiss
OPENAI_API_KEY=...
python scripts/seed_data.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m uvicorn app.main:app --reload
```

API docs are available at `http://localhost:8000/docs`.

With an empty `.env`, Aura Brain defaults to `VECTOR_STORE=pinecone`. Add a
valid `PINECONE_API_KEY` before calling `/api/v1/ingest` in production mode.
Use `VECTOR_STORE=faiss` only for local fallback runs, and set `OPENAI_API_KEY`
for that path.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Service health and vector backend |
| POST | `/api/v1/query` | Generate a personalised insider tip |
| POST | `/api/v1/ingest` | Scrape sources and store chunks |
