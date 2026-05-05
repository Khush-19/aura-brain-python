# Aura Brain

An AI-powered RAG (Retrieval-Augmented Generation) engine backend built with FastAPI.

## Overview

Aura Brain ingests documents, stores them as vector embeddings in ChromaDB, and answers
natural language questions by retrieving relevant context before generating a response
via an LLM.

## Project Structure

```
aura-brain-python/
├── app/
│   ├── main.py          # FastAPI app initialization
│   ├── api/
│   │   └── routes.py    # HTTP route definitions
│   └── rag/
│       ├── pipeline.py  # End-to-end RAG orchestration
│       ├── retriever.py # Vector similarity search (ChromaDB)
│       └── embeddings.py# Text → vector embedding logic
├── data/                # Persisted ChromaDB files
├── .env                 # Environment variables (not committed)
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env .env.local  # fill in your API keys
```

## Run

```bash
uvicorn app.main:app --reload
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## Endpoints

| Method | Path     | Description              |
|--------|----------|--------------------------|
| GET    | `/`      | Health check             |
| POST   | `/query` | Submit a question to RAG |
