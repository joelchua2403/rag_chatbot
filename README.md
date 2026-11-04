# arXiv RAG Assessment — Weaviate + Ollama

A locally deployed, version-aware Retrieval-Augmented Generation (RAG) system for academic papers from arXiv.

The application ingests multiple versions of research papers, stores their embeddings and metadata in Weaviate, answers questions using a locally hosted LLM, and compares paper revisions to distinguish substantive scientific changes from cosmetic and metadata updates.

## Features

- **Version-aware ingestion:** Downloads arXiv PDFs, extracts text, chunks documents, generates embeddings, and indexes each paper version independently.
- **Idempotent ingestion:** Uses deterministic identifiers and content hashes to prevent duplicate indexing when unchanged versions are ingested again.
- **Grounded RAG chatbot:** Retrieves relevant paper passages and generates answers using Ollama, with citations identifying paper versions and page numbers.
- **Version-specific retrieval:** Supports questions about particular versions of a paper, defaulting to the latest indexed version when none is specified.
- **Paper version comparison:** Uses deterministic text differences, PDF extraction artifact cleanup, metadata detection, and LLM-assisted classification.
- **FastAPI:** Provides REST endpoints and interactive Swagger documentation.
- **Automated tests:** Includes tests for chunking, deterministic IDs, numerical changes, metadata classification, and PDF alignment artifacts.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.11 |
| API framework | FastAPI |
| Vector database | Weaviate 1.30.5 |
| Embedding model | SentenceTransformers `all-MiniLM-L6-v2` |
| Local LLM | Ollama `qwen2.5:7b` |
| PDF extraction | PyMuPDF |
| Version comparison | Python `difflib` + Ollama |
| Containerization | Docker Compose |
| Testing | pytest |

## Architecture

### Document Ingestion

```text
arXiv API / PDFs
       |
       v
PDF Download
       |
       v
PyMuPDF Text Extraction
       |
       v
Text Cleaning and Chunking
       |
       v
SentenceTransformer Embeddings
       |
       v
Weaviate
  |              |
  v              v
PaperVersion   PaperChunk
(metadata)     (vectors + text)
```

### Question Answering

```text
User Question
       |
       v
Paper ID / Version Detection
       |
       v
Question Embedding
       |
       v
Weaviate Vector Search
       |
       v
Version-Filtered Retrieval
       |
       v
Context + Citation Labels
       |
       v
Ollama (Qwen2.5 7B)
       |
       v
Citation Validation
       |
       v
Grounded Answer + Sources
```

### Version Comparison

```text
Paper Version 1 + Version 2
              |
              v
      Extracted Full Text
              |
              v
      Paragraph Alignment
              |
              v
       Text Differences
              |
              v
    Moved-Text / PDF Artifact
           Cleanup
              |
              v
     Metadata / Cosmetic /
     Scientific Classification
              |
              v
       Structured Report
```

## Prerequisites

- Python 3.11 or 3.12
- Docker Desktop
- Ollama
- Internet access for downloading arXiv papers and embedding models
- Sufficient system memory for the embedding model and local LLM

## Installation

### 1. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Review `.env` and adjust the configuration if necessary.

### 3. Start Weaviate

```bash
docker compose up -d
docker compose ps
```

Verify readiness:

```bash
curl http://localhost:8080/v1/.well-known/ready
```

### 4. Prepare Ollama

```bash
ollama pull qwen2.5:7