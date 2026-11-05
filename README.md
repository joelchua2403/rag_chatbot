
# arXiv Version-Aware RAG Chatbot

A version-aware Retrieval-Augmented Generation (RAG) system for querying academic papers from arXiv, retrieving evidence with Weaviate, generating grounded answers using a local Ollama LLM, and comparing different versions of the same paper.

The project includes document ingestion, configurable dense or hybrid search, version-aware retrieval, automated document comparison, a command-line chatbot, a FastAPI interface, and automated tests.

## 1. Features

### Version-aware document ingestion

- Downloads and processes configured arXiv papers and their versions.
- Extracts text from PDFs using PyMuPDF.
- Splits extracted text into searchable chunks.
- Generates embeddings using SentenceTransformers.
- Stores documents and chunks in Weaviate.
- Maintains separate records for each paper version.
- Uses content hashes to identify unchanged documents and avoid unnecessary re-ingestion.

### Retrieval-Augmented Generation

- Uses configurable dense vector or hybrid BM25 and vector search to retrieve relevant paper chunks.
- Automatically detects arXiv IDs in user questions.
- Supports explicit version selection, such as `2606.05387v1`.
- Defaults to the latest indexed version when no version is specified.
- Uses Ollama to generate answers from retrieved evidence.
- Includes source references with arXiv ID, version, and page number.
- Instructs the LLM to abstain when evidence is insufficient.

### Version comparison

The chatbot detects questions asking about differences between paper versions.

For example:

```text
What changed between versions 1 and 2 of arXiv:2606.05387?
```

Instead of using ordinary semantic retrieval, the chatbot routes the request to the document comparison pipeline.

The comparison pipeline:

1. Retrieves both indexed document versions.
2. Normalizes PDF-extracted text.
3. Aligns paragraphs using Python's `difflib.SequenceMatcher`.
4. Detects inserted, deleted, and replaced text.
5. Filters identical moved-text and duplicated-fragment artifacts.
6. Identifies publication metadata and cosmetic changes.
7. Uses an Ollama LLM to classify ambiguous differences.
8. Returns structured findings and a comparison summary.

The standalone comparison command remains available.

### Additional functionality

- FastAPI endpoints for querying and comparing papers.
- Interactive command-line chatbot.
- Docker Compose deployment for Weaviate.
- Automated tests using pytest.
- Weaviate inspection utility for evidence collection.

---

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.11 / 3.12 |
| API framework | FastAPI |
| Vector database | Weaviate 1.30.5 |
| Embedding model | SentenceTransformers `all-MiniLM-L6-v2` |
| Local LLM | Ollama `qwen2.5:7b` |
| PDF processing | PyMuPDF |
| Document comparison | Python `difflib` |
| Data validation | Pydantic |
| Containerization | Docker Compose |
| Testing | pytest |

The LLM runs locally through Ollama. An external commercial LLM API key is not required.

An internet connection is needed for initial downloads and paper ingestion.

---

## 3. System Architecture

### Normal RAG workflow

```text
User Question
      |
      v
Parse arXiv ID and Version
      |
      v
Generate Query Embedding
      |
      v
Weaviate Dense or Hybrid Search
      |
      v
Retrieve Version-Filtered Chunks
      |
      v
Construct Evidence Context
      |
      v
Ollama (Qwen2.5)
      |
      v
Grounded Answer + Citations
```

### Version-comparison workflow

```text
User Question
      |
      v
Detect Comparison Intent
      |
      v
Extract Paper ID and Versions
      |
      v
Retrieve Both Paper Versions
      |
      v
Normalize and Align Paragraphs
      |
      v
Detect Textual Differences
      |
      v
Filter PDF Alignment Artifacts
      |
      v
Classify Changes
      |
      v
Structured Comparison Report
```

Version-comparison questions bypass ordinary RAG generation to avoid treating content found in one version as evidence that it was newly introduced in that version.

---

## 4. Prerequisites

Install the following software before running the project.

### Python

Python 3.11 is recommended. Python 3.12 is also supported.

Download Python:

https://www.python.org/downloads/

On Windows, enable the option to add Python to PATH during installation.

Verify installation.

Windows PowerShell or Git Bash:

```bash
py -3.11 --version
```

macOS/Linux:

```bash
python3.11 --version
```

### Docker Desktop

Install Docker Desktop:

https://www.docker.com/products/docker-desktop/

Windows:

https://docs.docker.com/desktop/setup/install/windows-install/

macOS:

https://docs.docker.com/desktop/setup/install/mac-install/

Linux users can install Docker Engine and the Docker Compose plugin.

Start Docker Desktop before proceeding.

Verify:

```bash
docker --version
docker compose version
docker info
```

All commands should succeed.

### Ollama

Download Ollama:

https://ollama.com/download

Install the version appropriate for your operating system.

Start the Ollama application.

Verify:

```bash
ollama --version
```

Check the local API:

```bash
curl http://localhost:11434/api/tags
```

The setup script automatically downloads the configured LLM if necessary.

To download it manually:

```bash
ollama pull qwen2.5:7b
```

### Hardware and network requirements

Recommended:

- Python 3.11 or 3.12.
- Docker Desktop or Docker Engine with Compose.
- Ollama installed and running.
- Approximately 16 GB RAM or more for comfortable local execution.
- Several gigabytes of available disk space.
- Internet connectivity for dependency, model, and arXiv downloads.

Required local ports:

| Port | Service |
|---|---|
| 8080 | Weaviate HTTP |
| 50051 | Weaviate gRPC |
| 11434 | Ollama |
| 8000 | FastAPI, when started |

Ensure these ports are not already occupied by other applications.

---

## 5. Project Structure

```text
rag_chatbot/
|
|-- app/
|   |-- __init__.py
|   |-- arxiv_source.py
|   |-- comparison.py
|   |-- config.py
|   |-- ids.py
|   |-- ingestion.py
|   |-- llm.py
|   |-- main.py
|   |-- models.py
|   |-- parsing.py
|   |-- rag.py
|   |-- store.py
|   |-- text_utils.py
|
|-- scripts/
|   |-- chat.py
|   |-- compare.py
|   |-- ingest.py
|   |-- inspect_weaviate.py
|
|-- tests/
|   |-- test_comparison.py
|   |-- test_core.py
|
|-- docker-compose.yml
|-- requirements.txt
|-- .env.example
|-- setup.sh
|-- README.md
```

Additional tests may be added under `tests/`, including regression tests for comparison-intent routing.

---

## 6. Quick Start — Automated Setup

The project includes a cross-platform `setup.sh` script designed for:

- Windows 11 using Git Bash.
- macOS.
- Linux.

On Windows, use **Git Bash**, not PowerShell, to execute `setup.sh`.

First, ensure Docker Desktop and Ollama are running.

Open a terminal in the project directory:

```bash
cd rag_chatbot
```

Run:

```bash
./setup.sh
```

If required on macOS/Linux:

```bash
chmod +x setup.sh
./setup.sh
```

The script performs the following operations:

1. Checks Python, Docker, Docker Compose, Ollama, and curl.
2. Creates a Python virtual environment if necessary.
3. Creates `.env` from `.env.example` if missing.
4. Installs Python dependencies.
5. Starts Weaviate using Docker Compose.
6. Waits for Weaviate to become ready.
7. Checks the Ollama API.
8. Downloads the configured Ollama model.
9. Ingests the configured arXiv paper versions.
10. Inspects the Weaviate collections.
11. Executes the automated tests.
12. Launches the interactive chatbot.

The initial setup may take several minutes, depending on network speed and hardware.

Re-running the setup reuses the existing virtual environment and is designed to skip unchanged indexed documents.

---

## 7. Manual Setup

If the automated setup cannot be used, follow these steps.

### Windows 11 — Git Bash

Create the virtual environment:

```bash
py -3.11 -m venv .venv
```

Install dependencies:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Create the environment configuration:

```bash
cp .env.example .env
```

Start Weaviate:

```bash
docker compose up -d
```

Check readiness:

```bash
curl http://localhost:8080/v1/.well-known/ready
```

Download the LLM:

```bash
ollama pull qwen2.5:7b
```

Ingest the papers:

```bash
.venv/Scripts/python.exe -m scripts.ingest
```

Start the chatbot:

```bash
.venv/Scripts/python.exe -m scripts.chat
```

### macOS / Linux

Create and activate the virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create the environment configuration:

```bash
cp .env.example .env
```

Start Weaviate:

```bash
docker compose up -d
```

Download the LLM:

```bash
ollama pull qwen2.5:7b
```

Ingest the papers:

```bash
python -m scripts.ingest
```

Start the chatbot:

```bash
python -m scripts.chat
```

---

## 8. Using the Chatbot

Start the interactive chatbot.

Windows Git Bash:

```bash
.venv/Scripts/python.exe -m scripts.chat
```

macOS/Linux, with the virtual environment activated:

```bash
python -m scripts.chat
```

The terminal displays:

```text
arXiv Version-Aware RAG Chatbot
Type 'quit' or 'exit' to stop.

Question>
```

Enter a question and press Enter.

### Example 1 — Main contribution

```text
What is the main research contribution of arXiv paper 2606.05387?
```

Expected behavior:

- Detect the arXiv ID.
- Select the latest indexed version.
- Retrieve relevant chunks from Weaviate.
- Generate an evidence-grounded answer.
- Include source labels and page references.

### Example 2 — Explicit version selection

```text
According to version 1 of arXiv 2606.05387, what reporting standards do the authors recommend?
```

Expected behavior:

- Retrieve only chunks associated with version 1.
- Generate an answer using those chunks.
- Cite `2606.05387v1`.

### Example 3 — Compare versions

```text
What changed between versions 1 and 2 of arXiv:2606.05387?
```

Expected behavior:

- Detect the version-comparison intent.
- Extract the paper ID and both versions.
- Execute the comparison pipeline.
- Report detected textual changes.
- Avoid inventing scientific differences.

### Example 4 — Unsupported question

```text
According to arXiv 2606.05387, what was Apple's exact stock price on 15 March 2024?
```

Expected behavior:

The chatbot should state that the indexed evidence does not provide the requested information, rather than inventing a stock price.

### Example 5 — Another indexed paper

```text
Summarize the key findings of arXiv 2605.18821.
```

Expected behavior:

The chatbot should retrieve evidence from the specified paper and provide appropriate source citations.

### Retrieval modes

The default is hybrid retrieval, configured in `.env`:

```dotenv
RETRIEVAL_MODE=hybrid
HYBRID_ALPHA=0.5
RETRIEVAL_TOP_K=6
```

Weaviate combines BM25 keyword relevance and vector similarity for hybrid queries. `HYBRID_ALPHA` controls the balance: `0.0` uses BM25 only, `1.0` uses vector search only, and intermediate values fuse both rankings. The chatbot uses the same externally generated SentenceTransformers vectors in either mode.

To select dense-only retrieval, set `RETRIEVAL_MODE=dense`. To return to hybrid, set `RETRIEVAL_MODE=hybrid`; the alpha value applies only in hybrid mode. The mode must be `dense` or `hybrid`, and alpha must be between 0 and 1. Version filters are passed to both search modes.

To exit:

```text
quit
```

---

## 9. Comparing Paper Versions

The application supports two ways to compare indexed paper versions.

### Method 1 — Chatbot

Start the chatbot and enter:

```text
What changed between versions 1 and 2 of arXiv:2606.05387?
```

The chatbot routes the question to the comparison pipeline rather than ordinary semantic retrieval.

### Method 2 — Dedicated comparison CLI

Windows:

```bash
.venv/Scripts/python.exe -m scripts.compare 2606.05387 --old 1 --new 2
```

macOS/Linux:

```bash
python -m scripts.compare 2606.05387 --old 1 --new 2
```

The CLI returns structured JSON containing:

- arXiv ID.
- Old and new versions.
- Changed text blocks.
- Change types: insert, delete, or replace.
- Classifications.
- Before and after evidence.
- Comparison summary.

### Observed comparison result

For `2606.05387v1` and `2606.05387v2`, the local comparison pipeline detected:

| Change category | Count |
|---|---:|
| Substantive changes | 0 |
| Metadata changes | 3 |
| Other cosmetic changes | 0 |
| Total changed blocks | 3 |

The detected changes involved:

1. Author or institutional affiliation metadata.
2. Publication date metadata.
3. Author contact or affiliation information.

Metadata is currently represented using the `cosmetic` category with a `[Metadata]` description prefix to maintain compatibility with the existing Pydantic model.

The result means that no substantive scientific changes were **identified by the current automated pipeline**. It does not establish that the two papers are scientifically identical.

---

## 10. Weaviate Data Model

The application uses two Weaviate collections.

### PaperVersion

Stores metadata and extracted text for individual paper versions.

Relevant properties include:

| Property | Description |
|---|---|
| `arxiv_id` | Base arXiv identifier |
| `version` | Paper version number |
| `version_key` | Combined identifier, e.g. `2606.05387v2` |
| `title` | Paper title |
| `content_hash` | Hash of document content |
| `chunk_count` | Number of indexed chunks |
| `full_text` | Extracted document text |

### PaperChunk

Stores text chunks used for dense and hybrid retrieval. Its `text` property is indexed for BM25 keyword search.

Relevant properties include:

| Property | Description |
|---|---|
| `arxiv_id` | Base arXiv identifier |
| `version` | Paper version number |
| `version_key` | Version-specific identifier |
| `page` | Source PDF page |
| `section` | Extracted section label |
| `chunk_index` | Chunk position |
| `text` | Extracted chunk content |
| `content_hash` | Chunk content hash |

The `version_key` field enables retrieval to be restricted to a particular version.

---

## 11. Indexed Dataset and Validation Results

The implementation was tested locally with the following indexed documents.

| arXiv ID | Version | Indexed chunks |
|---|---|---:|
| 2605.18821 | v1 | 216 |
| 2606.05387 | v1 | 138 |
| 2606.05387 | v2 | 138 |
| 2607.00365 | v1 | 459 |
| **Total** | **4 versions** | **951** |

Observed Weaviate collection counts:

```text
PaperVersion: 4 objects
PaperChunk: 951 objects
```

The two indexed versions of `2606.05387` have different document content hashes, demonstrating that their stored contents differ.

### Automated tests

Windows:

```bash
.venv/Scripts/python.exe -m pytest -q
```

macOS/Linux:

```bash
python -m pytest -q
```

Latest test result after adding retrieval regression coverage:

```text
20 passed
```

### Retrieval evaluation

Run the standalone evaluation on the same indexed Weaviate corpus with the same 14-query dataset for every configuration:

```bash
.venv/Scripts/python.exe -m scripts.evaluate_retrieval
```

The script compares dense retrieval and hybrid alpha values 0.25, 0.50, and 0.75 at K=5. It writes raw result rankings, including stable Weaviate object UUIDs, to `data/retrieval_eval_results.json`. It also records median latency over three repeats per query and configuration (excluding query-embedding time); use `--repeats` to change this. Use `--dataset`, `--output`, or `--k` to select the input, output, or cutoff. Latency is an indicative local measurement, not a relevance or answer-quality metric.

Questions in `tests/data/retrieval_eval.json` were constructed from indexed paper passages and cover technical terminology, paraphrases, paper concepts, explicit and latest-version queries, similar versions, and author/publication metadata. Relevance labels are intentionally unset (`relevant_chunk_ids: null`): a reviewer must label relevant chunk UUIDs before Recall@K, Hit Rate@K, and MRR are meaningful. Mark `judgments_complete` true only after reviewing the full relevant set for that query; Precision@K is reported only for such queries. Until then, metric values are `null` and the raw rankings are not verified accuracy results. The output reports the number of judged, completely judged, and unjudged queries so partial labels cannot be mistaken for a full benchmark.

#### Manually review relevance

1. Generate a pooled annotation template from the latest evaluation output:

   ```bash
   .venv/Scripts/python.exe -m scripts.prepare_retrieval_judgments
   ```

2. Open `data/retrieval_judgments.json` or review the candidate passages in `notebooks/retrieval_performance.ipynb`. For each query, read every candidate and add the UUID of each chunk that directly supports answering the question to `relevant_chunk_ids`. Set `candidate_pool_reviewed` to `true` only after every candidate for that query has been judged relevant or not relevant. Leave it false while any candidate is undecided. Do not mark a chunk relevant only because a retrieval mode returned it.

3. Rerun the same comparison with the reviewed labels:

   ```bash
   .venv/Scripts/python.exe -m scripts.evaluate_retrieval --judgments data/retrieval_judgments.json
   ```

   Then rerun the notebook cells to chart the metrics.

The annotation template pools the top-K results from dense retrieval and all three hybrid settings. Thus Recall@K measures recall within the **judged candidate pool**, not exhaustive recall over every relevant chunk in the full Weaviate corpus. Hit Rate@K and MRR compare the systems against the same judged pool. Precision@K is calculated only for queries whose complete pooled candidate set has been reviewed. The evaluation output records this scope; use exhaustive corpus-wide judgments if claiming corpus-wide recall.

No Weaviate schema migration or re-indexing is required for the current corpus: the existing `PaperChunk.text` property is already searchable for BM25. New `PaperChunk` collections explicitly enable its searchable index.

### Evaluation outcome

The evaluation was run against the current 951-chunk corpus in Weaviate 1.30.5 using 14 questions, three timed repeats per query/configuration, and warmed, interleaved query execution. Median retrieval-call latency (embedding time excluded) was:

| Configuration | Median latency | Difference from dense |
|---|---:|---:|
| Dense | 2.10 ms | baseline |
| Hybrid, alpha 0.25 | 3.14 ms | +49.4% |
| Hybrid, alpha 0.50 | 3.24 ms | +54.0% |
| Hybrid, alpha 0.75 | 3.26 ms | +55.1% |

These are local retrieval-latency observations for this small corpus, not a general performance guarantee. The current run has no manually reviewed relevance labels, so no retrieval-quality metric or accuracy improvement is claimed; the rankings are unlabeled. Use the annotation steps above to test relevance quality. Open `notebooks/retrieval_performance.ipynb` to review candidates and visualize the latency and judged quality metrics.

### Inspecting Weaviate

Windows:

```bash
.venv/Scripts/python.exe -m scripts.inspect_weaviate
```

macOS/Linux:

```bash
python -m scripts.inspect_weaviate
```

This displays indexed paper versions and sample chunks.

---

## 12. Weaviate Evidence Screenshots

Three screenshots were captured to demonstrate version-aware indexing.

### Screenshot 1 — Version 1 chunk

Shows a `PaperChunk` record containing:

```text
arxiv_id    : 2606.05387
version     : 1
version_key : 2606.05387v1
page        : 1
chunk_index : 0
section     : 1 Introduction
```

The screenshot also includes the extracted document text.

### Screenshot 2 — Version 2 chunk

Shows a corresponding `PaperChunk` record:

```text
arxiv_id    : 2606.05387
version     : 2
version_key : 2606.05387v2
page        : 1
chunk_index : 0
section     : 1 Introduction
```

The screenshot includes the version 2 extracted text.

### Screenshot 3 — Separate paper-version records

Shows both `PaperVersion` records with:

- The same arXiv ID.
- Different version numbers.
- Different version keys.
- Distinct content hashes.
- 138 chunks per version.

Together, these screenshots demonstrate that both paper versions are retained and that chunks carry version-specific metadata.

---

## 13. FastAPI Interface

The project also exposes a REST API.

Start the server.

Windows:

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

macOS/Linux:

```bash
uvicorn app.main:app --reload
```

Open Swagger UI:

http://localhost:8000/docs

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/papers/{arxiv_id}/versions` | List indexed versions |
| POST | `/chat` | Ask a RAG question |
| POST | `/compare` | Compare paper versions |

Example request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the main contribution of arXiv 2606.05387?"}'
```

The API server is optional and is not started automatically by `setup.sh`.

---

## 14. Configuration

Configuration is loaded from the project's environment settings.

On first setup, `.env` is created from `.env.example`.

Key settings include:

- Ollama model selection.
- Embedding model selection.
- Weaviate connection settings.
- Retrieval top-k.
- Configured arXiv paper identifiers.

Refer to `.env.example` and `app/config.py` for the actual supported environment variable names and defaults.

Avoid committing secrets or machine-specific credentials to version control.

---

## 15. Troubleshooting

### Docker is not running

Start Docker Desktop and verify:

```bash
docker info
```

### Port 8080 is already in use

Check running containers:

```bash
docker ps
```

Identify the container using port 8080 and stop it if appropriate.

For example:

```bash
docker stop <container_name>
```

Do not stop unrelated containers unnecessarily.

### Weaviate is not ready

Inspect the container logs:

```bash
docker compose logs --tail=100 weaviate
```

Check the readiness endpoint:

```bash
curl http://localhost:8080/v1/.well-known/ready
```

### Ollama is not responding

Start the Ollama application or service.

Check:

```bash
curl http://localhost:11434/api/tags
```

If necessary, start the server:

```bash
ollama serve
```

### Python version not detected on Windows

Verify:

```bash
py -3.11 --version
```

Use the Windows virtual-environment interpreter:

```bash
.venv/Scripts/python.exe
```

Do not use `.venv/bin/python` on Windows.

### Hugging Face authentication warning

You may see:

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

This warning does not necessarily indicate a failure.

The embedding model can still download and load without a Hugging Face token, subject to service rate limits.

### PDF extraction artifacts

Scientific PDFs may contain equations, multi-column layouts, or unusual fonts.

Extracted text may contain malformed symbols, missing spaces, or reading-order artifacts.

These issues can affect chunk quality, retrieval, and document comparison.

---

## 16. Limitations and Future Improvements

### PDF parsing

PyMuPDF extraction may not preserve the intended reading order of complex academic documents.

Potential improvements:

- Layout-aware PDF extraction.
- Better handling of mathematical equations.
- Sentence-aware chunking.
- Improved section detection.

### Retrieval quality

The system supports dense-only and hybrid keyword/vector retrieval. The included evaluation questions still require manual relevance judgments before retrieval quality can be measured.

Potential improvements:

- Cross-encoder reranking.
- Query rewriting.
- Retrieval relevance evaluation.

### Citation verification

The system validates that cited source labels correspond to retrieved chunks.

However, it does not independently verify that every factual claim is fully supported by the cited passage.

Potential improvements:

- Claim-level evidence verification.
- Natural Language Inference (NLI) checks.
- Citation relevance scoring.
- Automated hallucination evaluation.

### Version comparison

The comparison pipeline uses paragraph alignment, deterministic heuristics, and LLM classification.

Potential limitations:

- PDF extraction may create false positives.
- Paragraph alignment may miss changes.
- Scientific changes may be misclassified.
- Moved text may be incorrectly interpreted.
- Metadata classification relies partly on heuristics.

Future improvements include section-aware alignment, page-level provenance for changed text, stronger semantic verification, and more comprehensive comparison tests.

### Scalability and security

The current implementation is intended for a small assessment corpus.

For production use, improvements would include:

- Object storage for original PDFs and extracted documents.
- Asynchronous ingestion workers.
- Batch processing and retry policies.
- Authentication and authorization.
- Restricted Weaviate network access.
- Monitoring and structured logging.
- Retrieval and generation performance metrics.

The local Weaviate configuration enables anonymous access for development and should not be exposed publicly.

---

## 17. Assessment Deliverables

The assessment submission includes:

1. **Source code:** Ingestion, RAG chatbot, version comparison, FastAPI, and automated tests.
2. **README.md:** Installation, execution, testing, architecture, and limitations.
3. **Weaviate evidence:** Screenshots demonstrating version-tagged chunks and separate paper-version records.
4. **Design write-up:** A separate document of no more than two pages describing the architecture, features, design decisions, limitations, and proposed improvements.

## Summary

This project demonstrates an end-to-end version-aware RAG workflow using local models and a vector database.

The key architectural feature is the separation between normal evidence-grounded question answering and explicit document-version comparison.

Normal research questions use version-filtered dense or hybrid retrieval and Ollama generation, while comparison questions use an evidence-grounded text-difference pipeline.

For a senior AI engineer assessment write-up: *I chose hybrid retrieval because paper questions often combine conceptual paraphrases with exact technical terms, identifiers, and metadata that dense search alone may rank inconsistently. I integrated Weaviate BM25/vector fusion behind a configurable retrieval function while preserving the existing embedding model, latest-version defaults, explicit version filters, citation handling, and separate comparison route. I evaluated dense search against hybrid alpha values of 0.25, 0.50, and 0.75 on the same 14 corpus-grounded questions, using stable chunk UUIDs and warmed, interleaved latency measurements. On the local 951-chunk corpus, median retrieval latency was 2.28 ms for dense and 3.31-3.51 ms for hybrid; manual relevance judgments are now supported through a pooled candidate-review workflow, but quality results must be generated after those labels are completed. The notebook visualizes latency and the judged Pool Recall@K, Hit Rate@K, MRR, and Precision@K. Pool recall is limited to the union of evaluated top-K candidates rather than the entire corpus. The trade-offs are added retrieval latency for this test setup and the need to curate judgments and tune alpha for the corpus and query mix.*

The system has been tested locally with four indexed paper versions, 951 chunks, and 11 passing automated tests.
