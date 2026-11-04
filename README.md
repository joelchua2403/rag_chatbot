# arXiv RAG Assessment — Weaviate + Ollama

A version-aware arXiv retrieval-augmented generation project for the provided assessment. Includes **ingestion**, **grounded chatbot**, **deterministic text diff + LLM change classification**, **FastAPI**, tests and evidence export.

## Prerequisites — install these first

The one-command script configures the **project**, but it cannot install Docker Desktop, Python, or Ollama for you. Install these tools before running `./setup.sh`. The script is designed for **macOS or Linux**; Windows users should run it inside WSL2 with Docker Desktop WSL integration, or follow the manual commands below.

### 1. Install Python 3.11 or 3.12

**macOS:** Download Python 3.11 or 3.12 from [python.org/downloads](https://www.python.org/downloads/), run the installer, and reopen Terminal. Alternatively, with Homebrew already installed, run `brew install python@3.11`.

**Ubuntu/Debian Linux:** Install a supported Python version and its venv package using your distribution's package manager. For example, on a release providing Python 3.11 packages:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

Verify:

```bash
python3.11 --version
```

Python 3.11 is recommended. If `python3.11` is unavailable, Python 3.12 is also supported. Do not rely on Python 3.13 for this assessment.

### 2. Install Docker Desktop and Docker Compose

**macOS:** Download [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/), select the correct Apple Silicon or Intel installer, install it, and **open Docker Desktop**. Wait until the Docker engine is running.

**Windows / WSL2:** Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/) and enable WSL2 integration for your Linux distribution in Docker Desktop settings.

**Linux:** Install [Docker Engine](https://docs.docker.com/engine/install/) and the [Docker Compose plugin](https://docs.docker.com/compose/install/linux/). Ensure your user can run Docker commands.

Verify in a terminal:

```bash
docker --version
docker compose version
docker info
```

All three commands should succeed. If `docker info` reports that it cannot connect to the daemon, start Docker Desktop or the Docker service before proceeding.

### 3. Install Ollama

**macOS:** Download and install [Ollama for macOS](https://ollama.com/download/mac), then launch the Ollama app.

**Linux:** Follow [Ollama's Linux installation instructions](https://ollama.com/download/linux). The official install command is:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Review third-party installation scripts before executing them. Start the Ollama service if it is not already running (for example, `ollama serve` in a separate terminal).

Verify:

```bash
ollama --version
curl http://localhost:11434/api/tags
```

The setup script will download the configured `qwen2.5:7b` model automatically. You can optionally download it in advance with `ollama pull qwen2.5:7b`.

### 4. Check resources and network access

- Internet access is required for Python packages, the initial embedding-model download, the Ollama model, and arXiv PDFs.
- Allow several gigabytes of free disk space; `qwen2.5:7b` alone is approximately 4.7 GB.
- Ensure sufficient RAM for Docker, SentenceTransformers, and the local model; 16 GB system RAM is a practical starting point, though actual needs vary.
- Ports **8080** (Weaviate HTTP), **50051** (Weaviate gRPC), and **11434** (Ollama) must be available to the services as configured.
- `curl` must be installed (it is included by default on most macOS/Linux systems).

### 5. Get the project source code

If using Git:

```bash
git clone <YOUR_REPOSITORY_URL>
cd arxiv-rag-assessment
```

Or download the submission ZIP, extract it, and open a terminal inside the `arxiv-rag-assessment` folder. Replace `<YOUR_REPOSITORY_URL>` with the actual repository URL; do not type the placeholder literally.

## Quick start — one command (macOS/Linux)

**After completing the installation checks above**, ensure Docker and Ollama are running.

Open a terminal in the project folder and run:

```bash
cd arxiv-rag-assessment
chmod +x setup.sh
./setup.sh
```

The setup script will automatically:

1. Check Docker, Docker Compose, Ollama, and the Python version.
2. Create `.venv` and `.env` (from `.env.example`) if missing.
3. Install Python dependencies from `requirements.txt`.
4. Start Weaviate with Docker Compose and wait for its readiness endpoint.
5. Check that Ollama is running and pull the model configured in `app/config.py`.
6. Download and ingest the configured arXiv paper versions into Weaviate.
7. Print Weaviate index evidence and run the unit tests.
8. Launch the interactive chatbot in the same terminal.

**When the chatbot opens**, type a question and press Enter, for example:

```text
What is the main research contribution of arXiv paper 2606.05387?
According to version 1 of arXiv 2606.05387, what reporting standards do the authors recommend?
```

Type `quit` or `exit` to leave the chatbot. To open it again without repeating setup, run:

```bash
source .venv/bin/activate
python -m scripts.chat
```

**Re-running setup:** `./setup.sh` reuses the existing virtual environment and `.env`, and the ingestion code is designed to skip unchanged indexed versions. The script still checks dependencies, services, ingestion, and tests before launching chat. The first run may take several minutes; arXiv rate limits or network issues may interrupt downloading.

**Troubleshooting:** If Docker is unavailable, open Docker Desktop and retry. If Ollama is unavailable, start the Ollama application or run `ollama serve` in another terminal. If Weaviate does not become ready, inspect `docker compose logs weaviate`. If an existing `.venv` uses the wrong Python version, delete `.venv` and rerun `./setup.sh`.

**Optional API mode:** After setup, in a separate terminal run:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs. The setup script launches the CLI chatbot by default; it does **not** start the API server automatically.

## Manual start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
ollama pull qwen2.5:7b
curl http://localhost:8080/v1/.well-known/ready
python -m scripts.ingest
```

The first ingestion can take time and arXiv may rate-limit requests. arXiv API version history and PDF endpoints are external dependencies. Inspect the output to confirm which versions actually exist; not all papers necessarily have multiple versions.

## Chat
```bash
python -m scripts.chat
```

## Chatbot test prompts (copy and paste)

After `./setup.sh` opens the interactive chatbot (or after running `python -m scripts.chat`), paste the following questions **one at a time** at the `Question>` prompt. The expected behavior is a guide, not a guaranteed exact wording: responses depend on retrieved passages and the local LLM.

| Test | Example prompt | What to check |
|---|---|---|
| Main contribution | `What is the main research contribution of arXiv paper 2606.05387?` | Summarizes the paper using source labels such as `[S1]` and citations to `2606.05387v2` when the latest indexed version is v2. |
| Explicit version 1 | `According to version 1 of arXiv 2606.05387, what reporting standards do the authors recommend?` | Citations reference **`2606.05387v1`**, not v2. |
| Explicit version 2 | `According to version 2 of arXiv 2606.05387, what reporting standards do the authors recommend?` | Citations reference **`2606.05387v2`**, not v1. |
| Specific detail | `What information should researchers report about encoding strategies in arXiv 2606.05387?` | The answer cites relevant retrieved passages; check the cited page in the PDF before treating the claim as verified. |
| Out-of-scope / abstention | `According to arXiv 2606.05387, what was Apple's exact stock price on 15 March 2024?` | States that the indexed paper does not provide this information; **does not invent a stock price** or fabricate supporting citations. |
| Another indexed paper | `Summarize the key findings of arXiv 2605.18821.` | Answer is grounded in citations for **`2605.18821v1`**, rather than passages from another paper. |

**Check the source labels:** The CLI prints a source list below the answer, including arXiv ID, version, and page number. A citation label that matches a retrieved chunk is not, by itself, proof that the claim is factually supported; spot-check important statements against the original PDF.

**Test version comparison separately** (this is a CLI command, **not** a chatbot question):

```bash
python -m scripts.compare 2606.05387 --old 1 --new 2
```

This should compare the two indexed versions and print classified differences. PDF extraction can introduce false positives or miss changes, so inspect the reported before/after passages.

## Compare versions
```bash
python -m scripts.compare 2606.05387 --old 1 --new 2
```
Only use an ID with two indexed versions; otherwise the command correctly reports insufficient versions.

## API
```bash
uvicorn app.main:app --reload
```
Open http://localhost:8000/docs for interactive Swagger docs. Endpoints: `GET /health`, `GET /papers/{arxiv_id}/versions`, `POST /chat`, `POST /compare`.

Example:
```bash
curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{"question":"What experiments were conducted?"}'
```

## Evidence and tests
```bash
python -m scripts.inspect_weaviate | tee weaviate-evidence.txt
pytest -q
```
Run ingestion twice; second run should print `skipped` for unchanged paper versions and the collection counts should remain constant.

## Important limitations
- The application has been exercised locally; a fresh installation on the assessor’s machine has not been independently validated.
- `PaperVersion` stores full extracted text inside Weaviate; convenient for a small corpus, not recommended for large PDFs at scale.
- PDF extraction and paragraph alignment are heuristic; moved paragraphs, equations, figures and tables may produce misleading diffs.
- The classifier makes one Ollama call per changed block and is capped by `--max-changes` (default 30); review output manually.
- Citation tokens are prompt-constrained, not independently verified; production systems should check them.
- The local Weaviate instance enables anonymous access and must not be exposed to the public internet.
- API version history may require inspection or an alternate discovery strategy if arXiv changes its feed behavior.
- The design write-up is a draft, not a validated performance report.
