#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

info() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Missing $1. Install it, then rerun ./setup.sh"; }

need docker
need ollama
need curl

if ! docker info >/dev/null 2>&1; then
  fail 'Docker daemon is not running. Start Docker Desktop, then rerun ./setup.sh.'
fi
if ! docker compose version >/dev/null 2>&1; then
  fail 'Docker Compose v2 is required.'
fi

if [[ ! -d .venv ]]; then
  info 'Finding Python 3.11 or 3.12'
  PYTHON=''
  for candidate in python3.11 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12)' >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
  [[ -n "$PYTHON" ]] || fail 'Python 3.11 or 3.12 is required.'
  "$PYTHON" -m venv .venv
fi
source .venv/bin/activate
python -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12)' || fail 'Existing .venv has the wrong Python version. Delete .venv and rerun.'

if [[ ! -f .env ]]; then
  [[ -f .env.example ]] || fail 'Missing .env.example'
  cp .env.example .env
  info 'Created .env from .env.example'
fi

info 'Installing Python dependencies'
python -m pip install -r requirements.txt

info 'Starting Weaviate'
docker compose up -d

info 'Waiting for Weaviate readiness (up to 120 seconds)'
ready=0
for ((i=1; i<=60; i++)); do
  if curl -fsS --max-time 2 http://localhost:8080/v1/.well-known/ready >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || fail 'Weaviate did not become ready. Check: docker compose logs weaviate'

info 'Checking Ollama'
if ! curl -fsS --max-time 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
  if [[ "$(uname -s)" == 'Darwin' ]]; then
    open -a Ollama >/dev/null 2>&1 || true
  fi
  if ! curl -fsS --max-time 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
    fail 'Ollama is not responding at localhost:11434. Start Ollama (or run ollama serve), then rerun.'
  fi
fi

MODEL="$(python -c 'from app.config import settings; print(settings.ollama_model)')"
info "Ensuring Ollama model $MODEL is available"
ollama pull "$MODEL"

info 'Ingesting arXiv papers (unchanged versions should be skipped)'
python -m scripts.ingest

info 'Checking Weaviate contents'
python -m scripts.inspect_weaviate

info 'Running unit tests'
python -m pytest -q

info 'Setup complete. Launching interactive chatbot (type quit to exit)'
python -m scripts.chat
