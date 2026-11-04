from fastapi import FastAPI, HTTPException
from app.models import ChatRequest, ComparisonRequest
from app.rag import answer
from app.comparison import compare
from app.store import connect, available_versions

app = FastAPI(title="arXiv Version-aware RAG")

@app.get("/health")
def health():
    client = connect()
    try:
        return {"weaviate_ready": client.is_ready()}
    finally:
        client.close()

@app.get("/papers/{arxiv_id}/versions")
def versions(arxiv_id: str):
    client = connect()
    try:
        return {"arxiv_id": arxiv_id, "versions": available_versions(client, arxiv_id)}
    finally:
        client.close()

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        return answer(request.question, request.arxiv_id, request.version)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

@app.post("/compare")
def gap(request: ComparisonRequest):
    try:
        return compare(request.arxiv_id, request.old_version, request.new_version)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
