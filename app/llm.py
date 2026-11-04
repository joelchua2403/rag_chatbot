import json
import httpx
from app.config import settings

def generate(system: str, prompt: str, json_mode=False) -> str:
    payload = {"model": settings.ollama_model, "stream": False,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": prompt}],
               "options": {"temperature": 0.1}}
    if json_mode:
        payload["format"] = "json"
    with httpx.Client(timeout=300) as client:
        r = client.post(f"{settings.ollama_base_url.rstrip('/')}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]
