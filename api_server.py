#!/usr/bin/env python
"""
Secured FastAPI server that exposes your fine-tuned Mistral model as a chat API.

It forwards requests to a local Ollama instance that serves your model, so Ollama
handles the GPU and model loading while this layer provides a clean, authenticated
HTTP API for your applications (consistent with our Python/FastAPI stack).

Prerequisites:
  - Ollama running with your fine-tuned model loaded (see deploy_hf_ollama.md), for
    example a model named "cyber-qa".
  - Environment variables:
      API_KEY     required. Clients must send it. Generate one with:
                  python -c "import secrets; print(secrets.token_urlsafe(32))"
      API_MODEL   Ollama model name to call (default "cyber-qa").
      OLLAMA_URL  Ollama base URL (default http://localhost:11434).

Run:
    pip install -r requirements-api.txt
    set API_KEY=your-generated-key
    uvicorn api_server:app --host 127.0.0.1 --port 8000
    # interactive docs at http://127.0.0.1:8000/docs

Security (NOW-ISMS-AI-001, OWASP): API-key auth with constant-time comparison,
input size limits, request timeouts, and localhost binding by default. Do not expose
this to the internet without TLS and a reverse proxy. Never hardcode the API key.
"""

import os
import secrets
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
API_MODEL = os.environ.get("API_MODEL", "cyber-qa")
API_KEY = os.environ.get("API_KEY")
MAX_CHARS = 8000  # cap total input size to avoid abuse

app = FastAPI(title="Fine-tuned Mistral API", version="1.0.0")


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    answer: str
    model: str


def require_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    """Fail closed: refuse if no server key is configured; constant-time compare."""
    if not API_KEY:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Server API_KEY is not configured.")
    provided = x_api_key
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:]
    if not provided or not secrets.compare_digest(provided, API_KEY):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key.")


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Ollama not reachable: {exc}")
    return {"status": "ok", "ollama": "reachable", "model": API_MODEL}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat(req: ChatRequest):
    total_chars = sum(len(m.content) for m in req.messages)
    if total_chars > MAX_CHARS:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Input too large ({total_chars} > {MAX_CHARS} chars).",
        )
    payload = {
        "model": API_MODEL,
        "messages": [m.model_dump() for m in req.messages],
        "stream": False,
        "options": {"temperature": req.temperature, "num_predict": req.max_tokens},
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Ollama request failed: {exc}")
    answer = (data.get("message") or {}).get("content", "").strip()
    return ChatResponse(answer=answer, model=API_MODEL)
