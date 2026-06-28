# Setup guide: fine-tune Mistral, then serve it as an API

This is the full path from a fresh Windows PC (RTX 3060 12GB) to a running API in
front of your fine-tuned model. Two parts: A) fine-tuning, B) the API.

Files referenced here already exist in this folder. Detailed sub-steps for Ollama
are in `deploy_hf_ollama.md`; the data-specific notes are in `CYBER_QA_RUNBOOK.md`.

---

## Part A: Fine-tuning setup

### Step 1: Install prerequisites (one time)

1. Install an up-to-date NVIDIA driver.
2. Install Python 3.10 or 3.11 from python.org. Tick "Add Python to PATH".
3. Install Ollama (for serving later) from ollama.com.

### Step 2: Create a Hugging Face token

1. Sign in at huggingface.co, open Settings, then Access Tokens.
2. New token. Choose Write (needed to push your model later), name it, copy it.
3. Open the Mistral-7B-Instruct-v0.3 model page once and accept the terms.
   (Full detail: the "create API key" answer and https://huggingface.co/docs/hub/security-tokens)

### Step 3: Install the project (one time)

Double-click `setup_windows.bat`. It creates the virtual environment, installs the
CUDA build of PyTorch and the libraries, and runs the preflight check.

Or by hand in a terminal opened in this folder:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 4: Preflight (confirm the GPU is usable)

```bat
set HF_TOKEN=hf_your_token
python preflight_check.py
```

You want `CUDA available: True`, your GPU name, and `PASS`. If it says CUDA is
False, the CPU build of torch got installed; rerun the `cu121` line in Step 3.

### Step 5: Train

Double-click `train_windows.bat`, or run:

```bat
python finetune_mistral_qlora.py ^
  --train_file data/cybersecurity_qa_train.jsonl ^
  --eval_file data/cybersecurity_qa_val.jsonl ^
  --output_dir ./cyber-qa-out
```

Roughly 45 to 90 minutes, plus a one-time ~15 GB model download. Result: the adapter
in `cyber-qa-out\adapter`. Add `--push_repo your-username/name --push_private` to
publish it to Hugging Face.

### Step 6: Check quality and hallucination

```bat
python hallucination_check.py --adapter_dir ./cyber-qa-out/adapter --val_file data/cybersecurity_qa_val.jsonl
```

Reads `hallucination_report.md`. High overlap and zero fabrications is good.

---

## Part B: API setup

The API serves your fine-tuned model over HTTP. It calls Ollama under the hood, so
the model must first be in Ollama.

### Step 1: Put the fine-tuned model into Ollama (one time)

Follow `deploy_hf_ollama.md`: merge the adapter, convert to GGUF, quantize, then:

```bat
ollama create cyber-qa -f Modelfile
ollama run cyber-qa "What is a firewall?"
```

Confirm it answers. The model name `cyber-qa` is what the API will call.

### Step 2: Create an API key

```bat
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copy the printed value. This is the key your client apps must send. Keep it secret;
do not commit it.

### Step 3: Start the API

Double-click `start_api_windows.bat` (it asks for the API key), or run:

```bat
.venv\Scripts\activate
pip install -r requirements-api.txt
set API_KEY=the-key-from-step-2
set API_MODEL=cyber-qa
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

The API is now at http://127.0.0.1:8000.

### Step 4: Test it

Easiest: open http://127.0.0.1:8000/docs in a browser (interactive Swagger UI).
Click `/chat`, Authorize is not needed there; send the `x-api-key` header in the
request. Or from a terminal:

```bat
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat -H "x-api-key: YOUR_KEY" -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What is a firewall?\"}]}"
```

You get back `{"answer": "...", "model": "cyber-qa"}`.

### Endpoints

- `GET /health` - checks the API and that Ollama is reachable. No key required.
- `POST /chat` - requires the `x-api-key` header (or `Authorization: Bearer KEY`).
  Body: `{"messages": [{"role": "user", "content": "..."}], "max_tokens": 256,
  "temperature": 0.7}`.

---

## Security notes (NOW-ISMS-AI-001)

- The API requires a key, compares it in constant time, caps input size, and binds to
  localhost only. Do not expose it to the internet without TLS and a reverse proxy
  (and ideally per-client keys and rate limiting).
- Never hardcode the HF token or the API key; both are read from environment
  variables. Use a private HF repo unless the model is cleared for release.
- This model is for explaining and triaging, not for generating exploits. Keep an
  output guardrail in front of any client-facing deployment.
```
