# Fine-tune on your cybersecurity QA data, publish, and check hallucination

Your data is in `data/cybersecurity_qa_train.jsonl` (901 examples) and
`data/cybersecurity_qa_val.jsonl` (99 held out). Format is `{"messages": [system,
user, assistant]}`. The scripts fold the system message into the user turn
automatically, because Mistral's chat template has no separate system role.

The scan found no IPs, emails, credentials, keys, or company names, so this dataset
is safe to fine-tune and publish. Still, prefer a private repo unless you intend to
release it (NOW-ISMS-AI-001).

## One-time setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Hugging Face token (set once via .env).** Instead of exporting `HF_TOKEN` every
time, copy the template and paste your token in — every script loads it automatically:

```bash
cp .env.example .env      # Windows: copy .env.example .env
# then edit .env and set: HF_TOKEN=hf_xxx   (use a WRITE token if you push to the Hub)
```

`.env` is git-ignored, so your token is never committed. To override it for a single
run, set `HF_TOKEN` in the shell — the real environment wins over `.env`.

## Step 1: Fine-tune and push to Hugging Face in one command

```bash
python finetune_mistral_qlora.py \
  --train_file data/cybersecurity_qa_train.jsonl \
  --eval_file  data/cybersecurity_qa_val.jsonl \
  --output_dir ./cyber-qa-out \
  --push_repo your-username/mistral7b-cyber-qa \
  --push_private
```

This trains the QLoRA adapter, then uploads the adapter (small, a few hundred MB) to
your HF repo. On a 3060 expect roughly 45 to 90 minutes for 901 examples over 3
epochs, plus the one-time base model download (~15 GB).

If you want a standalone model on the Hub instead of an adapter, add `--push_merged`
(this merges to fp16 first and uploads ~15 GB).

### Training graphs (loss, validation loss, LR schedule)

Training writes two files into `--output_dir` automatically:

- `training_curves.png` — training loss + validation loss (top) and the learning-rate
  schedule (bottom).
- `trainer_log_history.json` — the raw numbers behind the plot.

Validation loss appears because `--eval_file` is passed; it is measured every
`--eval_steps` (default 20 — lower it for more points on the curve). For **live**
graphs while training runs, add `--tensorboard` and, in another terminal:

```bash
tensorboard --logdir cyber-qa-out/runs
```

### Automatic before/after metrics

Because `--eval_file` is passed, training **also runs the before/after metrics
automatically** once the adapter is saved, writing `cyber-qa-out/metrics_report.md`
(Accuracy, F1, Response Quality — base vs fine-tuned). Add `--skip_post_eval` to turn
this off, or `--metrics_threshold 0.6` to change the accuracy cutoff. This adds a few
minutes because it generates answers for all 99 held-out questions with both models.
See Step 3b for what the report contains and how to re-run it standalone.

## Step 2: Use the model on another PC

Install the same `requirements.txt` and set `HF_TOKEN` on the other machine, then:

Adapter (default push). The base model downloads automatically and the adapter loads
on top:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map="auto")
model = PeftModel.from_pretrained(base, "your-username/mistral7b-cyber-qa")
tok = AutoTokenizer.from_pretrained("your-username/mistral7b-cyber-qa")

msgs = [{"role": "user", "content": "You are a helpful cybersecurity assistant.\n\nWhat is a firewall?"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(inputs, max_new_tokens=256)[0][inputs.shape[1]:], skip_special_tokens=True))
```

Merged model (if you used `--push_merged`). Self-contained, load directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("your-username/mistral7b-cyber-qa", device_map="auto")
tok = AutoTokenizer.from_pretrained("your-username/mistral7b-cyber-qa")
```

For an Ollama deployment on the other PC instead, follow `deploy_hf_ollama.md`.

## Step 3: Check for hallucination

```bash
python hallucination_check.py \
  --adapter_dir ./cyber-qa-out/adapter \
  --val_file data/cybersecurity_qa_val.jsonl
```

This runs two tests and writes `hallucination_report.md`:

1. Groundedness: it answers all 99 held-out questions and scores each answer against
   your reference answer (token-F1 and ROUGE-L). It lists the lowest-overlap answers
   so you can read where the model drifted.
2. Fabrication probe: it asks about fictional CVEs, products, and events. A good model
   says it does not know; a hallucinating model invents confident details. The report
   counts how many it fabricated.

How to read it: high overlap and zero fabrications is healthy. Low overlap on many
answers, or fabricated details on the fictional probes, means hallucination risk.

## Step 3b: Before/after Accuracy, F1, and Response Quality

Training already produced `cyber-qa-out/metrics_report.md` automatically (see above).
Run this only if you want to re-generate it, change the threshold, or evaluate a
different adapter:

```bash
python eval_metrics.py \
  --adapter_dir ./cyber-qa-out/adapter \
  --val_file data/cybersecurity_qa_val.jsonl
```

This answers all 99 held-out questions with **both** the base model ("before") and
your fine-tuned model ("after"), then writes `metrics_report.md` with a comparison
table:

- **Accuracy** — fraction of answers whose token-F1 vs the reference is at or above
  `--threshold` (default 0.5). Free-form QA has no single gold string, so "correct"
  means "overlaps the reference strongly enough".
- **F1** — mean token-level F1 against the reference answer.
- **Response quality** — mean ROUGE-L, plus a semantic cosine score if you
  `pip install sentence-transformers`.

The "Change" column shows what the fine-tune actually bought you over the base model.
Add `--skip_base` to evaluate only the fine-tuned model, or `--limit 20` for a quick
smoke test.

## Step 4: Benchmark latency and throughput

```bash
python benchmark_latency.py \
  --adapter_dir ./cyber-qa-out/adapter \
  --prompts_file data/test_prompts.sample.jsonl \
  --compare_base
```

This writes `latency_report.md` with, per prompt and averaged: time to first token
(TTFT, the "lag" before anything appears) and steady-state tokens/sec after that. Use
`--compare_base` to see the (usually small) overhead the LoRA adapter adds over the
base model. This only measures speed, not answer quality — pair it with
`evaluate_model.py` and `hallucination_check.py`.

## Important caveat

Fine-tuning teaches style and behavior, not a reliable, current fact store. For
factual security Q&A that must stay accurate, the robust fix for hallucination is
RAG (retrieve real sources at answer time), with this fine-tune layered on for tone
and format. See `cybersecurity_training_plan.md` for that architecture.
```
