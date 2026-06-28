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
export HF_TOKEN=hf_xxx     # use a WRITE token so the push step works
```

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

## Important caveat

Fine-tuning teaches style and behavior, not a reliable, current fact store. For
factual security Q&A that must stay accurate, the robust fix for hallucination is
RAG (retrieve real sources at answer time), with this fine-tune layered on for tone
and format. See `cybersecurity_training_plan.md` for that architecture.
```
