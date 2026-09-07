# Mistral-7B QLoRA fine-tuning

Fine-tune `Mistral-7B-Instruct-v0.3` on chat / instruction data, sized for a single RTX 3060 (12 GB).

## Why QLoRA, not a full fine-tune

A full fine-tune of a 7B model needs roughly 80 GB+ of VRAM (weights + gradients + optimizer states). An RTX 3060 has 12 GB, so a full fine-tune cannot run on it. QLoRA loads the base model in 4-bit and trains small adapters, which fits in 12 GB and gives equivalent results for instruction tuning. If you later get multi-GPU or Azure ML (A100/H100) access and want a true full fine-tune script, ask and I will produce one.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export HF_TOKEN=...   # Mistral weights are gated; request access on the Hugging Face Hub first
```

## Data format

JSONL, one example per line. Either schema works:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"instruction": "...", "input": "...", "output": "..."}
```

See `data/train.sample.jsonl`. Do not put real customer or personal data in the training set (NOW-ISMS-AI-001); use anonymized or synthetic examples.

## Run the full pipeline (one command, any OS)

The whole workflow — preflight → train → hallucination check → latency — runs from a single portable Python command (Linux, macOS, or Windows):

```bash
python run_pipeline.py
```

It re-invokes the same Python interpreter it was launched with, so it uses whatever virtual environment you run it in. Training also produces the before/after metrics report and the training curves automatically. All outputs land in `./cyber-qa-out`.

```bash
python run_pipeline.py --epochs 2      # train longer
python run_pipeline.py --skip_train    # just re-evaluate an existing adapter
```

### Or run each step manually

```bash
# 0. Verify the environment (GPU, libraries, token)
python preflight_check.py

# 1. Fine-tune (writes adapter, training_curves.png, and metrics_report automatically)
python finetune_mistral_qlora.py \
  --train_file data/cybersecurity_qa_train.jsonl \
  --eval_file data/cybersecurity_qa_val.jsonl \
  --output_dir ./cyber-qa-out --epochs 1

# 2. Before/after metrics (F1, ROUGE-L, close-match) — base vs fine-tuned
python eval_metrics.py --adapter_dir ./cyber-qa-out/adapter --val_file data/cybersecurity_qa_val.jsonl

# 3. Hallucination check (groundedness + fake-CVE fabrication probe)
python hallucination_check.py --adapter_dir ./cyber-qa-out/adapter --val_file data/cybersecurity_qa_val.jsonl

# 4. Latency benchmark (tokens/sec, time-to-first-token)
python benchmark_latency.py --adapter_dir ./cyber-qa-out/adapter --prompts_file data/test_prompts.sample.jsonl

# 5. Chat with it interactively
python chat.py --adapter_dir ./cyber-qa-out/adapter
```

Output: a LoRA adapter in `cyber-qa-out/adapter`. Add `--merge_after` to the train step to also write a standalone merged fp16 model.

## Tuning for memory

If you hit out-of-memory on the 3060: lower `--max_seq_len` (e.g. 512), keep `--per_device_batch_size 1`, and raise `--grad_accum`. The defaults (4-bit NF4, gradient checkpointing, `paged_adamw_8bit`, batch 1 x grad-accum 16) are already conservative.

## Inference with the adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map="auto")
model = PeftModel.from_pretrained(base, "cyber-qa-out/adapter")
tok = AutoTokenizer.from_pretrained("cyber-qa-out/adapter")

msgs = [{"role": "user", "content": "What is a firewall?"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=256)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```
