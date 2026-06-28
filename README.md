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

## Train

```bash
python finetune_mistral_qlora.py \
  --train_file data/train.jsonl \
  --eval_file data/eval.jsonl \
  --output_dir ./mistral7b-qlora-out
```

Output: a LoRA adapter in `mistral7b-qlora-out/adapter`. Add `--merge_after` to also write a standalone merged fp16 model.

## Tuning for memory

If you hit out-of-memory on the 3060: lower `--max_seq_len` (e.g. 512), keep `--per_device_batch_size 1`, and raise `--grad_accum`. The defaults (4-bit NF4, gradient checkpointing, `paged_adamw_8bit`, batch 1 x grad-accum 16) are already conservative.

## Inference with the adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map="auto")
model = PeftModel.from_pretrained(base, "mistral7b-qlora-out/adapter")
tok = AutoTokenizer.from_pretrained("mistral7b-qlora-out/adapter")

msgs = [{"role": "user", "content": "What is the NOX platform?"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(inputs, max_new_tokens=256)[0], skip_special_tokens=True))
```
