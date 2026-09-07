# Fine-Tuning: Complete Technical Details

A full reference for the QLoRA fine-tune of Mistral-7B on cybersecurity Q&A. Every
number here comes from the actual run in this repo.

---

## 1. Overview

| | |
|---|---|
| **Goal** | Specialize a general LLM to answer cybersecurity questions in a concise, analyst-style format |
| **Base model** | `mistralai/Mistral-7B-Instruct-v0.3` (7B params, gated on Hugging Face) |
| **Method** | QLoRA (4-bit quantized base + trainable LoRA adapters) |
| **Hardware** | 1 x NVIDIA RTX 3060, 12 GB VRAM (consumer GPU) |
| **Output** | A 161 MB LoRA adapter (not a full 15 GB model) |
| **Training time** | ~21 minutes (2 epochs, 114 steps) |

The deliverable is an **adapter**: a small set of trained weights that layers on top
of the frozen base model. It is useless on its own — at inference you load the base
model + the adapter together.

---

## 2. Environment

| Component | Version |
|---|---|
| Python | 3.13.2 |
| PyTorch | 2.12.1+cu130 (CUDA build) |
| transformers | 5.12.1 |
| trl | 1.7.0 |
| peft | 0.19.1 |
| bitsandbytes | 0.49.2 |
| datasets | 5.0.0 |
| GPU driver / CUDA | 13.1 |
| bf16 supported | Yes |

> Note: the CPU-only build of torch does **not** work — QLoRA needs the CUDA build
> (`pip install torch --index-url https://download.pytorch.org/whl/cu130`) plus
> bitsandbytes for 4-bit and the paged 8-bit optimizer.

---

## 3. Why QLoRA (not a full fine-tune)

A full fine-tune of a 7B model updates all 7 billion weights and needs **~80+ GB of
VRAM** (weights + gradients + Adam optimizer states). A 12 GB 3060 cannot do that.

QLoRA fits it into 12 GB with two ideas:

1. **4-bit quantization of the base model.** The frozen base is compressed from 16-bit
   to 4-bit (~4x smaller). It is only ever read, never updated.
2. **LoRA adapters.** Instead of editing the big weight matrices, small low-rank
   "side" matrices are injected into each layer and only those train — roughly **1%**
   of the parameters.

Result: base model = frozen + 4-bit (cheap to hold); adapters = tiny + trainable.

---

## 4. Quantization config (the frozen base)

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for normally-distributed weights
    bnb_4bit_use_double_quant=True,     # quantize the quantization constants too (extra memory saving)
    bnb_4bit_compute_dtype=torch.bfloat16,   # math runs in bf16 on the 3060
)
```

- **NF4** is a 4-bit datatype designed for neural-net weights; more accurate than plain fp4/int4.
- **Double quantization** shaves a bit more VRAM.
- **compute_dtype = bf16** — matmuls dequantize to bf16 for the forward/backward pass.

---

## 5. LoRA config (the trainable part)

```python
LoraConfig(
    r=16,                 # rank of the low-rank update matrices
    lora_alpha=32,        # scaling (effective LR multiplier ~ alpha/r = 2.0)
    lora_dropout=0.05,    # light regularization on the adapter
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
```

- **r = 16 / alpha = 32** is a standard, safe choice for 7B instruction tuning.
- **target_modules** covers both the **attention** projections (q, k, v, o) and the
  **MLP** projections (gate, up, down) — the full standard target set for Mistral.
- Adapters are also prepared with **gradient checkpointing** (`use_gradient_checkpointing=True`)
  to trade compute for memory, and `use_cache=False` (incompatible with checkpointing).

---

## 6. Data

| | |
|---|---|
| Training examples | **901** (`data/cybersecurity_qa_train.jsonl`) |
| Validation examples | **99** (`data/cybersecurity_qa_val.jsonl`) |
| Format | `{"messages": [system, user, assistant]}` (JSONL, one per line) |

Example record:
```json
{"messages": [
  {"role": "system", "content": "You are a helpful cybersecurity assistant."},
  {"role": "user", "content": "What is the best way to reduce alert fatigue for security analysts?"},
  {"role": "assistant", "content": "Tune detection rules so they produce fewer false positives..."}
]}
```

**System-message folding:** Mistral's chat template has no standalone `system` role, so
the code merges any system message into the first user turn before training. The model
learns to produce the *assistant* text given the *user* text, rendered through the
tokenizer's chat template.

Three input schemas are accepted and normalized to the same internal form:
`{"messages": [...]}`, or `{"instruction", "input", "output"}`.

---

## 7. Training hyperparameters

| Setting | Value | Purpose |
|---|---|---|
| Epochs | 2 (this run) — **1 recommended** | passes over the 901 examples |
| per_device_batch_size | 1 | how many examples per forward pass |
| gradient_accumulation_steps | 16 | **effective batch = 1 x 16 = 16** |
| learning_rate | 2e-4 | step size |
| lr_scheduler | cosine | smooth decay to ~0 |
| warmup_ratio | 0.03 | ramp LR up over the first 3% of steps |
| weight_decay | 0.0 | — |
| optimizer | paged_adamw_8bit | memory-efficient 8-bit Adam |
| max_seq_len | 1024 | longest example the model reads |
| precision | bf16 | compute dtype on the 3060 |
| gradient_checkpointing | True (non-reentrant) | saves VRAM |
| packing | False | one example per sequence |
| eval_strategy | steps, every 20 | produces the validation-loss curve |
| save_steps / save_total_limit | 100 / 2 | checkpointing |
| early_stopping_patience | 3 | stop if val loss doesn't improve for 3 evals; keeps the best checkpoint (`load_best_model_at_end`, `metric_for_best_model=eval_loss`) |
| seed | 42 | reproducibility |

> **Early stopping** directly addresses the overfitting seen below: training halts once
> validation loss stops improving, and the **best** (lowest-val-loss) checkpoint is
> restored as the final adapter — not the last, possibly-overfit one. Set
> `--early_stopping_patience 0` to disable and always train the full `--epochs`.

**Step math:** 901 examples / effective-batch 16 ≈ 57 optimizer steps per epoch. 2
epochs ≈ **114 steps** (matches the run).

---

## 8. What happens during a run (flow)

1. Load base model in 4-bit (NF4) with `device_map="auto"`.
2. `prepare_model_for_kbit_training` + attach LoRA adapters.
3. Render each example through the chat template into a single training string.
4. Train with `SFTTrainer` (TRL), evaluating on the val set every 20 steps.
5. **Save training curves** (`training_curves.png`) + raw log (`trainer_log_history.json`).
6. Save the adapter to `cyber-qa-out/adapter/`.
7. **Auto-run the before/after metrics** (base vs fine-tuned) -> `metrics_report.md` + `.png`.
8. (Optional) `--merge_after` to write a standalone fp16 model; `--push_repo` to upload to HF.

---

## 9. Results (actual)

### 9a. Training dynamics
- **Training loss:** 2.586 -> 0.902 (steady drop)
- **Validation loss:** 1.634 -> **1.503 (lowest, step 60)** -> 1.558 (final)
- **Finding:** mild **overfitting** — val loss bottomed near the end of epoch 1, then
  rose while training loss kept falling. A 3-epoch run overfit worse (val ~1.83).
- **Takeaway:** ~1 epoch is the sweet spot for this dataset; `checkpoint-60` (lowest
  val loss) may be a better model than the final adapter.

### 9b. Before/after quality (99 held-out questions, base vs fine-tuned)
| Metric | Base | Fine-tuned | Change |
|---|---|---|---|
| Mean token-F1 | 0.266 | **0.379** | **+42%** relative |
| ROUGE-L | 0.163 | **0.249** | **+53%** relative |
| Close match (F1>=0.5) | 0.0% | 8.1% | +8.1 pts |

> These measure **word overlap with the reference answers**, so absolute values are
> modest by nature — a correct answer phrased differently still scores low. The
> **consistent improvement over base** is the real signal. "Close match" is a strict
> threshold, NOT real-world correctness (the base model's answers were correct but
> worded differently, so almost none crossed 0.5).

### 9c. Hallucination check (99 questions + 5 fictional probes)
- Mean token-F1: 0.379, ROUGE-L: 0.249 (consistent with 9b)
- **Grounded rate:** 84% (16 of 99 answers flagged below F1 0.30)
- **Fabrication on fake CVEs/products: 4/5** — the model confidently invents details
  for things that don't exist.
- **Takeaway:** fine-tuning teaches style, not a reliable fact store. The fix for
  factual accuracy is **RAG** (retrieve real sources at answer time), not more training.

### 9d. Inference speed (RTX 3060)
- Time to first token: ~0.36 s
- Decode speed: ~7.5 tokens/sec
- Perplexity on held-out set (fine-tuned): ~6.28 (lower = better fit)

---

## 10. Outputs (in `cyber-qa-out/`)

| File | What it is |
|---|---|
| `adapter/` | the trained LoRA adapter (161 MB) — **your model** |
| `adapter/adapter_model.safetensors` | the adapter weights |
| `adapter/adapter_config.json` | the LoRA config used |
| `training_curves.png` | training/validation loss + LR schedule |
| `trainer_log_history.json` | raw per-step metrics |
| `metrics_report.md` / `.png` | before/after F1, ROUGE-L, close-match |
| `hallucination_report.md` | groundedness + fabrication findings |
| `checkpoint-60/`, `checkpoint-114/` | mid-training snapshots (60 = lowest val loss) |

---

## 11. How to run (any machine)

Full pipeline (preflight -> train -> hallucination -> latency), portable across OSes:
```bash
python run_pipeline.py                 # full run, 1 epoch
python run_pipeline.py --epochs 2      # train longer
python run_pipeline.py --skip_train    # re-evaluate an existing adapter
```

Individual steps:
```bash
python preflight_check.py
python finetune_mistral_qlora.py --train_file data/cybersecurity_qa_train.jsonl \
    --eval_file data/cybersecurity_qa_val.jsonl --output_dir ./cyber-qa-out --epochs 1
python eval_metrics.py       --adapter_dir ./cyber-qa-out/adapter --val_file data/cybersecurity_qa_val.jsonl
python hallucination_check.py --adapter_dir ./cyber-qa-out/adapter --val_file data/cybersecurity_qa_val.jsonl
python benchmark_latency.py  --adapter_dir ./cyber-qa-out/adapter --prompts_file data/test_prompts.sample.jsonl
python chat.py               --adapter_dir ./cyber-qa-out/adapter
```

The Hugging Face token is read automatically from `.env` (copy `.env.example` to
`.env` and paste your token). Mistral is gated, so accept the license on the model
page first.

### Inference with the adapter (Python)
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

---

## 12. Key lessons

1. **Consumer hardware is enough.** QLoRA (4-bit base + LoRA adapters) put 7B
   fine-tuning on a 12 GB 3060 — no cloud, no A100.
2. **Read the curves, not just the final loss.** Rising validation loss while training
   loss falls = overfitting. Caught here; fixed by fewer epochs.
3. **Fine-tuning teaches style, not facts.** Measured 4/5 fabrication on fictional
   CVEs -> the real fix for factual accuracy is RAG.
4. **Metrics need interpretation.** Word-overlap metrics (F1/ROUGE-L) undersell
   correct-but-differently-worded answers; report *relative* gains and read samples.
5. **90% is engineering.** CUDA build matching, VRAM budgeting, secret handling, and
   reproducible evals were the bulk of the work; the training call was one line.

---

## 13. Next step

Layer **RAG** on top: retrieve real security sources at answer time for facts, and keep
this fine-tune for tone and format. Style from fine-tuning, facts from retrieval.
