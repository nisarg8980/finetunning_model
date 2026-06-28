# Fine-tuning POC: security finding triage

Goal of this POC: show that fine-tuning adapts Mistral-7B to classify pentest
findings by severity in our exact format, and that it beats the base model on a
held-out set. Everything uses synthetic data, so there is no customer or sensitive
data involved (NOW-ISMS-AI-001).

## What the POC proves

- Fine-tuning teaches the model our task and output format (`Severity: <label>`).
- Measurable result: base-model accuracy vs fine-tuned accuracy on held-out findings.
- The full workflow runs end to end: data -> train -> evaluate -> deploy in Ollama.

It does not claim the model now "knows" all security facts. Factual, changing
knowledge is a separate RAG track (see ../cybersecurity_training_plan.md).

## One-time setup

```bash
cd ..                       # repo root (FineTunning_model)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export HF_TOKEN=hf_xxx       # accept the Mistral license on the HF Hub first
```

## Step 1: Generate the synthetic dataset

```bash
cd poc
python make_triage_dataset.py --per_class 60
# -> triage_train.jsonl (~240 examples) and triage_eval.jsonl (~60 held out)
```

## Step 2: Fine-tune (QLoRA, runs on the RTX 3060)

```bash
cd ..
python finetune_mistral_qlora.py \
  --train_file poc/triage_train.jsonl \
  --output_dir poc/out \
  --epochs 3
```

Expect roughly 15 to 30 minutes on a 3060, plus a one-time ~15 GB model download on
the first run. The LoRA adapter lands in `poc/out/adapter`.

## Step 3: Measure accuracy, base vs fine-tuned (the headline slide)

```bash
cd poc
python eval_triage_accuracy.py \
  --adapter_dir out/adapter \
  --eval_file triage_eval.jsonl \
  --compare_base
```

This prints and saves `poc_triage_report.md` with overall accuracy for both models,
the improvement, and a per-class breakdown. That report is your main POC evidence.

## Step 4 (optional): Live demo in Ollama

```bash
cd ..
python finetune_mistral_qlora.py --train_file poc/triage_train.jsonl --output_dir poc/out --merge_after
# then convert poc/out/merged to GGUF and load it (see ../deploy_hf_ollama.md)
ollama create triage-poc -f Modelfile
ollama run triage-poc "Task: triage\nClassify the severity ... \n\nFinding: Stored XSS in /search executes in other users' sessions."
```

## Talking points for the presentation

1. Problem: triaging findings by hand is slow and inconsistent.
2. Approach: QLoRA fine-tune of Mistral-7B on labeled findings; fits a single GPU.
3. Result: show the accuracy table (base vs fine-tuned) and a couple of live answers.
4. Data safety: 100 percent synthetic for the POC; production uses anonymized data
   via the scrubbing step and stays inside our Azure or ISO 27001 boundary.
5. Next steps: add RAG for factual Q&A, expand to summarization and client-facing
   rewriting, evaluate a larger base model on rented or Azure ML GPUs.

## Honest caveats to mention

- Synthetic data makes the task cleaner than reality; production accuracy will differ.
- A 7B model on a 3060 is a prototype scale; production may want a larger base.
- Fine-tuning is for behavior and format, not for current threat intelligence.
