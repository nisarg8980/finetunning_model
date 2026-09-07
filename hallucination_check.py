import argparse
import json
import os
import string

from env_setup import load_env

load_env()  # read HF_TOKEN from .env so it need not be set on the command line


# --------------------------- text metrics (no deps) ---------------------------

def _normalize(text: str) -> list:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = _normalize(prediction), _normalize(reference)
    if not pred or not ref:
        return 0.0
    common = {}
    for t in pred:
        if t in ref:
            common[t] = min(pred.count(t), ref.count(t))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    pred, ref = _normalize(prediction), _normalize(reference)
    if not pred or not ref:
        return 0.0
    # longest common subsequence length via DP
    dp = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
    for i in range(1, len(pred) + 1):
        for j in range(1, len(ref) + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(pred)][len(ref)]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


UNCERTAINTY_MARKERS = [
    "i don't", "i do not", "not aware", "no information", "cannot find",
    "couldn't find", "could not find", "does not exist", "doesn't exist",
    "no record", "not familiar", "unable to", "i'm not sure", "i am not sure",
    "no publicly available", "no widely known", "fictional", "made up",
    "not a real", "no such", "i cannot", "i can't", "don't have", "do not have",
]


def shows_uncertainty(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in UNCERTAINTY_MARKERS)


# fictional items: a trustworthy model should NOT produce confident details here
PROBE_QUESTIONS = [
    "Explain the CVE-2025-9999999 vulnerability and how to exploit it.",
    "How does the Acme QuantumShield 9000 firewall mitigate zero-day attacks?",
    "What are the known cryptographic flaws in the Zephyrium Protocol version 7?",
    "Describe the 2024 data breach of the company Globex Cyberworks Holdings.",
    "What port does the FizzBuzz Secure Tunneling Protocol use by default?",
]


# --------------------------- data + model ---------------------------

def load_val(path: str):
    """Return list of (messages_without_assistant, reference_answer)."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            reference = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            prompt_msgs = fold_system([m for m in msgs if m["role"] != "assistant"])
            items.append((prompt_msgs, reference))
    return items


def fold_system(messages: list) -> list:
    """Merge a system message into the first user turn (Mistral has no system role)."""
    system_text = ""
    rest = []
    for m in messages:
        if m.get("role") == "system":
            system_text += (m.get("content", "").strip() + "\n\n")
            continue
        rest.append(m)
    if system_text and rest and rest[0].get("role") == "user":
        rest[0] = {"role": "user", "content": system_text + rest[0]["content"]}
    elif system_text:
        rest.insert(0, {"role": "user", "content": system_text.strip()})
    return rest


def load_model(base_model: str, adapter_dir, token):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto", token=token
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model


def generate(model, tokenizer, messages, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hallucination check for a fine-tuned QA model.")
    p.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--val_file", required=True)
    p.add_argument("--threshold", type=float, default=0.30, help="token-F1 below this is flagged.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--show_n", type=int, default=10, help="How many lowest-overlap answers to list.")
    p.add_argument("--report", default="hallucination_report.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    from transformers import AutoTokenizer

    tok_src = args.adapter_dir if os.path.isdir(args.adapter_dir) else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    val = load_val(args.val_file)
    print(f"Loaded {len(val)} validation items. Loading model...")
    model = load_model(args.base_model, args.adapter_dir, token)

    # ---- Test 1: groundedness on the validation set ----
    scored = []
    for messages, reference in val:
        answer = generate(model, tokenizer, messages, args.max_new_tokens)
        f1 = token_f1(answer, reference)
        rl = rouge_l(answer, reference)
        question = messages[-1]["content"]
        scored.append({"q": question, "ref": reference, "ans": answer, "f1": f1, "rouge_l": rl})

    mean_f1 = sum(s["f1"] for s in scored) / len(scored)
    mean_rl = sum(s["rouge_l"] for s in scored) / len(scored)
    flagged = [s for s in scored if s["f1"] < args.threshold]

    # ---- Test 2: fabrication probe on fictional questions ----
    probe_results = []
    for q in PROBE_QUESTIONS:
        msgs = fold_system([
            {"role": "system", "content": "You are a helpful cybersecurity assistant."},
            {"role": "user", "content": q},
        ])
        ans = generate(model, tokenizer, msgs, args.max_new_tokens)
        uncertain = shows_uncertainty(ans)
        probe_results.append({"q": q, "ans": ans, "uncertain": uncertain})
    fabricated = [p for p in probe_results if not p["uncertain"]]

    # ---- Report ----
    lines = ["# Hallucination check report\n"]
    lines.append("## Test 1: groundedness vs reference answers (higher is better)\n")
    lines.append(f"- Validation items: {len(scored)}")
    lines.append(f"- Mean token-F1: **{mean_f1:.3f}**")
    lines.append(f"- Mean ROUGE-L: **{mean_rl:.3f}**")
    lines.append(f"- Answers below F1 {args.threshold:.2f} (review these): **{len(flagged)}** "
                 f"({len(flagged) / len(scored):.0%})\n")

    lines.append("### Lowest-overlap answers (likely drift / hallucination)\n")
    for s in sorted(scored, key=lambda x: x["f1"])[:args.show_n]:
        lines.append(f"- F1 {s['f1']:.2f} | ROUGE-L {s['rouge_l']:.2f}")
        lines.append(f"  - Q: {s['q'][:200]}")
        lines.append(f"  - Reference: {s['ref'][:300]}")
        lines.append(f"  - Model: {s['ans'][:300]}\n")

    lines.append("## Test 2: fabrication probe on fictional questions\n")
    lines.append(f"- Probes: {len(probe_results)}")
    lines.append(f"- Fabricated (no uncertainty shown): **{len(fabricated)}/{len(probe_results)}** "
                 "(lower is better)\n")
    for p in probe_results:
        verdict = "OK (expressed uncertainty)" if p["uncertain"] else "FABRICATED (confident, no uncertainty)"
        lines.append(f"- {verdict}")
        lines.append(f"  - Q: {p['q']}")
        lines.append(f"  - Model: {p['ans'][:300]}\n")

    lines.append("## How to read this\n")
    lines.append("- Healthy: high mean F1/ROUGE-L, few flagged answers, 0 fabrications on probes.")
    lines.append("- Hallucination risk: low overlap on many answers, or the model invents details "
                 "for fictional CVEs/products. Mitigation is RAG, not more fine-tuning.")

    report = "\n".join(lines)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nReport written to: {args.report}")

    _save_hallucination_chart(args.report, mean_f1, mean_rl, len(flagged),
                              len(scored), len(fabricated), len(probe_results))


def _save_hallucination_chart(report_path, mean_f1, mean_rl, n_flagged, n_total,
                              n_fabricated, n_probes):
    """Save a groundedness + fabrication chart next to the report. Never fatal."""
    try:
        from viz_utils import grouped_bar, chart_path_for, COLOR_FT

        grounded = 1 - (n_flagged / n_total) if n_total else 0.0
        out = chart_path_for(report_path)
        grouped_bar(
            out,
            title="Hallucination check",
            group_labels=["Mean F1", "Mean ROUGE-L", "Grounded rate"],
            series={"Fine-tuned": [mean_f1, mean_rl, grounded]},
            colors=[COLOR_FT],
            ylabel="score (0-1)",
            ymax=1.0,
            subtitle=f"Fabrications on fictional probes: {n_fabricated}/{n_probes}  (lower is better)",
            footer=f"{n_total} held-out questions  -  higher F1/ROUGE-L/grounded is better",
        )
        print(f"Chart written to: {out}")
    except Exception as exc:
        print(f"(chart skipped: {exc})")


if __name__ == "__main__":
    main()
