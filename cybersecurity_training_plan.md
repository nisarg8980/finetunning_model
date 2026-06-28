# Training plan: cybersecurity assistant model

Goal: a model that helps with five tasks: explain security concepts and answer
questions, summarize scan and pentest findings, triage and classify findings,
rewrite findings for clients, and support pen testing (reporting and methodology,
not exploit generation).

## Headline recommendation: hybrid, not a single fine-tune

No single technique covers all five tasks well. Use both:

- RAG (retrieval-augmented generation) for knowledge that changes or must be cited:
  concept Q&A, CVE and advisory lookups, and pentest methodology references.
- QLoRA fine-tuning for fixed behavior, format, and tone: report summarization in
  your house style, triage and classification, and client-facing rewriting.

Fine-tuning is the wrong tool for factual knowledge that changes daily. It bakes
facts into weights, goes stale immediately, and hallucinates confidently. RAG keeps
a searchable knowledge base and feeds current, cited context at answer time. Fine-
tuning is the right tool for teaching the model how to respond, not what the latest
facts are.

## Approach per task

| Task | Best approach | Why | Data you need |
| --- | --- | --- | --- |
| Explain concepts / Q&A | RAG (light fine-tune for tone only) | Knowledge changes; citations matter | Curated KB: OWASP, CWE, MITRE ATT&CK, NIST, NVD/CVE, your sanitized docs |
| Summarize scan/pentest findings | Fine-tune | Output format and house style are fixed behavior | Pairs: raw findings -> house-style summary (anonymized) |
| Triage / classify findings | Fine-tune | Consistent labeling is a learned behavior | Pairs: finding -> severity/label, with balanced classes |
| Client-facing explanations | Fine-tune | Tone transformation is a learned behavior | Pairs: technical text -> plain-language version |
| Pen testing (reporting + methodology) | Hybrid | RAG for methodology knowledge; fine-tune for report writeups | Methodology KB (RAG) + sanitized report pairs (fine-tune) |

Pen testing scope: the model assists with finding writeups, remediation guidance,
CVSS scoring, framework mapping (OWASP, MITRE ATT&CK, PTES), and scoping documents.
It is not trained to generate working exploits, payloads, or malware. Keep offensive
payload generation out of both the data and the use cases (NOW-ISMS-AI-001).

## Data strategy

Quality beats quantity for QLoRA. Aim for 500 to 2,000 clean, consistent examples
per task to start; a few thousand excellent examples outperform millions of noisy
ones. For classification, balance the classes so no severity dominates.

Mandatory anonymization. Never train on real customer data, pentest findings of
real systems, credentials, IPs, hostnames, or client names. Run every example
through `scrub_dataset.py` and then review by hand. Over-redaction is safer than a
leak. If unsure a dataset is safe, ask Angelo Derksen before using it.

Format. Use the chat JSONL the training script already accepts. Tag the task in the
user turn so one model can serve all behavioral tasks, for example a line that
starts with "Task: triage" or "Task: client_summary". Hold out 10 to 15 percent of
each task as an eval set the model never sees in training.

## Method and models

Fine-tuning: QLoRA with the existing `finetune_mistral_qlora.py`. Start with a single
multi-task adapter trained on all three behavioral tasks together (with task tags),
which is simplest to deploy. If one task underperforms, split it into its own adapter.

Base model: start with Mistral-7B-Instruct-v0.3 (what you already have). When you get
larger GPUs, evaluate a bigger or newer instruct base (for example Mistral-Small-24B,
Llama-3.1-8B, or Qwen2.5-14B) on your eval set before committing. A strong general
instruct base plus good data usually beats a niche "security" base for these tasks.

Hardware: the RTX 3060 is fine for QLoRA 7B prototyping. For a larger base or faster
iteration, use Azure ML with an A100 or H100. Keep training data and model artifacts
inside the Azure tenant to stay within the ISO 27001 boundary.

RAG stack: since our platform is MongoDB-based, MongoDB Atlas Vector Search is the
natural fit for the knowledge base, with an embedding model to index OWASP, CWE,
MITRE ATT&CK, NIST, the NVD CVE feed, and your sanitized internal docs. Azure AI
Search is a managed alternative.

## Evaluation

- Triage / classification: accuracy, precision, recall, and a confusion matrix on the
  held-out set. These are objective; track them every run.
- Summarization and rewriting: held-out perplexity plus LLM-as-judge plus human spot
  checks. Use `evaluate_model.py` for the side-by-side base-vs-fine-tuned view.
- RAG Q&A: retrieval hit rate and answer faithfulness (does the answer match the
  retrieved source, and is the citation correct).

## Phased roadmap

1. Baseline. Prompt the base model with few-shot examples on all five tasks and
   measure. It may already be good enough for some tasks; do not fine-tune those.
2. RAG first. Stand up the knowledge base for concept Q&A and methodology. This
   removes the largest source of hallucination.
3. Data and fine-tune. Build and anonymize the three behavioral datasets, train one
   multi-task QLoRA adapter, and evaluate against the baseline.
4. Combine. Route knowledge tasks through RAG and behavioral tasks through the fine-
   tuned model. Evaluate end to end and iterate.
5. Deploy. Serve on Azure with access controls, audit logging, output guardrails
   (block exploit/malware output), and human review of client-facing text.

## Governance

Policy NOW-ISMS-AI-001 governs this work. No real customer or personal data in
training sets; anonymize and review. Keep the scope defensive and documentation-
focused. Add output guardrails so the deployed model refuses requests for working
exploits or malware. Log and review client-facing output before it reaches clients.
Escalate any data-handling doubt to the security officer, Angelo Derksen.
