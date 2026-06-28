#!/usr/bin/env python
"""
Generate a synthetic, balanced dataset for a security-finding triage POC.

Every example is fully synthetic (no real customer, asset, or pentest data), so it
is safe to use and present under NOW-ISMS-AI-001. Each line is chat JSONL:
    user:      Task: triage + the finding + the allowed labels
    assistant: "Severity: <label>\nRationale: <one line>"

Output: poc/triage_train.jsonl and poc/triage_eval.jsonl

Usage:
    python make_triage_dataset.py --per_class 60 --eval_frac 0.2
"""

import argparse
import json
import os
import random

random.seed(42)

ASSETS = [
    "the customer portal", "an internal API endpoint", "the admin dashboard",
    "a public-facing web server", "the authentication service", "a file upload endpoint",
    "the reporting module", "a legacy CMS instance", "the payment microservice",
    "an exposed staging environment", "the user profile page", "the password reset flow",
]
ENDPOINTS = ["/api/v1/users", "/login", "/admin", "/upload", "/search", "/export",
             "/account/settings", "/api/v2/orders", "/reset-password", "/files"]
PARAMS = ["id", "q", "redirect", "file", "user", "token", "callback", "next"]
SOFTWARE = ["nginx 1.18.0", "Apache 2.4.49", "OpenSSL 1.0.2", "jQuery 1.8.3",
            "WordPress 5.2", "PHP 7.2", "Tomcat 7.0", "Struts 2.3"]

# Templates per severity. {a}=asset, {e}=endpoint, {p}=param, {s}=software.
TEMPLATES = {
    "Critical": [
        "Unauthenticated remote code execution on {a} via a crafted request to {e}.",
        "SQL injection in the {p} parameter of {e} allows full database extraction.",
        "{a} exposes an admin interface accessible with default credentials.",
        "A private signing key was found hardcoded and reachable through {e}.",
        "Insecure deserialization on {e} permits arbitrary command execution.",
        "Authentication can be fully bypassed on {a}, granting administrator access.",
    ],
    "High": [
        "Stored cross-site scripting in {e} executes in other users' sessions.",
        "Server-side request forgery via the {p} parameter on {e} reaches internal services.",
        "Privilege escalation lets a standard user gain admin rights on {a}.",
        "Sensitive personal data is returned without authorization from {e}.",
        "An IDOR on {e} lets a user read other users' records by changing {p}.",
        "{a} accepts a weak password reset token that can be brute forced.",
    ],
    "Medium": [
        "Reflected cross-site scripting in the {p} parameter of {e}.",
        "Cross-site request forgery protection is missing on {e}.",
        "No rate limiting on {e} enables credential stuffing against {a}.",
        "{a} uses a weak TLS configuration that allows outdated ciphers.",
        "Verbose error messages on {e} disclose stack traces and internal paths.",
        "Directory listing is enabled on {a}, exposing internal file names.",
    ],
    "Low": [
        "{a} is missing the Content-Security-Policy security header.",
        "Session cookies on {a} are set without the Secure flag.",
        "Clickjacking is possible because X-Frame-Options is absent on {e}.",
        "{a} discloses its software version ({s}) in HTTP responses.",
        "The HSTS header is not set on {a}.",
        "Autocomplete is enabled on the password field at {e}.",
    ],
    "Informational": [
        "{a} returns a server banner revealing {s}.",
        "A robots.txt file on {a} lists internal-only paths.",
        "{a} exposes a non-sensitive build timestamp in its headers.",
        "Best practice note: {a} could enable HSTS preloading.",
        "An informational comment referencing {s} was found in the page source of {e}.",
        "{a} supports TLS 1.2; consider enabling TLS 1.3 as well.",
    ],
}

RATIONALES = {
    "Critical": "Direct path to full system or data compromise; exploit impact is severe.",
    "High": "Serious impact on confidentiality or integrity, exploitable with limited effort.",
    "Medium": "Meaningful weakness that needs a precondition or user interaction to exploit.",
    "Low": "Minor hardening gap with limited direct impact.",
    "Informational": "No direct security impact; noted for awareness or best practice.",
}

LABELS = list(TEMPLATES.keys())
INSTRUCTION = (
    "Task: triage\n"
    "Classify the severity of the following penetration test finding as exactly one of: "
    "Critical, High, Medium, Low, Informational.\n\nFinding: "
)


def fill(template: str) -> str:
    return template.format(
        a=random.choice(ASSETS), e=random.choice(ENDPOINTS),
        p=random.choice(PARAMS), s=random.choice(SOFTWARE),
    )


def build_examples(per_class: int) -> list:
    rows = []
    for sev, templates in TEMPLATES.items():
        seen = set()
        attempts = 0
        while len([r for r in rows if r["severity"] == sev]) < per_class and attempts < per_class * 50:
            attempts += 1
            finding = fill(random.choice(templates))
            if finding in seen:
                continue
            seen.add(finding)
            rows.append({
                "severity": sev,
                "messages": [
                    {"role": "user", "content": INSTRUCTION + finding},
                    {"role": "assistant", "content": f"Severity: {sev}\nRationale: {RATIONALES[sev]}"},
                ],
            })
    random.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=60)
    ap.add_argument("--eval_frac", type=float, default=0.2)
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    rows = build_examples(args.per_class)

    # Stratified split so eval stays balanced across classes.
    train, eval_ = [], []
    for sev in LABELS:
        subset = [r for r in rows if r["severity"] == sev]
        n_eval = max(1, int(len(subset) * args.eval_frac))
        eval_.extend(subset[:n_eval])
        train.extend(subset[n_eval:])
    random.shuffle(train)
    random.shuffle(eval_)

    def write(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps({"messages": r["messages"]}) + "\n")

    train_path = os.path.join(args.out_dir, "triage_train.jsonl")
    eval_path = os.path.join(args.out_dir, "triage_eval.jsonl")
    write(train_path, train)
    write(eval_path, eval_)
    print(f"Wrote {len(train)} train and {len(eval_)} eval examples.")
    print(f"  {train_path}")
    print(f"  {eval_path}")


if __name__ == "__main__":
    main()
