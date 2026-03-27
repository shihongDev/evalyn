# Competitive Landscape: Security, Governance & Enterprise

## LLM Security & Guardrails

| Tool | Approach | Key Feature |
|---|---|---|
| NeMo Guardrails | 5 rail types, Colang DSL, streaming | Production-grade, NVIDIA-backed |
| Guardrails AI | Validators + RAIL spec, 50+ validators | Pythonic validator composition |
| Lakera Guard | Prompt injection API (acquired by Check Point Sept 2025) | Enterprise security vendor backing |
| LLM Guard (Protect AI) | Scanner pipeline for input/output | Modular scanner architecture |
| Garak | 150+ probes, 3000+ prompts | Most comprehensive probe library |

## Governance & Compliance

**Key findings:**
- EU AI Act GPAI obligations in effect (Aug 2025): models >10^23 FLOPs must document evaluation methodology, benchmarks, and security controls. Penalties up to 7% of global revenue.
- NIST AI RMF and ISO 42001 are the two leading compliance frameworks for AI evaluation.
- Model card documentation is deeply inconsistent: analysis of 32K cards found 947 unique section names; safety info absent in 80%+ of cards.
- AuditableLLM's hash-chain approach is the leading research framework for audit trails, but no production tool offers this out of the box.

**Enterprise evaluation patterns:**
- Banks: focus on regulatory compliance, bias testing, explainability
- Healthcare: HIPAA-compliant evaluation environments, clinical accuracy metrics
- Scale AI pivoting to evaluation+annotation convergence for enterprise

## Data Privacy

- PII detection: regex patterns + NER models (spaCy, Presidio) are standard
- Anonymization: replace with synthetic equivalents (Faker library), not just masking
- Differential privacy for evaluation metrics is still research-stage
- Platform handling: HoneyHive (SOC2/HIPAA), Arize (cloud isolation), most others lack formal compliance

## Multi-Tenant & Team Features

| Feature | LangSmith | Langfuse | Others |
|---|---|---|---|
| RBAC | Yes (org/workspace) | Yes (project-level) | Varies |
| Annotation assignment | Annotation queues with SME assignment | Manual | Limited |
| Cost allocation per team | Requires manual tagging | Basic project-level | Gap across all |
| SSO/SAML | Enterprise tier | Enterprise tier | Enterprise tier |

## Gaps for Evalyn

1. **PII redaction pipeline** - integrate regex + optional NER, configurable strategy (mask/hash/remove)
2. **Evaluation audit trail** - append-only JSONL with hash-chain (AuditableLLM pattern)
3. **Compliance reporting** - auto-generate evaluation documentation for EU AI Act / NIST AI RMF
4. **Basic RBAC** - project-level access control for multi-user teams (future, when web dashboard exists)
5. **Data governance metadata** - classification tags on datasets (PII-present, internal-only, approved-for-eval)
6. **Lightweight guardrail metrics** - integrate basic prompt injection detection as an objective metric

## Impact on Evalyn Design

Evalyn is local-first and single-user, which simplifies security (no network attack surface). Key additions:
- PII redaction as a pre-storage hook (already in ROADMAP)
- Audit trail as append-only JSONL (already in DESIGN)
- Compliance report export for EU AI Act documentation requirements (new)
- Prompt injection detection as an objective metric (new)

*Sources: NeMo Guardrails docs, Guardrails AI docs, Lakera/Check Point announcement, LLM Guard GitHub, Garak GitHub, EU AI Act text, NIST AI RMF, ISO 42001, AuditableLLM paper, HoneyHive compliance docs*
