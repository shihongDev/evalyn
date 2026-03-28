# Technical Deep Dive: Security, Governance & PII

## NeMo Guardrails Internals

**Colang 2.0:** Event-driven with await/start/match keywords. 5 rail types: input, dialog, retrieval, execution, output. Streaming mode: chunked processing (200 token chunks, 50 token context buffer). Overhead: ~0.5s for 5 parallel GPU-accelerated rails.

**Performance optimization:** Dialog single-call mode consolidates 3 LLM calls into 1. Embeddings-only mode eliminates LLM call for intent classification.

## Prompt Injection Detection

**Regex patterns:** 4 tiers (instruction override, role injection, prompt extraction, encoding signals). Catches ~80% of simple injections, trivially bypassed.

**Lakera Guard:** Purpose-built classifier (not LLM), trained on 100K+ attacks daily from Gandalf, <50ms response, 100+ languages. Now part of Check Point.

**Rebuff:** 4-layer pipeline (heuristic scan 0.1ms, LLM classifier 500ms, vector DB 50ms, canary tokens passive). Self-hardening: detected attacks stored for future matching.

**State of art accuracy:** Hidden state features achieve 99.6% in-domain. BUT: "When Benchmarks Lie" (2026 paper) showed classifiers exploit dataset provenance artifacts, not real attack patterns. Lakera explicitly argues LLM-as-judge fails for injection defense because judge is vulnerable to same attacks.

## EU AI Act (Current Status)

**In effect (Aug 2025):** Article 53 - all GPAI providers must maintain technical documentation, training data summary. Article 55 - systemic risk models (>10^25 FLOPS) must do standardized evaluations, red-teaming, incident reporting.

**Annex IV:** Most prescriptive eval documentation requirement globally - dated/signed test logs with specific metrics.

**Penalties:** Up to 15M EUR or 3% worldwide turnover (Aug 2026 enforcement).

**Open-source exemption:** Models with publicly available params exempt from documentation (Articles 53.1-53.2) but NOT from copyright compliance or training data summary.

## PII Detection

**Microsoft Presidio:** 69+ entity types, pattern-based + NER-based. Custom recognizers via PatternRecognizer or LocalRecognizer subclass. 30% F-score improvement with context words.

**Embedding PII leakage:** Embedding inversion attacks recover 93-98% of text from ada-002 embeddings. Clinical data: sex (88%), diseases (70%), symptoms (82%) extracted. Best defense: Eguard (3.5-5.6% inversion F1 while maintaining 93-97% downstream accuracy).

**Key insight:** Embeddings are NOT safe stores for PII. Vector stores need same access controls as raw PII.

## Audit Trail Design

**Hash-chain:** entry_hash = SHA256(timestamp || actor || action || resource || metadata || previous_hash). Genesis block with "GENESIS" prev_hash.

**Tamper-evidence:** Merkle tree for batch verification, root anchoring in separate trust domain (HSM-signed or external).

**Retention:** HIPAA 6y, SOX 7y, GDPR "as necessary", legal 15y, PCI-DSS 1y.

*Sources: NeMo Guardrails docs/GitHub, Lakera Guard docs, Rebuff GitHub, EU AI Act Articles 53/55/Annex IV/XI, Presidio docs, Eguard paper, AuditableLLM paper*
