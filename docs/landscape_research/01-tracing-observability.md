# Competitive Landscape: LLM Tracing & Observability

## Key Players

| Platform | License | Trace Model | Unique Strength |
|---|---|---|---|
| Arize Phoenix | OSS (no gates) | OpenInference (OTel) | Clustering, anomaly detection, replay |
| Langfuse | MIT | Observations (span, generation, agent, tool) | Prompt versioning + trace linking |
| LangSmith | Proprietary | Runs in trace trees | Expert annotation calibrates auto-eval |
| Braintrust | Proprietary | DAG of spans (OTel) | Low-score-to-dataset feedback loop |
| W&B Weave | OSS (SDK) | Ops and Calls | Leaderboards, Model Registry link |
| OpenLLMetry | Apache 2.0 | Standard OTel spans | Pure OTel, any backend, 4 languages |
| HoneyHive | Proprietary | Session events (OTel) | SOC2/HIPAA, multi-agent edge cases |

## Key Patterns

1. **Integration converges on three approaches:** decorator-based (@observe, @weave.op), OTel-native (OTLP to any backend), client wrappers (wrap OpenAI/Anthropic)
2. **Evaluation is the differentiator, not tracing.** Tracing is commoditized via OpenTelemetry.
3. **Feedback loops matter most:** Production traces become test cases, human annotations calibrate automated scorers, low-scoring outputs feed back into datasets.

## Gaps Evalyn Should Fill

- **Trace replay workflow** (Phoenix): pull production traces into a playground, test prompt variants against real data
- **DAG trace model** (Braintrust): spans can have multiple parents, more flexible than trees for complex agents
- **Evaluation leaderboards** (W&B Weave): org-wide comparison dashboard
- **Production-to-test-case pipeline** (Braintrust, LangSmith): one-click conversion of production traces to evaluation datasets
- **SOC2/HIPAA compliance** (HoneyHive): governance features for enterprise adoption

## Impact on Evalyn Design

- Evalyn already has OpenInference span conventions and OTel bridge - maintain and expand this
- Add "trace replay" as a first-class workflow: extract inputs from spans, re-execute with different model
- Add "trace-to-dataset" one-click pipeline (partially exists via build-dataset)
- Consider DAG support for multi-parent spans in future trace model evolution

*Sources: Arize Phoenix docs, Langfuse docs, LangSmith docs, Braintrust docs, W&B Weave docs, OpenLLMetry GitHub, HoneyHive docs*
