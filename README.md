# DataSage

**A Supervisor-Based Multi-Agent Framework for Evidence-Based Dataset Diagnostics and ML Readiness**

DataSage is designed to investigate dataset quality risks with specialized diagnostic agents coordinated by a supervisor. Statistical tools generate evidence; validation checks findings before explanations or recommendations are presented.

## Principles
- Evidence before explanations
- Deterministic calculations, structured and traceable findings
- Read-only dataset analysis, with no automatic data modifications
- No unsupported conclusions about leakage, missingness mechanisms, or model readiness
- Human approval before destructive operations

## Roadmap
1. Foundation: ingestion, schemas, supervisor skeleton and tests
2. Profiling and missingness diagnostics
3. Anomaly, distribution and ML-risk agents
4. Validation and evidence reconciliation
5. Dashboard and export
6. Controlled research evaluation and baselines

The initial implementation is developed in feature branches and reviewed through pull requests.

## Evidence-driven follow-up investigations

After the existing parallel core diagnostics, DataSage can choose targeted follow-up checks from observed findings. The default supervisor runs locally using evidence-driven rules; an optional **LLM-guided planner** selects only permitted checks when explicitly enabled in the dashboard with `OPENAI_API_KEY`.

Focused checks review prediction-time feature timestamps, recorded missing-value reasons, and target-proxy risk signals. They report evidence, unsupported assumptions, and the next source logs needed; **they do not claim to prove missingness causality or target leakage from observations alone**. See [adaptive investigation documentation](docs/adaptive-investigations.md) for an example JSON evidence specification and exact safety limits.
