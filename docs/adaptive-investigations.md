# Adaptive, evidence-seeking follow-up investigations

DataSage now has two stages:

1. **Independent core diagnostics** run in parallel: profiling, missingness, anomaly, and distribution. With a selected target, a conservative ML-risk screen also runs.
2. **Follow-up supervisor** chooses focused checks from the *actual* diagnostic findings and any source metadata provided by the user. Each check returns a status, observations, a caveat, and a concrete next step. This stage runs before the existing independent evidence validator and report reconciliation.

The follow-up supervisor supports:
- **Local evidence-driven mode (default)**: deterministic selection of eligible checks, no external calls.
- **Opt-in LLM-guided mode**: a bounded model can select a subset of evidence-eligible checks and explain its choice, but cannot create arbitrary tools. The user must select LLM mode and supply `OPENAI_API_KEY`. It receives aggregate counts, column names, and finding categories—not raw dataset rows. Column names and metadata may still be sensitive.

This is an *adaptive follow-up*, not a fully unconstrained autonomous scientist. The four core checks remain part of the fixed safety-first screening pipeline.

## Source evidence for stronger leakage and missingness investigations

A dataset alone usually cannot prove *why* a value is missing or whether a feature was genuinely available at prediction time. Supplying independently collected metadata can enable narrower checks.

In the dashboard **Prediction context / source evidence** field, optionally enter structured JSON such as:

```json
{
  "prediction_time_column": "prediction_at",
  "feature_time_columns": {
    "outcome_proxy": "outcome_proxy_recorded_at"
  },
  "missing_reason_columns": {
    "income": "income_missing_reason"
  }
}
```

- **temporal_availability** compares the recorded feature timestamp with the prediction cutoff on rows where both timestamps and the feature are present. It reports how many recorded values are later than the cutoff, along with rows lacking comparable timestamps. **This is timing evidence, not definitive proof of target leakage**; the source timestamp semantics and production feature availability still need independent verification.
- **missingness_provenance** checks which missing values have an accompanying recorded collection reason and whether reasons are attached to present values. **These are source-recorded explanations, not causal proof of MCAR, MAR, MNAR, or why a value is absent**. Collection logs and source-system audits are still needed.
- **target_proxy_review** identifies features flagged by existing target-agreement or identifier heuristics and records the further availability evidence that must be supplied. Matching the target does not prove leakage.

If the structured metadata is absent, DataSage can still identify statistical candidate problems and request supporting provenance. Free-form prediction context remains accepted, but the app will not infer operational timestamps or collection reasons from prose.

## Controls, reproducibility, and limits

- Up to three focused checks per run. No arbitrary code execution or unapproved tools.
- Selected checks, reason, source (`evidence_rules` or `opt_in_llm`), and results appear under `follow_up_plan`, `follow_up_planner`, and `follow_up_investigations` in the exported JSON report.
- Failures to obtain a valid bounded LLM plan are surfaced as errors; no quiet substitution of a claimed LLM plan.
- Original uploaded data remains read-only and is processed with the dashboard's temporary-file handling.
- Deterministic validation continues to apply to core numeric findings; source-provenance observations are displayed separately and are **not** automatically labeled as numerically verified findings.
- The existing optional local report-saving feature can persist derived evidence, including sensitive column names. Avoid using confidential data without considering that risk.

### Current research limits

An observational dataset does not identify a missing-data mechanism by itself, nor can a high correlation establish temporal target leakage. These outputs should guide a human investigation, not issue a leakage/no-leakage certificate. The LLM chooses from existing tools; it cannot prove the scientific interpretation of their outputs.
