# DataSage dashboard

Run locally:

```bash
python -m pip install -e ".[dashboard,dev]"
streamlit run src/datasage/dashboard.py
```

Upload one CSV, XLSX or Parquet file (50 MiB maximum), optionally choose a target and describe prediction-time context, then click **Run diagnostics**. Review findings by validation status and severity, inspect calculation evidence, and export JSON or HTML.

## Important boundaries

- The uploaded dataset is processed on the machine running Streamlit. Uploaded bytes are written to a temporary file for the existing workflow, then removed after processing.
- Recent runs appear in session history. You can explicitly opt in to saving reports on your local computer under `~/.datasage/investigations/`. Saved reports are not encrypted, and may contain sensitive column names, aggregate counts, and other derived evidence; raw uploaded files are not persisted by this feature. Do not use on shared devices or public deployments. There are no user accounts or multi-tenant security sandbox.
- The dashboard does not certify ML readiness or establish target leakage causally.
- The 50 MiB source-file guard is not sufficient by itself to protect against decompression and in-memory expansion risks. Do not expose the app to untrusted public uploads without further hardening.
- Prediction context is recorded for future work but is not currently used to prove feature availability.

## Human-readable reports

The dashboard now shows plain-language observations, possible interpretations, and suggested checks. Raw JSON metrics remain under an optional technical section, with JSON and styled HTML exports. The category chart summarizes findings, not dataset population.

## Optional LLM explanations

The default human-readable text is deterministic and runs locally. If you explicitly want LLM-written supplementary explanations:

```bash
python -m pip install -e ".[dashboard,llm]"
# Set OPENAI_API_KEY in your local environment (do not commit credentials).
python -m streamlit run src/datasage/dashboard.py
```

Then click **Generate optional AI explanation** on a finding. Only its category, column name, description, aggregate metrics, validation status, and listed limitations are sent to the provider. These data may still be sensitive. This is a separately billed external API request. Never treat generated explanation text as independently verified scientific evidence. The tool does not silently call an LLM when processing uploads. The model can be changed with the `DATASAGE_LLM_MODEL` environment variable.

## Adaptive follow-up stage

The optional JSON prediction context can provide independently recorded availability timestamps and missing-value reason columns. The local supervisor selects eligible probes automatically; the LLM-guided selector is explicitly opt-in and never uploads raw rows. The dashboard displays selected follow-ups, observed evidence and unresolved assumptions separately from numerically validated findings. See [adaptive-investigations.md](adaptive-investigations.md).
