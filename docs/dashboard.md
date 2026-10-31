# DataSage dashboard

Run locally:

```bash
python -m pip install -e ".[dashboard,dev]"
streamlit run src/datasage/dashboard.py
```

Upload one CSV, XLSX or Parquet file (50 MiB maximum), optionally choose a target and describe prediction-time context, then click **Run diagnostics**. Review findings by validation status and severity, inspect calculation evidence, and export JSON or HTML.

## Important boundaries

- The uploaded dataset is processed on the machine running Streamlit. Uploaded bytes are written to a temporary file for the existing workflow, then removed after processing.
- This version keeps the report in the Streamlit session. It does not provide user accounts, long-term investigation history, or a multi-tenant security sandbox.
- The dashboard does not certify ML readiness or establish target leakage causally.
- The 50 MiB source-file guard is not sufficient by itself to protect against decompression and in-memory expansion risks. Do not expose the app to untrusted public uploads without further hardening.
- Prediction context is recorded for future work but is not currently used to prove feature availability.
