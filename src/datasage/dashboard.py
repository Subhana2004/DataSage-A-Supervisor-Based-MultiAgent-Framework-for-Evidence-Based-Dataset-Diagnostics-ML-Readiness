"""Streamlit dashboard for evidence-based diagnostics.

Uploaded files are handled in temporary storage and deleted after analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from datasage.ingestion import MAX_BYTES, SUPPORTED, load_dataset
from datasage.reporting import render_html
from datasage.workflow import build_workflow


def analyze_upload(uploaded_bytes: bytes, filename: str, target_column: str | None = None,
                   prediction_context: str | None = None) -> dict:
    """Analyze uploaded bytes using existing read-only workflow."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError("Unsupported file format")
    if len(uploaded_bytes) > MAX_BYTES:
        raise ValueError("File exceeds 50 MiB limit")
    with tempfile.TemporaryDirectory(prefix="datasage-") as folder:
        path = Path(folder) / ("upload" + suffix)
        path.write_bytes(uploaded_bytes)
        frame, _ = load_dataset(path)
        if target_column is not None and target_column not in frame.columns:
            raise ValueError("Target column absent")
        state = {"dataset_path": str(path)}
        if target_column is not None:
            state["target_column"] = target_column
        if prediction_context:
            state["prediction_context"] = prediction_context
        return build_workflow().invoke(state)["final_report"]


def filter_findings(report: dict, statuses: list[str] | None = None,
                    severities: list[str] | None = None) -> list[dict]:
    """Filter findings without modifying verification status."""
    statuses = statuses if statuses is not None else ["verified", "rejected", "not_verified"]
    rows = []
    for status in statuses:
        for item in report["findings"].get(status, []):
            if severities is None or item["severity"] in severities:
                rows.append(item)
    return rows


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="DataSage", page_icon="🔎", layout="wide")
    st.title("DataSage")
    st.caption("Evidence-based dataset diagnostics · statistical evidence is not proof of a dataset defect.")
    uploaded = st.file_uploader("Upload CSV, XLSX or Parquet (max 50 MiB)",
                                type=["csv", "xlsx", "parquet"])
    if uploaded is None:
        st.info("Upload a dataset to begin.")
        return
    raw = uploaded.getvalue()
    if len(raw) > MAX_BYTES:
        st.error("File exceeds the 50 MiB limit.")
        return
    # Select columns using existing ingestion, rather than guessing CSV schema.
    try:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in SUPPORTED:
            raise ValueError("Unsupported file format")
        with tempfile.TemporaryDirectory(prefix="datasage-preview-") as folder:
            path = Path(folder) / ("preview" + suffix)
            path.write_bytes(raw)
            preview, metadata = load_dataset(path)
            columns = list(preview.columns)
    except (ValueError, OSError, UnicodeError, ImportError, KeyError) as exc:
        st.error(f"Unable to read dataset: {exc}")
        return

    st.subheader("Dataset overview")
    c1, c2 = st.columns(2)
    c1.metric("Rows", metadata.rows)
    c2.metric("Columns", metadata.columns)
    st.dataframe(preview.head(20), use_container_width=True)
    target = st.selectbox("Prediction target (optional)", ["None"] + columns)
    context = st.text_area("Prediction context (optional)",
                           placeholder="Describe what is known at prediction time.")
    # Without user action, do not run repeated computations on each filter change.
    if st.button("Run diagnostics", type="primary"):
        try:
            report = analyze_upload(raw, uploaded.name,
                                    None if target == "None" else target, context)
            st.session_state["datasage_report"] = report
            st.session_state["datasage_file"] = (uploaded.name, metadata.fingerprint)
        except (ValueError, OSError, UnicodeError, KeyError, TypeError) as exc:
            st.error(f"Analysis failed: {exc}")
            return
    if st.session_state.get("datasage_file") != (uploaded.name, metadata.fingerprint):
        st.info("Run diagnostics to view findings for this dataset.")
        return
    report = st.session_state.get("datasage_report")
    if report is None:
        return

    st.subheader("Diagnostic report")
    cols = st.columns(3)
    for col, status in zip(cols, ("verified", "rejected", "not_verified")):
        col.metric(status.replace("_", " ").title(), report["summary"][status])
    st.warning("Verified refers to supporting evidence only; no ML-readiness certificate is implied.")
    statuses = st.multiselect("Validation status", ["verified", "rejected", "not_verified"],
                              default=["verified", "not_verified"])
    severities = st.multiselect("Severity", ["low", "medium", "high", "critical"],
                                default=["low", "medium", "high", "critical"])
    rows = filter_findings(report, statuses, severities)
    st.caption(f"{len(rows)} findings shown")
    for item in rows:
        with st.expander(f"{item['title']} · {item.get('column') or 'dataset'} · {item['validation_status']}"):
            st.write(item["description"])
            st.write("Severity:", item["severity"], "| Confidence:", item["confidence"])
            st.json(item["evidence"])
            st.write("Validation:", item["validation_reason"])
            st.write("Limitations:", item.get("limitations", []))
            st.write("Recommendations:", item.get("recommendations", []))
    st.download_button("Download JSON report",
                       data=json.dumps(report, indent=2, ensure_ascii=False),
                       file_name="datasage-report.json", mime="application/json")
    st.download_button("Download HTML report", data=render_html(report),
                       file_name="datasage-report.html", mime="text/html")


if __name__ == "__main__":
    main()
