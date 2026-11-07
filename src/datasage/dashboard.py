"""Streamlit dashboard for evidence-based diagnostics.

Uploaded files are handled in temporary storage and deleted after analysis.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from datasage.ingestion import MAX_BYTES, SUPPORTED, load_dataset
from datasage.explanations import explain_finding, overall_summary
from datasage.history_store import save_investigation, list_investigations
from datasage.llm_explanations import explain_with_llm
from datasage.reporting import render_html
from datasage.workflow import build_workflow


def analyze_upload(uploaded_bytes: bytes, filename: str, target_column: str | None = None,
                   prediction_context: str | None = None, investigation_mode: str = "rules") -> dict:
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
        state["investigation_mode"] = investigation_mode
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

    st.set_page_config(page_title="DataSage | Dataset Health Studio", page_icon="🔎", layout="wide")
    st.markdown("""
    <style>
      .block-container {max-width:1180px; padding-top:1.7rem; padding-bottom:3.5rem}
      [data-testid="stMetric"] {background:rgba(105,145,185,.10);padding:18px;border-radius:16px;border:1px solid rgba(140,160,185,.17)}
      div[data-testid="stExpander"] {border-radius:14px;border:1px solid rgba(140,160,185,.25)}
      .stButton button[kind="primary"] {border-radius:11px;font-weight:700;padding:0.65rem 1.1rem}
    </style>
    """, unsafe_allow_html=True)
    st.title("DataSage | Dataset Health Studio")
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

    st.subheader("1 · Understand your dataset")
    c1, c2 = st.columns(2)
    c1.metric("Rows", metadata.rows)
    c2.metric("Columns", metadata.columns)
    st.dataframe(preview.head(20), use_container_width=True)
    target = st.selectbox("Prediction target (optional)", [None] + columns,
                          format_func=lambda value: "(No target)" if value is None else str(value))
    st.caption("Selecting a target activates limited additional checks; it does not guarantee more findings.")
    context = st.text_area(
        "Prediction context / source evidence (optional)",
        placeholder='Describe prediction-time information, or supply JSON evidence metadata.',
        help=('For independently checkable evidence, use JSON such as '
              '{"prediction_time_column":"prediction_at",'
              '"feature_time_columns":{"outcome_proxy":"outcome_proxy_recorded_at"},'
              '"missing_reason_columns":{"income":"income_missing_reason"}}. '
              "Timestamps and source-reason labels must be audited independently.")
    )
    planner_label = st.selectbox(
        "Follow-up investigation supervisor",
        ["Evidence-driven local checks (no API)", "LLM-guided selection (opt-in API)"]
    )
    investigation_mode = "llm" if planner_label.startswith("LLM") else "rules"
    if investigation_mode == "llm" and not os.getenv("OPENAI_API_KEY"):
        st.warning("LLM planning requires OPENAI_API_KEY. Choose local checks without a key.")
    # Without user action, do not run repeated computations on each filter change.
    if st.button("Run diagnostics", type="primary"):
        try:
            report = analyze_upload(raw, uploaded.name,
                                    target, context, investigation_mode)
            st.session_state["datasage_report"] = report
            st.session_state["datasage_file"] = (uploaded.name, metadata.fingerprint, target, context, investigation_mode)
            history = st.session_state.setdefault("datasage_history", [])
            history.insert(0, {
                "file": uploaded.name, "fingerprint": metadata.fingerprint,
                "target": target or "(No target)", "summary": dict(report["summary"]),
                "report": report,
            })
            del history[10:]
        except (ValueError, OSError, UnicodeError, KeyError, TypeError) as exc:
            st.error(f"Analysis failed: {exc}")
            return
    if st.session_state.get("datasage_file") != (uploaded.name, metadata.fingerprint, target, context, investigation_mode):
        st.info("Run diagnostics to view findings for this dataset.")
        return
    report = st.session_state.get("datasage_report")
    if report is None:
        return

    st.subheader("2 · What DataSage discovered")
    st.info(overall_summary(report))
    st.caption("Plain-language explanations are based on evidence templates, not generated AI claims.")
    category_counts = {}
    for items in report["findings"].values():
        for finding in items:
            label = finding["category"].replace("_", " ").title()
            category_counts[label] = category_counts.get(label, 0) + 1
    if category_counts:
        st.subheader("Types of observations")
        chart = pd.DataFrame({"Finding type": list(category_counts), "Count": list(category_counts.values())})
        st.bar_chart(chart.set_index("Finding type"))
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
            explanation = explain_finding(item)
            st.markdown("**What we observed**")
            st.write(explanation["observation"])
            st.markdown("**What this means**")
            st.write(explanation["meaning"])
            st.markdown("**What to check next**")
            st.write(explanation["next_step"])
            st.caption(f"Status: {item['validation_status']} · Severity: {item['severity']} · Confidence: {item['confidence']}")
            if os.getenv("OPENAI_API_KEY"):
                if st.button("Generate optional AI explanation", key=f"llm-{metadata.fingerprint}-{target}-{item['finding_id']}"):
                    try:
                        st.session_state[f"llm_text_{metadata.fingerprint}_{target}_{item['finding_id']}"] = explain_with_llm(item)
                    except (ValueError, ImportError, RuntimeError, OSError) as exc:
                        st.error(f"AI explanation unavailable: {exc}")
                generated = st.session_state.get(f"llm_text_{metadata.fingerprint}_{target}_{item['finding_id']}")
                if generated:
                    st.info("AI-generated interpretation (not independently verified): " + generated)
            with st.expander("Technical details (optional)"):
                st.json(item["evidence"])
                st.write("Validation:", item["validation_reason"])
                st.write("Limitations:", item.get("limitations", []))
    st.subheader("3 · Focused follow-up investigations")
    selected_checks = report.get("follow_up_plan", [])
    planner = report.get("follow_up_planner", {})
    if selected_checks:
        st.caption("Supervisor: " + planner.get("source", "evidence_rules")
                   + " · Selected: " + ", ".join(selected_checks))
        st.write(planner.get("reason", ""))
        for item in report.get("follow_up_investigations", []):
            with st.expander(item["check"].replace("_", " ").title()
                             + " · " + item["status"].replace("_", " ")):
                st.write(item["interpretation"])
                st.write("What to check next:", item["next_step"])
                st.json(item["observations"])
    else:
        st.info("No additional evidence-driven checks were eligible for this dataset.")
    st.caption("A timestamp comparison or recorded collection reason is not proof of causal leakage or a missingness mechanism.")

    st.subheader("4 · Recent investigations")
    st.caption("History is kept only in this browser session, not permanently saved.")
    for index, entry in enumerate(st.session_state.get("datasage_history", [])):
        with st.expander(f"Run {index + 1} · {entry['file']} · Target: {entry['target']}"):
            st.write(overall_summary(entry["report"]))
            st.caption(f"Dataset fingerprint: {entry['fingerprint'][:16]}…")
    st.markdown("**Save an investigation permanently (local device only)**")
    st.caption("Opt-in: saves report metadata, findings and metrics to your computer, not the raw dataset. Do not use on shared machines with confidential data.")
    if st.button("Save this report on my computer"):
        try:
            save_investigation(report, uploaded.name, target)
            st.success("Saved locally. You can reopen it under Saved investigations.")
        except (ValueError, OSError, TypeError) as exc:
            st.error(f"Save failed: {exc}")
    if st.checkbox("Show saved investigations"):
        for record in list_investigations():
            with st.expander(f"{record['saved_at_utc'][:16]} · {record['source_filename']} · {record['target'] or '(No target)'}"):
                st.write(overall_summary(record["report"]))
                st.download_button("Download this saved report",
                    data=json.dumps(record["report"], indent=2, ensure_ascii=False),
                    file_name=f"datasage-saved-{record['id']}.json", mime="application/json",
                    key=f"saved-{record['id']}")
    st.subheader("5 · Export")
    st.caption("Read the HTML report, or export JSON for reproducibility.")
    st.download_button("Download JSON report",
                       data=json.dumps(report, indent=2, ensure_ascii=False),
                       file_name="datasage-report.json", mime="application/json")
    st.download_button("Download HTML report", data=render_html(report),
                       file_name="datasage-report.html", mime="text/html")


if __name__ == "__main__":
    main()
