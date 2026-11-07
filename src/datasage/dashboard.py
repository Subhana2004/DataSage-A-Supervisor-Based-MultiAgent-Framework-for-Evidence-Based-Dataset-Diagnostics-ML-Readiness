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

# Public deployments must not persist visitors' reports on a shared host.
PUBLIC_DEMO = os.getenv("DATASAGE_PUBLIC_DEMO") == "1"
PUBLIC_LIMIT_BYTES = 2 * 1024 * 1024
PUBLIC_MAX_ROWS = 20000
PUBLIC_MAX_COLUMNS = 100


def analyze_upload(uploaded_bytes: bytes, filename: str, target_column: str | None = None,
                   prediction_context: str | None = None, investigation_mode: str = "rules") -> dict:
    """Analyze uploaded bytes using existing read-only workflow."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError("Unsupported file format")
    if len(uploaded_bytes) > MAX_BYTES:
        raise ValueError("File exceeds 50 MiB limit")
    if PUBLIC_DEMO and (suffix != ".csv" or len(uploaded_bytes) > PUBLIC_LIMIT_BYTES):
        raise ValueError("Public demo accepts CSV files up to 2 MiB")
    with tempfile.TemporaryDirectory(prefix="datasage-") as folder:
        path = Path(folder) / ("upload" + suffix)
        path.write_bytes(uploaded_bytes)
        frame, _ = load_dataset(path)
        if PUBLIC_DEMO and (len(frame) > PUBLIC_MAX_ROWS or len(frame.columns) > PUBLIC_MAX_COLUMNS):
            raise ValueError("Public demo limit: 20,000 rows and 100 columns")
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


from html import escape as escape_html

from datasage.ui_theme import hero, install, sidebar


def make_sample_csv() -> bytes:
    """Small deterministic synthetic dataset; never presented as real customer data."""
    data = pd.DataFrame({
        "customer_id": [f"C-{i:03d}" for i in range(1, 81)],
        "segment": ["retail" if i % 5 else "enterprise" for i in range(80)],
        "monthly_spend": [None if i % 11 == 0 else (59 + i * 4 if i < 72 else 1100 + i * 9)
                          for i in range(80)],
        "days_active": [i * 2 % 91 for i in range(80)],
        "churn": [int(i % 7 == 0) for i in range(80)],
        "churn_proxy": [int(i % 7 == 0) for i in range(80)],
    })
    data.loc[79, ["customer_id", "segment", "monthly_spend", "days_active",
                  "churn", "churn_proxy"]] = data.loc[
        78, ["customer_id", "segment", "monthly_spend", "days_active",
             "churn", "churn_proxy"]
    ].to_numpy()
    return data.to_csv(index=False).encode("utf-8")


def _heading(st, kicker: str, title: str, description: str = "") -> None:
    st.markdown(
        '<div class="ds-section"><div>'
        f'<div class="ds-kicker">{escape_html(kicker)}</div>'
        f'<h2>{escape_html(title)}</h2>'
        f'<p>{escape_html(description)}</p></div></div>',
        unsafe_allow_html=True,
    )


def _footnote(st) -> None:
    st.markdown(
        '<div class="ds-foot">DataSage Studio · Evidence before explanations · '
        'Read-only dataset analysis</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="DataSage Studio · Dataset Intelligence",
        page_icon="🧬", layout="wide", initial_sidebar_state="expanded"
    )
    install(st)
    sidebar(st, PUBLIC_DEMO)
    hero(st)

    left, right = st.columns([1.7, 1], gap="large")
    with left:
        _heading(st, "01 / DATA SOURCE", "Bring your dataset",
                 "Import a file or try our ready-made synthetic example.")
        uploaded = st.file_uploader(
            "Choose a CSV file" if PUBLIC_DEMO else "Choose CSV, XLSX or Parquet",
            type=["csv"] if PUBLIC_DEMO else ["csv", "xlsx", "parquet"],
            help="Public demo accepts non-sensitive CSV files up to 2 MiB." if PUBLIC_DEMO
                 else "Local version accepts CSV, Excel, or Parquet files up to 50 MiB.",
        )
        demo_col, _ = st.columns([1, 1])
        with demo_col:
            if st.button("✦  Explore sample dataset", use_container_width=True):
                st.session_state["ds_demo_active"] = True
    with right:
        _heading(st, "QUICK TOUR", "From raw data to clarity",
                 "A practical workflow, without black-box conclusions.")
        st.markdown(
            '<div class="ds-note">'
            '<strong>01</strong> · Preview your data and choose a target.<br>'
            '<strong>02</strong> · Launch parallel diagnostic agents.<br>'
            '<strong>03</strong> · Check findings against their evidence.<br>'
            '<strong>04</strong> · Explore adaptive follow-up investigations.'
            '</div>',
            unsafe_allow_html=True,
        )
        if PUBLIC_DEMO:
            st.caption("PUBLIC DEMO · 2 MiB / 20k rows / 100 columns · Non-sensitive data only")
        else:
            st.caption("LOCAL STUDIO · 50 MiB file limit · Optional local history")

    if uploaded is not None:
        raw, filename = uploaded.getvalue(), uploaded.name
        using_sample = False
    elif st.session_state.get("ds_demo_active"):
        raw, filename = make_sample_csv(), "datasage_synthetic_customers.csv"
        using_sample = True
    else:
        st.info("Choose a dataset above, or open the synthetic sample to experience DataSage.")
        _footnote(st)
        return

    if using_sample:
        st.success("Synthetic customer dataset loaded. These records are artificial and for demonstration only.")
        if st.button("Clear sample"):
            st.session_state["ds_demo_active"] = False
            st.session_state.pop("datasage_report", None)
            st.session_state.pop("datasage_file", None)
            st.rerun()

    if len(raw) > MAX_BYTES or (PUBLIC_DEMO and len(raw) > PUBLIC_LIMIT_BYTES):
        st.error("File exceeds the public 2 MiB limit." if PUBLIC_DEMO
                 else "File exceeds the 50 MiB limit.")
        return

    try:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED:
            raise ValueError("Unsupported file format")
        with tempfile.TemporaryDirectory(prefix="datasage-preview-") as folder:
            path = Path(folder) / ("preview" + suffix)
            path.write_bytes(raw)
            preview, metadata = load_dataset(path)
            if PUBLIC_DEMO and (
                len(preview) > PUBLIC_MAX_ROWS or len(preview.columns) > PUBLIC_MAX_COLUMNS
            ):
                raise ValueError("Public demo limit: 20,000 rows and 100 columns")
            columns = list(preview.columns)
    except (ValueError, OSError, UnicodeError, ImportError, KeyError) as exc:
        st.error(f"Unable to read dataset: {exc}")
        return

    _heading(st, "02 / UNDERSTAND", "A first look",
             "Inspect the shape and schema before running a full analysis.")
    metrics = st.columns(4, gap="medium")
    metrics[0].metric("Rows", f"{metadata.rows:,}")
    metrics[1].metric("Columns", f"{metadata.columns:,}")
    metrics[2].metric("Missing cells", f"{int(preview.isna().sum().sum()):,}")
    metrics[3].metric("Dataset", "SAMPLE" if using_sample else suffix.lstrip(".").upper())
    with st.expander("▤  Preview first 15 rows and column types", expanded=True):
        st.dataframe(preview.head(15), use_container_width=True, hide_index=True)
        st.caption("Source: " + filename + "  ·  Read-only preview")

    _heading(st, "03 / CONFIGURATION", "Tell us what to investigate",
             "Target selection adds conservative model-risk checks.")
    target_col, planner_col = st.columns([1, 1], gap="large")
    with target_col:
        target = st.selectbox(
            "Prediction target (optional)", [None] + columns,
            format_func=lambda v: "No target selected" if v is None else str(v),
        )
    with planner_col:
        planner_label = st.selectbox(
            "Investigation supervisor",
            ["Evidence-driven (local, no API)"] if PUBLIC_DEMO else
            ["Evidence-driven (local, no API)", "LLM-guided (opt-in API)"],
        )
    investigation_mode = "llm" if planner_label.startswith("LLM") else "rules"
    if investigation_mode == "llm" and not os.getenv("OPENAI_API_KEY"):
        st.warning("LLM planning requires OPENAI_API_KEY. Use local mode without credentials.")

    with st.expander("⚙  Advanced · Prediction context and source evidence"):
        context = st.text_area(
            "Prediction-time context (optional)",
            placeholder="Describe the prediction context, or paste JSON with prediction_time_column, feature_time_columns, and missing_reason_columns.",
            help="Recorded timestamps and collection reasons still require independent verification.",
            height=105,
        )
        st.caption("Free-form context is recorded; structured JSON enables specific provenance checks.")
    run_col, note_col = st.columns([1, 1.5], gap="large")
    with run_col:
        run_clicked = st.button("↗  Run DataSage analysis", type="primary", use_container_width=True)
    with note_col:
        st.caption("Read-only · No automatic data repair · No causal or ML-readiness certificate")

    fingerprint = (filename, metadata.fingerprint, target, context, investigation_mode)
    if run_clicked:
        try:
            with st.spinner("Coordinating agents and validating evidence…"):
                report = analyze_upload(raw, filename, target, context, investigation_mode)
            st.session_state["datasage_report"] = report
            st.session_state["datasage_file"] = fingerprint
            history = st.session_state.setdefault("datasage_history", [])
            history.insert(0, {
                "file": filename, "fingerprint": metadata.fingerprint,
                "target": target or "(No target)", "summary": dict(report["summary"]),
                "report": report,
            })
            del history[10:]
        except (ValueError, OSError, UnicodeError, KeyError, TypeError, RuntimeError) as exc:
            st.error(f"Analysis failed: {exc}")
            return

    if st.session_state.get("datasage_file") != fingerprint:
        st.info("Ready to scan. Choose your settings and select **Run DataSage analysis**.")
        _footnote(st)
        return
    report = st.session_state.get("datasage_report")
    if report is None:
        return

    _heading(st, "04 / ANALYSIS", "Your evidence workspace",
             "Explore calculated findings, follow-up investigations, and downloadable reports.")
    values = report["summary"]
    findings = [f for group in report["findings"].values() for f in group]
    top_metrics = st.columns(4, gap="medium")
    top_metrics[0].metric("Observations", len(findings))
    top_metrics[1].metric("Verified calculations", values.get("verified", 0))
    top_metrics[2].metric("Not verified", values.get("not_verified", 0))
    top_metrics[3].metric("Follow-up checks", len(report.get("follow_up_investigations", [])))
    st.markdown(
        '<div class="ds-note"><strong>Evidence ≠ diagnosis.</strong> '
        'A verified result means the supporting calculation passed independent checking. '
        'It does not prove a data-quality defect, target leakage, unfairness, or ML readiness.</div>',
        unsafe_allow_html=True,
    )

    overview_tab, findings_tab, followup_tab, history_tab, export_tab = st.tabs([
        "◈ Overview", "◎ Findings", "↗ Investigations", "◷ History", "⇩ Exports"
    ])

    with overview_tab:
        chart_col, summary_col = st.columns([1.35, 1], gap="large")
        with chart_col:
            st.markdown("#### Observation mix")
            counts = {}
            for finding in findings:
                category = finding["category"].replace("_", " ").title()
                counts[category] = counts.get(category, 0) + 1
            if counts:
                chart = pd.DataFrame({
                    "Finding type": list(counts), "Count": list(counts.values())
                }).set_index("Finding type")
                st.bar_chart(chart, use_container_width=True, height=320)
            else:
                st.info("No candidate findings surfaced under these specific checks.")
        with summary_col:
            st.markdown("#### Analysis summary")
            st.write(overall_summary(report))
            st.markdown("**How findings are checked**")
            st.write("Five specialized agents run diagnostic calculations. Independent validation recomputes supported evidence, and follow-up checks investigate available provenance.")
            st.caption("Not every statistical observation requires a cleaning action.")
        st.markdown("#### Dataset profile")
        st.dataframe(
            pd.DataFrame({
                "Column": [str(c) for c in preview.columns],
                "Type": [str(t) for t in preview.dtypes],
                "Missing": [int(preview[c].isna().sum()) for c in preview.columns],
            }),
            use_container_width=True, hide_index=True, height=290,
        )

    with findings_tab:
        st.markdown("#### Diagnostic findings")
        status_col, severity_col = st.columns(2, gap="large")
        with status_col:
            statuses = st.multiselect(
                "Validation status", ["verified", "rejected", "not_verified"],
                default=["verified", "not_verified"]
            )
        with severity_col:
            severities = st.multiselect(
                "Severity", ["low", "medium", "high", "critical"],
                default=["low", "medium", "high", "critical"],
            )
        selected = filter_findings(report, statuses, severities)
        st.caption(f"Showing {len(selected)} of {len(findings)} observations")
        if not selected:
            st.info("No findings match these filters.")
        for item in selected:
            status = item["validation_status"]
            with st.expander(
                f"◉  {item['title']}   ·   {item.get('column') or 'Dataset'}   ·   {status.replace('_',' ')}"
            ):
                st.caption(
                    f"Severity: {item['severity'].upper()} · "
                    f"Evidence confidence: {item['confidence']} · Status: {status}"
                )
                explanation = explain_finding(item)
                left_e, right_e = st.columns([1, 1], gap="large")
                with left_e:
                    st.markdown("**What we observed**")
                    st.write(explanation["observation"])
                    st.markdown("**What this might mean**")
                    st.write(explanation["meaning"])
                with right_e:
                    st.markdown("**Suggested next check**")
                    st.write(explanation["next_step"])
                    st.markdown("**Validation**")
                    st.write(item["validation_reason"])
                if not PUBLIC_DEMO and os.getenv("OPENAI_API_KEY"):
                    key = f"llm-{metadata.fingerprint}-{target}-{item['finding_id']}"
                    if st.button("✦  Generate optional AI explanation", key=key):
                        try:
                            st.session_state[f"text-{key}"] = explain_with_llm(item)
                        except (ValueError, ImportError, RuntimeError, OSError) as exc:
                            st.error(f"AI explanation unavailable: {exc}")
                    if st.session_state.get(f"text-{key}"):
                        st.info("AI-generated commentary (not verified evidence): " + st.session_state[f"text-{key}"])
                with st.expander("View source metrics and limitations"):
                    st.json(item["evidence"])
                    st.write("Limitations:", item.get("limitations", []))

    with followup_tab:
        st.markdown("#### Adaptive investigations")
        plan = report.get("follow_up_plan", [])
        planner = report.get("follow_up_planner", {})
        if plan:
            st.caption("Supervisor: " + planner.get("source", "evidence_rules"))
            st.write(planner.get("reason", ""))
            for item in report.get("follow_up_investigations", []):
                with st.expander(
                    item["check"].replace("_", " ").title() + "   ·   " +
                    item["status"].replace("_", " ")
                ):
                    st.markdown("**What the evidence supports**")
                    st.write(item["interpretation"])
                    st.markdown("**What to verify next**")
                    st.write(item["next_step"])
                    st.markdown("**Recorded observations**")
                    st.json(item["observations"])
        else:
            st.info("No targeted follow-up checks were eligible. Provide source logs or prediction-time metadata to enable deeper checks.")
        st.caption(
            "Temporal comparisons and recorded missing-value reasons cannot alone establish leakage or the causal missingness mechanism."
        )

    with history_tab:
        st.markdown("#### Recent analyses")
        st.caption("The last 10 reports stay in the current browser session only.")
        for index, entry in enumerate(st.session_state.get("datasage_history", [])):
            with st.expander(
                f"Run {index + 1} · {entry['file']} · Target: {entry['target']}"
            ):
                st.write(overall_summary(entry["report"]))
                st.caption(f"Source fingerprint: {entry['fingerprint'][:16]}…")
        if not PUBLIC_DEMO:
            st.markdown("#### Optional local report storage")
            st.caption("Stores findings and metadata on this computer, not raw uploaded rows.")
            if st.button("Save current report locally"):
                try:
                    save_investigation(report, filename, target)
                    st.success("Report saved locally.")
                except (ValueError, OSError, TypeError) as exc:
                    st.error(f"Save failed: {exc}")
            if st.checkbox("Browse saved reports"):
                for saved in list_investigations():
                    with st.expander(
                        f"{saved['saved_at_utc'][:16]} · {saved['source_filename']} · "
                        f"{saved['target'] or '(No target)'}"
                    ):
                        st.write(overall_summary(saved["report"]))
                        st.download_button(
                            "Download saved JSON",
                            data=json.dumps(saved["report"], indent=2, ensure_ascii=False),
                            file_name=f"datasage-saved-{saved['id']}.json",
                            mime="application/json", key=f"saved-{saved['id']}",
                        )

    with export_tab:
        st.markdown("#### Export reproducible evidence")
        st.write("Keep the calculations, limitations, and follow-up observations with your research notes.")
        json_col, html_col = st.columns(2, gap="large")
        with json_col:
            st.download_button(
                "↓  Download full JSON report",
                data=json.dumps(report, indent=2, ensure_ascii=False),
                file_name="datasage-report.json", mime="application/json",
                use_container_width=True,
            )
            st.caption("Structured metrics and validation details.")
        with html_col:
            st.download_button(
                "↓  Download readable HTML report",
                data=render_html(report), file_name="datasage-report.html",
                mime="text/html", use_container_width=True,
            )
            st.caption("Human-readable, static diagnostic summary.")
    _footnote(st)


if __name__ == "__main__":
    main()
