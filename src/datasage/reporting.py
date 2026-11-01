"""Evidence-first reconciliation and JSON/HTML reporting."""
from __future__ import annotations

from collections import Counter
from html import escape
import json
from pathlib import Path
from typing import Any

from datasage.models import Finding
from datasage.explanations import explain_finding, overall_summary

STATUSES = ("verified", "rejected", "not_verified")


def reconcile(findings: list[Finding], validation_results: list[dict[str, str]],
              metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Join by finding ID, retaining conflicts instead of silently discarding them."""
    by_id: dict[str, list[dict[str, str]]] = {}
    for outcome in validation_results:
        by_id.setdefault(outcome.get("finding_id", ""), []).append(outcome)
    id_counts = Counter(f.finding_id for f in findings)
    groups: dict[str, list[dict[str, Any]]] = {k: [] for k in STATUSES}
    for finding in findings:
        matches = by_id.get(finding.finding_id, [])
        if id_counts[finding.finding_id] > 1:
            status, reason = "rejected", "Conflicting duplicate finding ID"
        elif len(matches) != 1:
            status, reason = "not_verified", "Missing or ambiguous validation decision"
        elif matches[0].get("status") not in STATUSES:
            status, reason = "not_verified", "Unknown validation status"
        else:
            status = matches[0]["status"]
            reason = matches[0].get("reason", "")
        entry = finding.model_dump(mode="json")
        entry["validation_status"] = status
        entry["validation_reason"] = reason
        groups[status].append(entry)
    unknown = sorted(set(by_id) - set(id_counts))
    return {
        "report_version": "1.0",
        "dataset_metadata": metadata or {},
        "summary": {k: len(v) for k, v in groups.items()},
        "findings": groups,
        "orphan_validation_ids": unknown,
        "limitations": [
            "Verified means supporting numerical evidence was checked, not that an issue is causally established.",
            "Unverified and rejected findings must not be promoted to confirmed data-quality problems.",
            "ML readiness is not certified by this report.",
        ],
    }


def write_json_report(report: dict[str, Any], destination: str | Path) -> Path:
    target = Path(destination)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return target


def render_html(report: dict[str, Any]) -> str:
    """Render report as static HTML with escaped untrusted dataset content."""
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<title>DataSage diagnostic report</title>",
             "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:3rem auto;line-height:1.6;padding:0 1.25rem;color:#183049}h1{color:#173c68}li{background:#f4f7fb;border-radius:12px;padding:16px;margin:12px 0}small{color:#556}h2{margin-top:2rem}</style>",
             "</head><body><h1>DataSage · Diagnostic Report</h1>",
             "<p><strong>Interpretation:</strong> evidence-backed observations, not certified dataset defects.</p>"]
    summary = report["summary"]
    parts.append("<p>" + " · ".join(
        f"{escape(key)}: {int(summary[key])}" for key in STATUSES) + "</p>")
    parts.append(f"<p>{escape(overall_summary(report))}</p>")
    for status in STATUSES:
        parts.append(f"<h2>{escape(status.replace('_', ' ').title())}</h2><ul>")
        for item in report["findings"][status]:
            title = escape(str(item["title"]))
            column = escape(str(item.get("column") or "dataset"))
            reason = escape(str(item["validation_reason"]))
            summary = explain_finding(item)
            parts.append(
                f"<li><strong>{title}</strong> <small>({column})</small>"
                f"<p><b>What we observed:</b> {escape(summary['observation'])}</p>"
                f"<p><b>What it means:</b> {escape(summary['meaning'])}</p>"
                f"<p><b>What to check next:</b> {escape(summary['next_step'])}</p>"
                f"<small>Evidence validation: {reason}</small></li>"
            )
        parts.append("</ul>")
    parts.append("<h2>Limitations</h2><ul>")
    for limitation in report["limitations"]:
        parts.append(f"<li>{escape(str(limitation))}</li>")
    parts.append("</ul></body></html>")
    return "".join(parts)
