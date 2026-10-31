"""Evidence-first reconciliation and JSON/HTML reporting."""
from __future__ import annotations

from collections import Counter
from html import escape
import json
from pathlib import Path
from typing import Any

from datasage.models import Finding

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
             "<title>DataSage diagnostic report</title></head><body>",
             "<h1>DataSage diagnostic report</h1>"]
    summary = report["summary"]
    parts.append("<p>" + " · ".join(
        f"{escape(key)}: {int(summary[key])}" for key in STATUSES) + "</p>")
    for status in STATUSES:
        parts.append(f"<h2>{escape(status.replace('_', ' ').title())}</h2><ul>")
        for item in report["findings"][status]:
            title = escape(str(item["title"]))
            column = escape(str(item.get("column") or "dataset"))
            reason = escape(str(item["validation_reason"]))
            parts.append(f"<li><strong>{title}</strong> ({column}): {reason}</li>")
        parts.append("</ul>")
    parts.append("<h2>Limitations</h2><ul>")
    for limitation in report["limitations"]:
        parts.append(f"<li>{escape(str(limitation))}</li>")
    parts.append("</ul></body></html>")
    return "".join(parts)
