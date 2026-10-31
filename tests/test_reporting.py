import json
from datasage.models import Evidence, Finding
from datasage.reporting import reconcile, render_html, write_json_report
from datasage.workflow import build_workflow
import pandas as pd


def finding(fid="A", title="<script>alert(1)</script>"):
    return Finding(
        finding_id=fid, agent="profiling_agent", category="test",
        column="<unsafe>", title=title, description="test",
        evidence=Evidence(method="not_supported", metrics={"n": 1}, sample_size=1),
        severity="low", confidence="low")


def test_three_way_reconciliation_and_orphans():
    items = [finding("A"), finding("B"), finding("C")]
    checks = [{"finding_id": "A", "status": "verified", "reason": "checked"},
              {"finding_id": "B", "status": "rejected", "reason": "bad"},
              {"finding_id": "C", "status": "not_verified", "reason": "unsupported"},
              {"finding_id": "unused", "status": "verified"}]
    report = reconcile(items, checks)
    assert report["summary"] == {"verified": 1, "rejected": 1, "not_verified": 1}
    assert report["orphan_validation_ids"] == ["unused"]


def test_duplicate_ids_not_silently_accepted():
    report = reconcile([finding(), finding()], [{"finding_id": "A", "status": "verified"}])
    assert report["summary"]["rejected"] == 2


def test_missing_decision_not_verified():
    report = reconcile([finding()], [])
    assert report["summary"]["not_verified"] == 1


def test_html_escapes_user_controlled_content():
    rendered = render_html(reconcile([finding()], []))
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_json_export(tmp_path):
    report = reconcile([finding()], [])
    output = write_json_report(report, tmp_path / "diagnostics.json")
    assert json.loads(output.read_text())["summary"]["not_verified"] == 1


def test_workflow_produces_report(tmp_path):
    path = tmp_path / "small.csv"
    pd.DataFrame({"a": [1, 1], "b": [2, 2]}).to_csv(path, index=False)
    result = build_workflow().invoke({"dataset_path": str(path)})
    assert result["status"] == "report_complete"
    assert result["final_report"]["report_version"] == "1.0"
    assert result["final_report"]["summary"]["not_verified"] >= 1
