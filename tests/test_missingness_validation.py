import pandas as pd

from datasage.missingness import missingness_findings
from datasage.validation import validate_findings
from datasage.workflow import build_workflow


def make_frame():
    return pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "income": [None] * 30 + list(range(20)) + list(range(50)),
    })


def test_missing_counts_and_association():
    frame = make_frame()
    findings = missingness_findings(frame)
    counts = [f for f in findings if f.category == "missing_data"]
    assert len(counts) == 1
    assert counts[0].evidence.metrics["missing_count"] == 30
    assoc = [f for f in findings if f.category == "missingness_association"]
    assert len(assoc) >= 1
    assert assoc[0].evidence.metrics["adjusted_p_value"] < 0.05
    statuses = validate_findings(frame, findings)
    assert all(item["status"] == "verified" for item in statuses)


def test_forged_count_is_rejected():
    frame = make_frame()
    finding = next(f for f in missingness_findings(frame) if f.category == "missing_data")
    finding.evidence.metrics["missing_count"] = 999
    assert validate_findings(frame, [finding])[0]["status"] == "rejected"


def test_unsupported_method_is_not_verified():
    frame = make_frame()
    finding = missingness_findings(frame)[0]
    finding.evidence.method = "invented"
    assert validate_findings(frame, [finding])[0]["status"] == "not_verified"


def test_no_missing_values_has_no_missingness_findings():
    assert missingness_findings(pd.DataFrame({"x": [1, 2, 3]})) == []


def test_small_contingency_suppressed():
    frame = pd.DataFrame({"group": ["A", "A", "B", "B"], "value": [None, 1, None, 2]})
    assert not [f for f in missingness_findings(frame) if f.category == "missingness_association"]


def test_workflow_includes_validation(tmp_path):
    path = tmp_path / "sample.csv"
    make_frame().to_csv(path, index=False)
    result = build_workflow().invoke({"dataset_path": str(path)})
    assert result["status"] == "report_complete"
    assert "missingness" in result["completed_agents"]
    assert "validation" in result["completed_agents"]
    assert any(x["status"] == "verified" for x in result["validation_results"])
