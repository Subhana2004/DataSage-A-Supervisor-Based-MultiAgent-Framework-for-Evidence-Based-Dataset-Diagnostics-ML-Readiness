import pandas as pd
import pytest
from datasage.ml_risk import ml_risk_findings
from datasage.validation import validate_findings
from datasage.workflow import build_workflow


def sample():
    return pd.DataFrame({"customer_id": [f"C{i}" for i in range(40)],
                         "outcome": [i % 2 for i in range(40)],
                         "outcome_copy": [i % 2 for i in range(40)],
                         "noise": [1] * 40})


def test_identifies_candidates_and_verifies_evidence():
    df = sample()
    findings = ml_risk_findings(df, "outcome")
    assert {f.category for f in findings} == {"identifier_candidate", "target_proxy_candidate"}
    assert all(x["status"] == "verified" for x in validate_findings(df, findings))


def test_tampered_agreement_rejected():
    df = sample()
    f = next(f for f in ml_risk_findings(df, "outcome") if f.category == "target_proxy_candidate")
    f.evidence.metrics["agreement"] = 0.5
    assert validate_findings(df, [f])[0]["status"] == "rejected"


def test_no_target_no_workflow_risk(tmp_path):
    path = tmp_path / "sample.csv"
    sample().to_csv(path, index=False)
    result = build_workflow().invoke({"dataset_path": str(path)})
    assert "ml_risk" not in result["completed_agents"]


def test_with_target_workflow_runs_risk(tmp_path):
    path = tmp_path / "sample.csv"
    sample().to_csv(path, index=False)
    result = build_workflow().invoke({"dataset_path": str(path), "target_column": "outcome"})
    assert "ml_risk" in result["completed_agents"]
    assert any(f["category"] == "target_proxy_candidate" for f in result["findings"])


def test_invalid_target_rejected():
    with pytest.raises(ValueError, match="Unknown target"):
        ml_risk_findings(sample(), "unknown")


def test_small_dataset_skipped():
    assert ml_risk_findings(sample().iloc[:10], "outcome") == []
