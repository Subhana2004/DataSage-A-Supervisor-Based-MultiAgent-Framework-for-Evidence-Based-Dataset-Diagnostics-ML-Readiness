import pandas as pd
import pytest

from datasage.distribution import distribution_findings
from datasage.validation import validate_findings
from datasage.workflow import build_workflow


def test_target_imbalance_verified():
    df = pd.DataFrame({"label": ["rare"] * 10 + ["common"] * 90})
    findings = distribution_findings(df, "label")
    assert len(findings) == 1
    assert findings[0].category == "class_imbalance"
    assert validate_findings(df, findings)[0]["status"] == "verified"


def test_tampered_frequencies_rejected():
    df = pd.DataFrame({"label": ["rare"] * 10 + ["common"] * 90})
    finding = distribution_findings(df, "label")[0]
    finding.evidence.metrics["max_min_ratio"] = 1
    assert validate_findings(df, [finding])[0]["status"] == "rejected"


def test_no_target_no_class_claim():
    df = pd.DataFrame({"group": ["a"] * 90 + ["b"] * 10})
    findings = distribution_findings(df)
    assert findings[0].category == "categorical_representation"


def test_balanced_and_small_data_ignored():
    assert distribution_findings(pd.DataFrame({"label": ["a"] * 20 + ["b"] * 20}), "label") == []
    assert distribution_findings(pd.DataFrame({"label": ["a"] * 3 + ["b"]})) == []


def test_unknown_target_rejected():
    with pytest.raises(ValueError, match="Unknown target"):
        distribution_findings(pd.DataFrame({"x": [1]}), "missing")


def test_workflow_distribution_integration(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"label": ["a"] * 90 + ["b"] * 10}).to_csv(path, index=False)
    result = build_workflow().invoke({"dataset_path": str(path), "target_column": "label"})
    assert "distribution" in result["completed_agents"]
    assert "distribution" in result["investigation_plan"]
    assert any(f["category"] == "class_imbalance" for f in result["findings"])
