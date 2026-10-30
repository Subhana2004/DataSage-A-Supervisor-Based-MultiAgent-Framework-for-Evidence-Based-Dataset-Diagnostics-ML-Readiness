import pandas as pd

from datasage.anomaly import anomaly_findings, outlier_summary
from datasage.validation import validate_findings


def sample():
    return pd.DataFrame({"metric": list(range(30)) + [1000], "flag": [True] * 31})


def test_detect_outlier_and_exclude_boolean():
    frame = sample()
    findings = anomaly_findings(frame)
    assert len(findings) == 1
    assert findings[0].column == "metric"
    assert findings[0].evidence.metrics["count"] == 1
    assert validate_findings(frame, findings)[0]["status"] == "verified"


def test_tampered_evidence_rejected():
    frame = sample()
    finding = anomaly_findings(frame)[0]
    finding.evidence.metrics["count"] = 999
    assert validate_findings(frame, [finding])[0]["status"] == "rejected"


def test_small_and_constant_columns_skipped():
    assert anomaly_findings(pd.DataFrame({"x": [1, 2, 100]})) == []
    assert outlier_summary(pd.Series([1] * 30)) is None


def test_input_unchanged():
    frame = sample()
    before = frame.copy(deep=True)
    anomaly_findings(frame)
    pd.testing.assert_frame_equal(frame, before)
