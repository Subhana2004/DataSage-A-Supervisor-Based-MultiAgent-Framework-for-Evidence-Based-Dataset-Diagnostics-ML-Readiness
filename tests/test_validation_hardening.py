import pandas as pd
from datasage.profiling import profile_dataset, profile_findings
from datasage.missingness import missingness_findings
from datasage.validation import validate_findings


def test_duplicate_and_constant_are_verified():
    frame = pd.DataFrame({"constant": [4] * 30, "other": [1] * 15 + [2] * 15})
    findings = profile_findings(profile_dataset(frame))
    assert {"duplicate_rows", "constant_feature"} <= {f.category for f in findings}
    assert all(x["status"] == "verified" for x in validate_findings(frame, findings))


def test_tampering_with_duplicate_evidence_fails():
    frame = pd.DataFrame({"x": [1] * 30})
    finding = next(f for f in profile_findings(profile_dataset(frame)) if f.category == "duplicate_rows")
    finding.evidence.metrics["duplicate_rows"] = 1000
    assert validate_findings(frame, [finding])[0]["status"] == "rejected"


def test_tampering_with_constant_evidence_fails():
    frame = pd.DataFrame({"x": [1] * 30})
    finding = next(f for f in profile_findings(profile_dataset(frame)) if f.category == "constant_feature")
    finding.evidence.metrics["unique_non_null"] = 3
    assert validate_findings(frame, [finding])[0]["status"] == "rejected"


def test_bh_adjustment_tamper_fails_even_when_in_valid_p_interval():
    frame = pd.DataFrame({"group": ["A"] * 50 + ["B"] * 50,
                          "income": [None] * 30 + list(range(20)) + list(range(50))})
    finding = next(f for f in missingness_findings(frame) if f.category == "missingness_association")
    assert validate_findings(frame, [finding])[0]["status"] == "verified"
    p = finding.evidence.metrics["p_value"]
    finding.evidence.metrics["adjusted_p_value"] = min(1.0, p + 0.1)
    assert validate_findings(frame, [finding])[0]["status"] == "rejected"
