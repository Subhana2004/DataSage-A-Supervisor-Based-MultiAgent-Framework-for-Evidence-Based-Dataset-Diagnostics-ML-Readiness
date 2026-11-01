from datasage.explanations import explain_finding, overall_summary


def test_duplicates_readable():
    item = {"category": "duplicate_rows", "evidence": {"metrics": {"duplicate_rows": 22, "percentage": 0.31}}}
    result = explain_finding(item)
    assert "22 rows" in result["observation"]
    assert "not necessarily" in result["meaning"]


def test_missingness_human_readable():
    item = {"category": "missing_data", "column": "salary", "evidence": {"metrics": {"missing_count": 5, "missing_percentage": 25.0}}}
    result = explain_finding(item)
    assert "25.0%" in result["observation"]
    assert "salary" in result["observation"]


def test_categories_no_fabricated_causality():
    item = {"category": "class_imbalance", "column": "PhoneService",
            "evidence": {"metrics": {"counts": {"Yes": 90, "No": 10}}}}
    result = explain_finding(item)
    assert "Yes: 90" in result["observation"]
    assert "not by themselves" in result["meaning"]


def test_summary_distinguishes_evidence_from_harm():
    result = overall_summary({"summary": {"verified": 3, "rejected": 0, "not_verified": 1}})
    assert "3 numerically verified" in result
    assert "not that a harmful data problem was proven" in result
