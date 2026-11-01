import pytest
from datasage.blinded_review import prepare_blinded_review, adjudicate_reviews, score_review


def sample():
    return {"findings": {
        "verified": [{"category": "duplicate_rows", "column": None, "description": "Exact duplicates",
                       "evidence": {"metrics": {"duplicate_rows": 2}}}],
        "rejected": [], "not_verified": [],
    }}


def test_source_blinded_and_sealed_separately():
    cases, sealed = prepare_blinded_review(sample(), {("missing_data", "salary")}, "dataset-v1")
    assert len(cases) == 2
    assert all("datasage" not in case and "baseline" not in case for case in cases)
    assert set(sealed) == {case["case_id"] for case in cases}
    assert sum(int(v["datasage"]) for v in sealed.values()) == 1


def test_agreement_required_and_scores():
    cases, sealed = prepare_blinded_review(sample(), {("missing_data", "salary")}, "v1")
    labels = {c["case_id"]: ("actionable" if c["category"] == "duplicate_rows" else "not_actionable") for c in cases}
    consensus = adjudicate_reviews(labels, labels, set(labels))
    ds = score_review(sealed, consensus, "datasage")
    base = score_review(sealed, consensus, "baseline")
    assert ds["tp"] == 1 and ds["fp"] == 0 and ds["recall"] == 1.0
    assert base["fp"] == 1 and base["fn"] == 1


def test_disagreement_not_treated_as_ground_truth():
    cases, sealed = prepare_blinded_review(sample(), set(), "v1")
    case_id = cases[0]["case_id"]
    labels = adjudicate_reviews({case_id: "actionable"}, {case_id: "not_actionable"}, set(sealed))
    assert labels[case_id] == "unresolved"
    assert score_review(sealed, labels, "datasage")["resolved_cases"] == 0


def test_unknown_case_rejected():
    with pytest.raises(ValueError):
        adjudicate_reviews({"fake": "actionable"}, {}, set())
